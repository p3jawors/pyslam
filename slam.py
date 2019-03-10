"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import time
import cv2
from enum import Enum

from frame import Frame, match_frames

import optimizer_g2o

from map_point import MapPoint
from map import Map
from geom_helpers import triangulate, add_ones, poseRt
from pinhole_camera import Camera, PinholeCamera
from initializer import Initializer
from timer import TimerFps

from helpers import Printer

import constants 

kVerbose=True     
kRansacThresholdNormalized = 0.0003  # 0.0003 # metric threshold used for normalized image coordinates 
kRansacProb = 0.999
kNumMinInliersEssentialMat = 8
kUseGroundTruthScale = False 
kLocalWindow = constants.kLocalWindow
kUseMotionModel = constants.kUseMotionModel
kUseLargeWindowBA = constants.kUseLargeWindowBA
kTimerVerbose = False 

class SlamStage(Enum):
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    LOST=3

class SLAM(object):
    def __init__(self, camera, tracker, grountruth = None):
        self.cam = camera 

        self.map = Map()

        Frame.set_tracker(tracker) # set the static field of the class 

        # camera info 
        self.W, self.H = camera.width, camera.height 
        self.K = camera.K
        self.Kinv = camera.Kinv
        self.D = camera.D        # distortion coefficients [k1, k2, p1, p2, k3]

        self.stage = SlamStage.NO_IMAGES_YET

        self.intializer = Initializer()

        self.cur_R = None # current rotation w.r.t. world frame 
        self.cur_t = None # current translation w.r.t. world frame 

        self.num_matched_kps = None      # current number of matched keypoints 
        self.num_inliers = None          # current number of matched inliers 
        self.num_vo_map_points = None    # current number of valid VO map points (matched and found valid in current pose optimization)          

        self.trueX, self.trueY, self.trueZ = None, None, None
        self.grountruth = grountruth

        self.mask_match = None 

        self.velocity = None 
 
        self.init_history = True 
        self.poses = []       # history of poses
        self.t0_est = None    # history of estimated translations      
        self.t0_gt = None     # history of ground truth translations (if available)
        self.traj3d_est = []  # history of estimated translations centered w.r.t. first one
        self.traj3d_gt = []   # history of estimated ground truth translations centered w.r.t. first one   

        self.timer_verbose = kTimerVerbose  # set this to True if you want to print timings 
        self.timer_main_track = TimerFps('Track', is_verbose = self.timer_verbose)
        self.timer_pose_opt = TimerFps('Pose optimization', is_verbose = self.timer_verbose)
        self.timer_match = TimerFps('Match', is_verbose = self.timer_verbose)        
        self.timer_pose_est = TimerFps('Pose estimation', is_verbose = self.timer_verbose)
        self.timer_frame = TimerFps('Frame', is_verbose = self.timer_verbose)
        self.timer_seach_map = TimerFps('Search Map', is_verbose = self.timer_verbose)     
        self.timer_triangulation = TimerFps('Triangulation', is_verbose = self.timer_verbose)     
        self.time_local_opt = TimerFps('Local optimization', is_verbose = self.timer_verbose)         


    # fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
    # out: [Rrc, trc]   (with respect to 'ref' frame) 
    # N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with the previous estimated ones)
    # N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
    # - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie a ruled quadric 
    # - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
    # N.B.3: the five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
    def estimate_pose_ess_mat(self, kpn_ref, kpn_cur):	     
        E, self.mask_match = cv2.findEssentialMat(kpn_cur, kpn_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)                         
        _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0., 0.))   
        return poseRt(R,t.T)  # Rrc,trc (cur with respect to 'ref' frame)         

    def track(self, img, frame_id, pose=None, verts=None):

        assert img.shape[0:2] == (self.H, self.W)        
        
        self.timer_main_track.start()

        self.timer_frame.start()        
        f_cur = Frame(self.map, img, self.K, self.Kinv, self.D, des=verts) 
        self.timer_frame.refresh()            

        if self.stage == SlamStage.NO_IMAGES_YET: 
            self.intializer.init(f_cur) # add first frame in the inizializer 
            self.stage = SlamStage.NOT_INITIALIZED
            return # EXIT (jump to second frame)
        
        if self.stage == SlamStage.NOT_INITIALIZED:
            initializer_output, is_ok = self.intializer.initialize(f_cur, img)
            if is_ok:
                f_ref = self.intializer.f_ref
                # add the two initialized frames in the map 
                self.map.add_frame(f_ref) # add first frame in map and update its id
                self.map.add_frame(f_cur) # add second frame in map and update its id
                # add points in map 
                new_pts_count,_ = self.map.add_points(initializer_output.points4d, None, f_cur, f_ref, initializer_output.idx_cur, initializer_output.idx_ref, img)
                Printer.green("map: initialized %d new points" % (new_pts_count))                   
                self.stage = SlamStage.OK               
            return # EXIT (jump to next frame)
        
        f_ref = self.map.frames[-1] # get last frame in map            
        self.map.add_frame(f_cur)   # add f_cur to map 
        
        self.timer_match.start()
        # find image point matches
        idx_cur, idx_ref = match_frames(f_cur, f_ref)
        self.num_matched_kps = idx_cur.shape[0]     
        print('# keypoint matches: ', self.num_matched_kps)
        self.timer_match.refresh()        


        if kUseMotionModel is not True:
            print('estimating pose by fitting essential mat')   
            # N.B.: please, in order to understand the limitations of fitting an essential mat, read the comments of the method self.estimate_pose_ess_mat() 
            self.timer_pose_est.start()
            # estimate inter frame camera motion 
            Mrc = self.estimate_pose_ess_mat(f_ref.kpsn[idx_ref], f_cur.kpsn[idx_cur])
            Mcr = np.linalg.inv(poseRt(Mrc[:3, :3], Mrc[:3, 3]))
            f_cur.pose = np.dot(Mcr, f_ref.pose)
            self.timer_pose_est.refresh()      

            # remove outliers from matches by using the mask computed with inter frame pose estimation 
            mask_index = [ i for i,v in enumerate(self.mask_match) if v > 0] 
            self.num_inliers = len(mask_index)
            print('# inliers: ', self.num_inliers )
            idx_ref = idx_ref[mask_index]
            idx_cur = idx_cur[mask_index]

            if self.num_inliers < kNumMinInliersEssentialMat:
                f_cur.pose = f_ref.pose.copy()  # reset pose estimation to previous frame 
                Printer.red('Essential mat: not enough inliers!')  

            f_cur.pose[:,3] = f_ref.pose[:,3].copy() # keep the estimated rotation and override translation with ref frame translation                      

        else:
            print('using motion model')          
            self.velocity = np.dot(f_ref.pose, np.linalg.inv(self.map.frames[-2].pose))
            predicted_pose = np.dot(self.velocity, f_ref.pose)
            # set intial guess for pose optimization             
            f_cur.pose = predicted_pose.copy()
            #f_cur.pose = f_ref.pose.copy()  # get the last pose as an initial guess for optimization
         
        # update current frame with observed map points: 
        # these corresponds to curr frame keypoints which have matches with {prev frame keypoints having a corresponding map point} 
        # update the observations of these observed map points 
        num_found_map_pts_inter_frame = 0
        for i, idx in enumerate(idx_ref):
            if f_ref.points[idx] is not None: 
                f_ref.points[idx].add_observation(f_cur, idx_cur[i])
                num_found_map_pts_inter_frame += 1
        print("# matched map points in prev frame: %d " % num_found_map_pts_inter_frame)   


        # current pose optimization 1   
        self.timer_pose_opt.start()          
        pose_opt_error, pose_is_ok, self.num_vo_map_points = optimizer_g2o.poseOptimization(f_cur, verbose=False)
        print("pose opt err1: %f,  ok: %d" % (pose_opt_error, int(pose_is_ok)) ) 
        # discard outliers detected in pose optimization (in current frame)
        #f_cur.reset_outlier_map_points()       

        if pose_is_ok is True: 
            # discard outliers detected in pose optimization (in current frame)
            f_cur.reset_outlier_map_points()  
        else:            
            # if current pose optimization failed, reset current pose             
            f_cur.pose = f_ref.pose.copy()                 

        self.timer_pose_opt.pause()                                         
                       

        # TODO: add recover in case of current pose optimization failure 


        # find local map points whose descriptors match with a {keypoint in curr frame that was not matched with any prev frame keypoint)
        if pose_is_ok is True and not self.map.local_map.is_empty():
            self.timer_seach_map.start()
            #num_found_map_pts = self.searchLocalFramesByProjection(f_cur, local_window = kLocalWindow)    
            num_found_map_pts = self.searchByProjection(self.map.local_map.points, f_cur) 
            print("# new matched map points: %d " % num_found_map_pts)               
            self.timer_seach_map.refresh()

            # current pose optimization 2 with all the found map points 
            self.timer_pose_opt.resume()        
            pose_opt_error, pose_is_ok, self.num_vo_map_points = optimizer_g2o.poseOptimization(f_cur, verbose=False)
            print("pose opt err2: %f,  ok: %d" % (pose_opt_error, int(pose_is_ok)) ) 
            print("# valid matched map points: %d " % self.num_vo_map_points)      
            # discard outliers detected in pose optimization (in current frame)
            if pose_is_ok is True:
                f_cur.reset_outlier_map_points()        
            self.timer_pose_opt.refresh()     


        # if pose is ok, then we can try to triangulate the matched keypoints without a corresponding map point 
        if pose_is_ok is True:                 
            self.timer_triangulation.start()

            # TODO: this triangulation should be done from keyframes!
            good_pts4d = np.array([f_cur.points[i] is None for i in idx_cur]) # matched keypoints without a corresponding map point 

            # do triangulation in global frame
            pts4d = triangulate(f_cur.pose, f_ref.pose, f_cur.kpsn[idx_cur], f_ref.kpsn[idx_ref])
            good_pts4d &= np.abs(pts4d[:, 3]) != 0
            #good_pts4d &= f_ref.outliers[idx_ref] == False 
            pts4d /= pts4d[:, 3:]       # get homogeneous 3-D coords

            new_pts_count,_ = self.map.add_points(pts4d, good_pts4d, f_cur, f_ref, idx_cur, idx_ref, img, check_parallax=True)
            print("# added map points: %d " % (new_pts_count))
            self.timer_triangulation.refresh()

        self.time_local_opt.start()
        err = self.map.locally_optimize(local_window=kLocalWindow)
        print("local optimization error:   %f" % err)
        self.time_local_opt.refresh()

        # large window optimization of the map
        # TODO: move this in a seperate thread with local mapping 
        if f_cur.id >= constants.kEveryNumFramesLargeWindowBA and f_cur.id % constants.kEveryNumFramesLargeWindowBA == 0 and kUseLargeWindowBA:
            err = self.map.optimize(local_window=constants.kLargeWindow)  # verbose=True)
            Printer.green("large window optimization error:   %f" % err)

        print("map: %d points, %d frames" % (len(self.map.points), len(self.map.frames)))
        #self.updateHistory()

        self.timer_main_track.refresh()

    # search by projection the map points of local frames on frame f_cur
    def searchLocalFramesByProjection(self, f_cur, local_window = kLocalWindow):
        # TODO: implement a proper local mapping 
        # take the points in the last N frame 
        points = []
        point_id_set = set()
        frames = self.map.frames[-local_window:]
        for f in frames:
            #f_points = [p for p,outlier in zip(f.points,f.outliers) if (p is not None) and (outlier is not True)] 
            f_points = [p for p in f.points if (p is not None)]
            for p in f_points: 
                if p.id not in point_id_set:
                    points.append(p)
                    point_id_set.add(p.id)
        print('searching %d map points' % len(points))
        return self.searchByProjection(points, f_cur)  

    # search by projection all the map points on frame f_cur
    def searchMapByProjection(self, f_cur):
        return self.searchByProjection(self.map.points, f_cur)                

    # search by projection the input map points on frame f_cur
    def searchByProjection(self, points, f_cur):
        found_pts_count = 0
        if len(points) > 0:

            # project the points on frame f_cur
            projs = f_cur.project_map_points(points)

            # check if points are visible 
            visible_pts = (projs[:, 0] > 0) & (projs[:, 0] < f_cur.W) & \
                          (projs[:, 1] > 0) & (projs[:, 1] < f_cur.H)

            for i, p in enumerate(points):
                if not visible_pts[i] or p.is_bad is True:
                    # point not visible in frame or is bad 
                    continue
                if f_cur in p.frames:
                    # we already matched this map point to this frame
                    continue
                for m_idx in f_cur.kd.query_ball_point(projs[i], constants.kMaxReprojectionDistance):
                    # if no point associated 
                    if f_cur.points[m_idx] is None: 
                        orb_dist = p.orb_distance(f_cur.des[m_idx])
                        #print('b_dist : ', orb_dist)
                        if orb_dist < constants.kMaxOrbDistanceSearchByReproj:
                            p.add_observation(f_cur, m_idx)
                            found_pts_count += 1
                            break      
        return found_pts_count;            

    # get current translation scale from ground-truth if this is set 
    def getAbsoluteScale(self, frame_id):  
        if self.grountruth is not None and kUseGroundTruthScale:
            self.trueX, self.trueY, self.trueZ, scale = self.grountruth.getPoseAndAbsoluteScale(frame_id)
            return scale
        else:
            self.trueX = 0 
            self.trueY = 0 
            self.trueZ = 0
            return 1

    def updateHistory(self):
        f_cur = self.map.frames[-1]
        self.cur_R = f_cur.pose[:3,:3].T
        self.cur_t = np.dot(-self.cur_R,f_cur.pose[:3,3])
        if (self.init_history is True) and (self.trueX is not None):
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
            self.t0_gt  = np.array([self.trueX, self.trueY, self.trueZ])           # starting translation 
        if (self.t0_est is not None) and (self.t0_gt is not None):             
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            self.traj3d_est.append(p)
            self.traj3d_gt.append([self.trueX-self.t0_gt[0], self.trueY-self.t0_gt[1], self.trueZ-self.t0_gt[2]])            
            self.poses.append(poseRt(self.cur_R, p))    
