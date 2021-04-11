# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 14:20
# @Author  : xylon
import cv2
import torch
import random
import argparse
import numpy as np

from rfnet.utils.common_utils import gct
from rfnet.utils.eval_utils import nearest_neighbor_distance_ratio_match
from rfnet.model.rf_des import HardNetNeiMask
from rfnet.model.rf_det_so import RFDetSO
from rfnet.model.rf_net_so import RFNetSO
from rfnet.config import cfg

# if __name__ == "__main__":
class RfnetFeature:
    def __init__(self,
                 num_features=2000,
                 nms_window_size=5,      # NMS windows size
                 desc_dim=128,           # descriptor dimension. Needs to match the checkpoint value
                 mode = 'nms',           # choices=['nms', 'rng'], Whether to extract features using the non-maxima suppresion mode or through training-time grid sampling technique'
                 do_cuda=True):
        print('Using RfnetFeature')
 
        # parser = argparse.ArgumentParser(description="example")
        # parser.add_argument("--imgpath", default=None, type=str)  # image path
        # parser.add_argument("--resume", default=None, type=str)  # model path
        # args = parser.parse_args()

        print(f"{gct()} : start time")

        random.seed(cfg.PROJ.SEED)
        torch.manual_seed(cfg.PROJ.SEED)
        np.random.seed(cfg.PROJ.SEED)

        print(f"{gct()} : model init")
        self.det = RFDetSO(
            cfg.TRAIN.score_com_strength,
            cfg.TRAIN.scale_com_strength,
            cfg.TRAIN.NMS_THRESH,
            cfg.TRAIN.NMS_KSIZE,
            cfg.TRAIN.TOPK,
            cfg.MODEL.GAUSSIAN_KSIZE,
            cfg.MODEL.GAUSSIAN_SIGMA,
            cfg.MODEL.KSIZE,
            cfg.MODEL.padding,
            cfg.MODEL.dilation,
            cfg.MODEL.scale_list,
        )
        self.des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
        self.model = RFNetSO(
            self.det, self.des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
        )

        print(f"{gct()} : to device")
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)
        # resume = args.resume
        # resume = None
        resume = 'runs/10_24_09_25/model/e121_NN_0.480_NNT_0.655_NNDR_0.813_MeanMS_0.649.pth.tar'
        print(f"{gct()} : in {resume}")
        checkpoint = torch.load(resume)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.ref_img = None

    def track(self, a, b, c, f):
        print('\n\nWHAT THE ACTUAL FUCKSDKJSDLJASDLKJASD')
        raise Exception
    ###############################################################################
    # detect and compute
    ###############################################################################
    # img1_path, img2_path = args.imgpath.split("@")
    def detectAndCompute(self, curr_img, mask):
        def to_cv2_kp(kp):
            # kp is like [batch_idx, y, x, channel]
            # print(kp[2])
            # print(type(kp[2]))
            # print(kp[2].shape)
            return cv2.KeyPoint(float(kp[2]), float(kp[1]), 0)

        def to_cv2_dmatch(m):
            return cv2.DMatch(m, m, m, float(m))

        def reverse_img(img):
            """
            reverse image from tensor to cv2 format
            :param img: tensor
            :return: RBG image
            """
            img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
            img = (img * 255).astype(np.uint8)  # change to opencv format
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # gray to rgb
            return img


        if self.ref_img is None:
            self.ref_img = curr_img
            #TODO what size should I be passing in here?
            self.kp_ref, self.des_ref, self.ref_img = self.model.detectAndCompute(
                    curr_img, self.device, (240, 320))
            # convert to np.array
            kp_array = self.kp_ref.cpu().numpy()#[:, 1:3]

            keypoints1 = list(map(to_cv2_kp, kp_array))

            returned_des =  self.des_ref.cpu().detach().numpy()
            return keypoints1, returned_des
        else:
            print('change output size to match data')
            self.kp_curr, self.des_curr, curr_img = self.model.detectAndCompute(curr_img, self.device, (240, 320))
            # kp1, des1, img1 = self.model.detectAndCompute(img1_path, self.device, (240, 320))
            # kp2, des2, img2 = model.detectAndCompute(img2_path, self.device, (240, 320))

            # predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(des1, des2, kp2, 0.7)
            predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(self.des_ref, self.des_curr, self.kp_curr, 0.7)
            idx = predict_label.nonzero().view(-1)
            # mkp1 = kp1.index_select(dim=0, index=idx.long())  # predict match keypoints in I1
            mkp1 = self.kp_ref.index_select(dim=0, index=idx.long())  # predict match keypoints in I1
            mkp2 = nn_kp2.index_select(dim=0, index=idx.long())  # predict match keypoints in I2
            # print('\n\nMY KP ARE TYPE: ', typ(mkp2))
            mkp1 = mkp1.cpu().numpy()
            mkp2 = mkp2.cpu().numpy()
            # print('MY converted KP ARE TYPE: ', typ(mkp2))

            # img1, img2 = reverse_img(img1), reverse_img(img2)
            keypoints1 = list(map(to_cv2_kp, mkp1))
            keypoints2 = list(map(to_cv2_kp, mkp2))
            self.kp_ref = self.kp_curr
            self.des_ref = self.des_curr

            # DMatch = list(map(to_cv2_dmatch, np.arange(0, len(keypoints1))))

            # matches1to2	Matches from the first image to the second one, which means that
            # keypoints1[i] has a corresponding point in keypoints2[matches[i]] .
            # outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, DMatch, None)
            # cv2.imwrite("outImg.png", outImg)

            # return self.kp_curr, self.des_cur
            returned_des =  self.des_curr.cpu().detach().numpy()
            print('returned_dtype: ', returned_des.dtype)
            print(type(returned_des))
            print('returned des: ', returned_des.shape)
            return keypoints2, returned_des
            # return keypoints2, DMatch

