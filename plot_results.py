import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

for ii in range(0, 10):
    # ii = 0
    save_name = 'RFNET-KITTI-%02d.npz' % ii
    data = np.load(save_name)
    traj = np.squeeze(data['traj']).T
    traj_gt = np.squeeze(data['traj_gt']).T
    errors = np.squeeze(data['errors']).T
    print(traj.shape)
    print(traj_gt.shape)
    print(errors.shape)

    plt.figure()
    plt.subplot(211, projection='3d')
    plt.title(save_name.split('.')[0])
    plt.plot(traj[0], traj[1], traj[2], label='RFNET trajectory')
    plt.plot(traj_gt[0], traj_gt[1], traj_gt[2], label='GT trajectory')
    plt.legend()
    plt.subplot(212)
    plt.ylabel('Error')
    plt.plot(errors[0], label='X')
    plt.plot(errors[1], label='Y')
    plt.plot(errors[2], label='Z')
    plt.plot(np.linalg.norm((traj_gt-traj), axis=0), label='2norm')
    plt.legend()
    plt.show()
