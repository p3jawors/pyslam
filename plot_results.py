import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

show_traj = False
plt.figure(figsize=(14, 8))
avg_errs = []
for ii in range(0, 11):
    save_name = 'RFNET-KITTI-%02d.npz' % ii
    data = np.load(save_name)
    traj = np.squeeze(data['traj']).T
    traj_gt = np.squeeze(data['traj_gt']).T
    errors = np.squeeze(data['errors']).T
    avg_errs.append(np.mean(errors))
    # print(traj.shape)
    # print(traj_gt.shape)
    # print(errors.shape)

    if show_traj:
        fig2 = plt.figure()
        ax = plt.subplot(211, projection='3d')
        ax.set_title(save_name.split('.')[0])
        ax.plot(traj[0], traj[1], traj[2], label='RFNET trajectory')
        ax.plot(traj_gt[0], traj_gt[1], traj_gt[2], label='GT trajectory')
        plt.legend()
        plt.show()

    plt.subplot(3, 4, ii+1)
    plt.ylabel('Error')
    plt.title('Scene %02d' % ii)
    plt.plot(errors[0], label='X')
    plt.plot(errors[1], label='Y')
    plt.plot(errors[2], label='Z')
    plt.plot(np.linalg.norm((traj_gt-traj), axis=0), label='2norm')
    plt.legend()

ax12 = plt.subplot(3, 4, 12)
plt.bar(range(0,11), avg_errs)
plt.xlabel('KITTI Sequence')
plt.ylabel('Average Translation Error [m]')
ax12.set_ylim((0,np.max(avg_errs)+10))
for ii, err in enumerate(avg_errs):
    ax12.text(ii-0.5, err + 2, '%.2f' % err, color='blue')
plt.show()
