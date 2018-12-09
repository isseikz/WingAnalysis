"""3次元シミュレーションの可視化用."""
import numpy as np
import matplotlib.pyplot as plt


def load_x(path):
    """メインで作ったログを読み込み、各状態量のログにする."""
    x = np.load(path)
    t = x[0:-1, 0]
    q = x[0:-1, 1:5]
    pqr = x[0:-1, 5:8]
    xyz_i = x[0:-1, 8:11]
    uvw = x[0:-1, 11:14]

    return t, q, pqr, xyz_i, uvw


def load_FM(path):
    """メインで作ったログを読み込み、各状態量のログにする."""
    FM = np.load(path)
    t = FM[0:-1, 0]
    data = FM[0:-1, 1:4]
    return t, data


if __name__ == '__main__':
    t, q, pqr, xyz_i, uvw = load_x('x_1.npy')
    t, F = load_FM('F_1.npy')
    t, M = load_FM('M_1.npy')

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True)
    ax[0, 0].plot(t, q)
    ax[0, 0].legend(['q0', 'q1', 'q2', 'q3'])
    ax[0, 0].set_ylabel('quartanion')

    ax[1, 0].plot(t, pqr)
    ax[1, 0].legend(['p', 'q', 'r'])
    ax[1, 0].set_ylabel('angular velocity[rad/s]')

    ax[0, 1].plot(t, xyz_i)
    ax[0, 1].legend(['x', 'y', 'z'])
    ax[0, 1].set_ylabel('position, inertial frame [m]')

    ax[1, 1].plot(t, uvw)
    ax[1, 1].legend(['u', 'v', 'w'])
    ax[1, 1].set_ylabel('velocity, body frame [m/s]')

    ax[0, 2].plot(t, F)
    ax[0, 2].legend(['Fx', 'Fy', 'Fz'])
    ax[0, 2].set_ylabel('Force [N]')

    ax[1, 2].plot(t, M)
    ax[1, 2].legend(['Mx', 'My', 'Mz'])
    ax[1, 2].set_ylabel('Moment[Nm]')

    for col in range(ax.shape[1]):
        ax[1, col].set_xlabel('time[s]', fontname='serif')

    plt.tight_layout()
    plt.show()
