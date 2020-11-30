from MathMethods.Scaler import *
from MathMethods.Activation import *
from MathMethods.LossFunction import *
from MathMethods.Optimizer import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

sns.set_style('dark')

np.random.seed(6)


def show_scaler():
    def plot(d1, d2, title):
        fig, axs = plt.subplots(1, 3, squeeze=False)
        fig.set_size_inches(12, 4)
        i = 0

        axs[i][0].plot(np.arange(-10, 10, .1), d2[:, 0])
        axs[i][0].set_xlabel('x')
        axs[i][0].set_ylabel('y')
        axs[i][0].set_title('function diagram')

        axs[i][1].set_xlabel('x1')
        axs[i][1].set_ylabel('x2')
        axs[i][1].set_title('data distribution')
        axs[i][1].scatter(d1[:, 0], d1[:, 1])

        axs[i][2].set_xlabel('x')
        axs[i][2].set_ylabel('frequency')
        axs[i][2].set_title('feature frequency')
        sns.distplot(d1[:, 0], color='blue', ax=axs[i][2], label='x1')
        sns.distplot(d1[:, 1], color='green', ax=axs[i][2], label='x2')
        axs[i][2].legend()

        plt.savefig('../img/MathMethods/%s.png' % title)

    x1 = np.random.normal(loc=100, scale=5, size=1000).reshape(-1, 1)
    x2 = np.random.normal(loc=50, scale=10, size=1000).reshape(-1, 1)
    x3 = np.arange(-10, 10, .1).reshape(-1, 1)

    data = np.concatenate((x1, x2), axis=1)

    scaler = Scaler()
    plot(data, x3, 'Original')

    for name in dir(scaler):
        if name.startswith('__'):
            continue

        a = getattr(scaler, name)(data)
        if isinstance(a, np.ndarray):
            d = a
        else:
            d = a[0]

        a = getattr(scaler, name)(x3)
        if isinstance(a, np.ndarray):
            d2 = a
        else:
            d2 = a[0]

        plot(d, d2, name)


def show_activation():
    for name in dir(activation):
        if name.startswith('__'):
            continue

        if name.startswith('d_'):
            x = np.arange(-5, 5, .1)
            if 'sigmoid' in name:
                x = np.arange(-10, 10, .1)

            y = x
            if name[2:] == 'softmax':
                y = np.zeros_like(x)
                y[x < 0] = x[x < 0]
                y[x >= 0] = -x[x >= 0]

            title = name[2:]

            fig, ax = plt.subplots()
            ax.grid()

            a = getattr(activation, name[2:])(y)
            da = getattr(activation, name)(y)

            l1, = ax.plot(x, y, c='red', label='original')
            ax.tick_params(axis='y', labelcolor='r')

            ax1 = ax.twinx()
            l2, = ax1.plot(x, a, c='blue', label='activation')
            l3, = ax1.plot(x, da, c='green', label='gradient')

            lines = [l1, l2, l3]
            ax.legend(lines, [l.get_label() for l in lines])
            ax.set_title(title)

            plt.savefig('../img/MathMethods/%s.png' % title)
            # plt.show()


def show_loss():
    # ft, at = plt.subplots()

    for name in dir(loss):
        if name.startswith('__'):
            continue

        if name.startswith('d_'):
            if any([_ in name for _ in ['zero', 'cross', 'exp', 'per', 'Hinge', 'KL']]):
                p = np.linspace(-1, 3, 100)
                r = np.ones_like(p)

                x = np.linspace(-1, 3, 100)
                real, pred = np.meshgrid(x, x)
                continue
            else:
                p = np.linspace(-2, 2, 100)
                r = np.zeros_like(p)

                x = np.linspace(-2, 2, 100)
                real, pred = np.meshgrid(x, x)
                # continue

            title = name[2:]

            a = getattr(loss, name[2:])(r, p)
            da = getattr(loss, name)(r, p)

            # at.plot(p, a, label=title)
            # at.set_xlabel('f(x) where y=1')
            # at.set_ylabel('Loss')

            fig = plt.figure()
            fig.set_size_inches(12, 4)
            ax = fig.add_subplot(131)
            l1, = ax.plot(p, a, c='blue', label='loss')
            ax1 = ax.twinx()
            l2, = ax1.plot(p, da, c='green', label='gradient')
            lines = [l1, l2]
            ax.legend(lines, [l.get_label() for l in lines])
            ax.grid()
            ax.set_xlabel('f(x) where y=0')
            ax.set_ylabel('Loss')

            a = getattr(loss, name[2:])(real, pred)

            ax = fig.add_subplot(132)
            ax.pcolormesh(real, pred, a, cmap='Blues')
            cs = ax.contour(real, pred, a)
            ax.clabel(cs)
            ax.set_title(title)
            ax.set_xlabel('y')
            ax.set_ylabel('f(x)')

            ax = fig.add_subplot(133, projection='3d')
            ax.plot_surface(real, pred, a, linewidth=0, cmap='rainbow')
            ax.view_init(25, 45)

            ax.set_xlabel('y')
            ax.set_ylabel('f(x)')
            ax.set_zlabel('Loss')

            plt.subplots_adjust(left=0.05, right=0.95, wspace=0.4)

            fig.savefig('../img/MathMethods/%s.png' % title)
            # plt.show()
            # break

    # at.grid()
    # at.legend()
    # ft.savefig('../img/MathMethods/loss2.png')


def show_optimizer():
    def fx(x1, x2):
        # return 4 * x1 ** 2 - 2.1 * x1 ** 4 + 1 /3 * x1 ** 6 + x1 * x2 - 4 * x2 ** 2 + 4 * x2 ** 4
        return x1 ** 2 - x2 ** 2
        # return np.cos(x1) - np.sin(x2)

    def d_fx(x1, x2):
        return 2 * x1, -2 * x2
        # return -np.sin(x1), -np.cos(x2)

    # area = (-2, 8)
    area = (-4, 4)
    x, y = np.linspace(*area, 100), np.linspace(*area, 100)

    xx, yy = np.meshgrid(x, y)
    zz = fx(xx, yy)

    fig0 = plt.figure()
    fig0.set_size_inches(10, 5)
    ax0 = fig0.add_subplot(121)
    ax0.contourf(xx, yy, zz, cmap='Blues')

    ax0.set_ylim(area)
    ax0.set_xlim(area)
    ax0.set_xlabel('$\\theta_1$')
    ax0.set_ylabel('$\\theta_2$')

    ax1 = fig0.add_subplot(122, projection='3d')
    ax1.plot_surface(xx, yy, zz, linewidth=0, cmap='rainbow')

    ax1.set_ylim(area)
    ax1.set_xlim(area)
    ax1.set_zlim([-15, 15])
    ax1.set_xlabel('$\\theta_1$')
    ax1.set_ylabel('$\\theta_2$')
    ax1.set_zlabel('$J(\\theta)$')

    itera = 1000
    start = (-4, -.1)
    # start = (1, 4)

    for name in dir(optimizer):
        if name.startswith('__'):
            continue

        px = np.zeros(itera)
        py = np.zeros(itera)

        px[0], py[0] = start

        xarg, yarg = None, None
        for i in range(itera - 1):
            gx, gy = d_fx(px[i], py[i])
            if xarg:
                if isinstance(xarg, tuple):
                    _ = getattr(optimizer, name)(px[i], gx, *xarg)
                else:
                    _ = getattr(optimizer, name)(px[i], gx, xarg)
            else:
                _ = getattr(optimizer, name)(px[i], gx)

            if isinstance(_, np.float):
                px[i + 1] = _
            else:
                px[i + 1], xarg = _[0], _[1]

            if yarg:
                if isinstance(yarg, tuple):
                    _ = getattr(optimizer, name)(py[i], gy, *yarg)
                else:
                    _ = getattr(optimizer, name)(py[i], gy, yarg)
            else:
                _ = getattr(optimizer, name)(py[i], gy)

            if isinstance(_, np.float):
                py[i + 1] = _
            else:
                py[i + 1], yarg = _[0], _[1]

        fig = plt.figure()
        fig.set_size_inches(8, 4)
        title = name

        pz = fx(px, py)
        ax = fig.add_subplot(121)
        ax.contourf(xx, yy, zz, levels=10, cmap='Blues')
        ax.plot(px, py, color='r')
        ax.set_ylim(area)
        ax.set_xlim(area)
        ax.set_title(name)
        ax.set_xlabel('$\\theta_1$')
        ax.set_ylabel('$\\theta_2$')

        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(xx, yy, zz, cmap='rainbow', linewidth=0)
        ax.plot(px, py, pz, color='b')

        ax.set_ylim(area)
        ax.set_xlim(area)
        ax.set_zlim([-15, 15])
        ax.set_xlabel('$\\theta_1$')
        ax.set_ylabel('$\\theta_2$')
        ax.set_zlabel('$J(\\theta)$')

        ax0.plot(px, py, label=name)
        ax1.plot(px, py, pz, label=name)

        fig.savefig('../img/MathMethods/%s.png' % title)
        # plt.show()
        # break

    ax0.legend()
    ax1.legend()
    fig0.savefig('../img/MathMethods/optimizer.png')


def show_regularization():
    def lp(x1, x2, p):
        return (np.abs(x1) ** p + np.abs(x2) ** p) ** (1 / p)

    area = (-10, 10)
    x, y = np.linspace(*area, 100), np.linspace(*area, 100)

    xx, yy = np.meshgrid(x, y)

    fig = plt.figure()
    fig.set_size_inches(16, 8)

    for i in range(4):
        p = 0.5 * 2 ** i
        zz = lp(xx, yy, p)

        ax = fig.add_subplot(241 + i)
        ax.contour(xx, yy, zz, cmap='Blues')
        ax.set_axis_off()
        ax.set_title('p=%s' % p)

        ax = fig.add_subplot(245 + i, projection='3d')
        ax.plot_surface(xx, yy, zz, cmap='rainbow', linewidth=0)
        ax.set_axis_off()

    fig.savefig('../img/MathMethods/Regularization2.png')
    plt.show()


if __name__ == '__main__':
    # show_scaler()
    # show_activation()
    # show_loss()
    # show_optimizer()
    show_regularization()
