from utils import *


class MyPainter(Painter):
    def show_pic(self, data, label, predict, img_save_path=None, *args, **kwargs):
        for a, b, i in self.draw_pic(data, label, predict, img_save_path, *args, **kwargs):
            pass


class PCA:
    def __init__(self, data, k=2, n_features=None, show_img=False):
        self.k = k
        self.data = np.array(data)  # data -> [150, 4]

        self.show_img = show_img

        if self.show_img:
            self.painter = Painter(n_features)
            self.painter.beautify()
            self.painter.init_pic()

    def fit_transform(self, data, label=None, img_save_path=None):
        cov = np.cov(self.data, rowvar=False)  # cov -> [4, 4]
        eigen_values, eigen_vectors = np.linalg.eig(cov)  # eigen_values -> [4, 1], eigen_vectors -> [4, 4]

        argmax = np.argsort(eigen_values)[::-1]

        # select the first k column from eigen_vectors eigenvectors -> [4, 2]
        self.eigenvectors = eigen_vectors[:, argmax[:self.k]]

        if self.show_img:
            self.painter.show_pic(data, label, self.transform, img_save_path)
            self.painter.show()

        return self.transform(data)

    def transform(self, x):
        return np.matmul(x, self.eigenvectors)


def sample_test():
    # x, y = datasets.make_blobs(centers=3, n_features=3, n_samples=500)
    x, y = datasets.make_s_curve(n_samples=1000)
    transx = PCA(x).transform()
    fig1, ax1 = plt.subplots()
    fig2, _ = plt.subplots()
    ax2 = Axes3D(fig2)
    ax1.scatter(transx[:, 0], transx[:, 1], c=y)
    ax2.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
    # fig1.savefig('../img/PCA_after.png')
    # fig2.savefig('../img/PCA_before.png')
    plt.show()


def sklearn_test():
    from sklearn.decomposition import PCA

    x, y = datasets.make_s_curve(n_samples=1000)

    model = PCA()

    trans = model.fit_transform(x)

    print(trans)


if __name__ == '__main__':
    sklearn_test()
