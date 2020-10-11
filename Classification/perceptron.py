from utils import *


class MyPainter(Painter):
    def img_collections(self, data, label, w, b):
        self.ax[0][0].set_ylim([-20, 10])
        x = np.array(list(range(4, 12)))

        y = -(w[0, 0] * x + b) / w[0, 1]
        line, = self.ax[0][0].plot(x, y, c='black', animated=True)
        super(MyPainter, self).img_collections(data, label, insert_img=[line])


class Perceptron:
    def __init__(self, n_features, show_img=False):
        self.w = np.zeros((1, n_features))
        self.b = .0
        self.show_img = show_img

        if self.show_img:
            self.painter = MyPainter(n_features)

    def fit(self, data, label, lr=1e-1, it=100, **kwargs):
        data = np.array(data, dtype=float)

        for _ in range(it):
            for i in range(data.shape[0]):
                xi, yi = data[i], label[i]
                if yi * (self.w @ xi + self.b) <= 0:
                    self.w += lr * yi * xi
                    self.b += lr * yi

                    if self.show_img:
                        self.painter.img_collections(data, label, self.w, self.b)

        if self.show_img:
            self.painter.show_ani(**kwargs)

    def predict(self, data):
        return np.int_(np.sign(self.w @ data.T + self.b).reshape(-1))


def simple_test():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    np.random.seed(6)

    x, y = datasets.make_blobs(centers=2, n_samples=200)
    y[y == 0] -= 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Perceptron(x.shape[1], show_img=True)
    model.fit(x_train, y_train,
              img_save_path='../img/perception.mp4',
              fps=2)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))


def iris_test():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    np.random.seed(6)

    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    y[y == 0] -= 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Perceptron(x.shape[1])
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))


if __name__ == '__main__':
    simple_test()
    # iris_test()
