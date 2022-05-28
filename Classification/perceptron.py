import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter


class Perceptron:
    def __init__(self, n_features=None, show_img=False, show_ani=False, painter=None):
        self.show_img = show_img
        self.show_ani = show_ani

        if self.show_img or self.show_ani:
            self.painter = painter or Painter(n_features)
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

            if not painter and self.show_ani:
                self.painter.init_ani()

    def fit(self, data, label, lr=1e-1, it=100, ani_save_path=None, img_save_path=None, **kwargs):
        data = np.array(data, dtype=float)
        n_features = data.shape[1]
        self.w = np.zeros((n_features,))
        self.b = .0
        for _ in tqdm(range(it)):
            for i in range(data.shape[0]):
                xi, yi = data[i], label[i]
                if yi * (self.w @ xi + self.b) <= 0:
                    self.w += lr * yi * xi
                    self.b += lr * yi

                    if self.show_ani:
                        self.painter.img_collections(data, label, self.w, self.b)

        if self.show_ani:
            self.painter.show_ani(ani_save_path)

        if self.show_img:
            self.painter.show_pic(data, label, self.predict, img_save_path)
            self.painter.show()

    def predict(self, x):
        return np.int_(np.sign(self.w @ x.T + self.b).reshape(-1))


def simple_test():
    x, y = datasets.make_blobs(centers=2, n_samples=200)
    y[y == 0] -= 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Perceptron(x.shape[1], show_img=True, show_ani=True)
    model.fit(x_train, y_train,
              # ani_save_path='../img/perception.mp4',
              # img_save_path='../img/perception.png',
              fps=2)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.975"""


def real_data_test():
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    y[y == 0] -= 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Perceptron()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.7280701754385965"""


if __name__ == '__main__':
    simple_test()
    # real_data_test()
