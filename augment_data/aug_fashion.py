import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from augment_data.aug import Aug
from keras.datasets import fashion_mnist



class augFashion(Aug):
    def __init__(self):
        super().__init__("fashion")
        self.train_size = 60000
        self.test_size = 10000
        self.nb_classes = 10
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        self.x_train=x_train
        self.y_train=y_train

    # 用于扩增图片
    # 加载原始数据
    def load_data(self, use_norm=False):  # use_norm=False 用于图片扩增  # use_norm=True 用于训练数据和实验
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        if use_norm:
            x_test = x_test.astype('float32')
            x_train = x_train.astype('float32')
            x_train /= 255
            x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    # 获得当前的所有操作和参数
    def get_dau_params(self):
        params = {
            "SF": [(0.01, 0.13), (0.01, 0.13)],
            "RT": (5, 20),  # rotation
            "ZM": ((0.7, 1.5), (0.7, 1.5)),  # zoom
            "BR": 0.5,
            "SR": [10, 30],  # sheer
            "BL": "hard",  # blur
            "CT": [0.5, 1.5],
        }
        return params


if __name__ == '__main__':
    #dau = FashionDau()
    #dau.run("test")
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.datasets import fashion_mnist

    # 加载CIFAR-10测试集
    (_, _), (x_test, y_test) = fashion_mnist.load_data()

    # 将图像数据归一化到 [0, 1] 范围
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    y_test= np.squeeze(y_test)
    # 保存测试集的图像和标签为.npy文件
    np.save('None_test_x.npy', x_test)
    np.save('None_test_y.npy', y_test)
    print(x_test.shape)
    print(y_test.shape)
