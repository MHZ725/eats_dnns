import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from augment_data.aug import Aug
import numpy as np
from augment_data import SVNH_DatasetUtil


class augSvhn(Aug):
    def __init__(self):
        super().__init__("svhn")
        self.train_size = 73257
        self.test_size = 26032
        self.nb_classes = 10
        (x_train, y_train), (x_test, y_test) = SVNH_DatasetUtil.load_data()
        y_train = np.argmax(y_train, axis=1)
        self.x_train=x_train
        self.y_train=y_train

    def load_data(self, use_norm=False):
        
        (x_train, y_train), (x_test, y_test) = SVNH_DatasetUtil.load_data()
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        print(np.max(x_test), np.min(x_test), x_test.dtype)
        if use_norm:
            x_test = x_test.astype('float32')
            x_train = x_train.astype('float32')
            x_train /= 255
            x_test /= 255
        y_test = np.argmax(y_test, axis=1)
        y_train = np.argmax(y_train, axis=1)
        return (x_train, y_train), (x_test, y_test)

    def get_dau_params(self):
        params = {
            "SF": [(0, 0.15), (0, 0.15)],
            "RT": (5, 20),  # rotation
            "ZM": ((0.8, 1.5), (0.8, 1.5)),  # zoom
            "BR": 0.3,
            "SR": [10, 30],  # sheer
            "CT": [0.5, 1.5],
            "BL": None,  # blur
        }

        return params


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #dau = augSvhn()
    #dau.run("test")
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.datasets import fashion_mnist

    # 加载CIFAR-10测试集
    (_, _), (x_test, y_test) = SVNH_DatasetUtil.load_data()

    # 将图像数据归一化到 [0, 1] 范围
    x_test = x_test.astype('float32') / 255.0
    #x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    y_test= np.squeeze(y_test)
    y_test_new=np.argmax(y_test,axis=1)
    # 保存测试集的图像和标签为.npy文件
    #np.save('None_test_x.npy', x_test)
    np.save('None_test_y.npy', y_test_new)
    print(x_test.shape)
    print(y_test.shape)
