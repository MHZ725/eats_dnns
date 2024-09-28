import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from augment_data.aug import Aug
from keras.datasets import mnist
class augMnist(Aug):
    def __init__(self):
        super().__init__("mnist")
        #对mnsit数据集进行设计
        self.name="mnist"
        self.train_size=60000
        self.test_size=10000
        self.class_num=10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        self.x_train=x_train
        self.y_train=y_train

    #加载数据
    def load_data(self, use_norm=False):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
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

    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dau = augMnist()
    dau.run("test")



