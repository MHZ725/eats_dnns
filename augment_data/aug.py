import os
import abc
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
from tqdm import tqdm

class Aug(object):
    def __init__(self, data_name, classes_num=10):
        self.base_dir = "aug_new"
        self.aug_dir = os.path.join(self.base_dir, data_name)
        self.class_num=classes_num
        self.parms=self.get_base_dau_params()
        self.cache_map = {}    #定义缓存空间
        self.dau_op = None  # "constant"
    
    def init_dir(self):
        if not os.path.exists(self.aug_dir):
            os.makedirs(self.aug_dir)
    
    def load_dop(self):
        import sys
        o_path = os.getcwd()
        sys.path.append(o_path)
        from data.augment import DauOperator
        
        self.dau_op = DauOperator()

    @staticmethod
    def get_base_dau_params():
        params = {
            "SF": [(0, 0.15), (0, 0.15)],        # 平移
            "RT": (0, 15),  # rotation           # 旋转
            "ZM": ((0.8, 1.2), (0.8, 1.2)),      # zoom 缩放
            "BR": 0.2,                           # 亮度
            "SR": [5, 20],                       # shear 剪切
            "BL": None,                          # blur模糊  
            "CT": [0.5, 1.5],                    # 对比度
            #"None":None
        }
        #print(params)   #输出几种数据增广参数
        return params

    #保存路径
    def get_data_save_path(self):
        return os.path.join(self.aug_dir,"{}_{}_{}.npy")   #  aug/mnist  SF_train_x.npy

    #加载扩增数据
    def load_aug_data(self,aug_op,prefix="test",use_norm=True,use_cache=True):
        #是否使用缓存
        if use_cache:
            #是
            key=aug_op+"_"+prefix+"_"+str(int(use_norm))
            if key not in self.cache_map.keys():
                x_path=self.get_data_save_path().format(aug_op,prefix,"x")
                y_path=self.get_data_save_path().format(aug_op,prefix,"y")
                x=np.load(x_path)
                y=np.load(y_path)
                if use_norm:
                    x=x.astype('float32')/255
                self.cache_map[key+"_x"]=x
                self.cache_map[key+"_y"]=y
            else:
                return self.cache_map[key+"_x"],self.cache_map[key+"_y"]
        else:
            #否
            x_path=self.get_data_save_path().format(aug_op,prefix,"x")
            y_path=self.get_data_save_path().format(aug_op,prefix,"y")
            x=np.load(x_path)
            y=np.load(y_path)
            if use_norm:
                x = x.astype('float32') / 255
        return x,y
    
    @abc.abstractmethod
    def load_data(self, use_norm=False):
        pass

    def run(self, prefix, num=1):
        self.load_dop()
        self.init_dir()
        (x_train, y_train), (x_test, y_test) = self.load_data(use_norm=False)
        params = self.get_base_dau_params()
        # for i in range(num):
        for k, v in params.items():
            img_list = []
            label_list = []
            dau_op_name = k
            print(k)
            x_path = self.get_data_save_path().format(dau_op_name, prefix, "x")
            y_path = self.get_data_save_path().format(dau_op_name, prefix, "y")
            if prefix == "train":
                data = zip(x_train, y_train)
            else:
                data = zip(x_test, y_test)
            for i, (x, y) in tqdm(enumerate(data)):
                dau_func, dau_v = self.dau_op.get_dau_func_and_values(k, v, seed=None)
                img = dau_func(x, dau_v, seed=None)
                img_list.append(img)
                label_list.append(y)
            xs = np.array(img_list)
            ys = np.array(label_list)
            np.save(x_path, xs)
            np.save(y_path, ys)
            print(np.max(xs), xs.dtype)
            
            





