import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from approach.configArt import shuffle_data
from augment_data.aug_mnist import augMnist

#将测试集按照标签进行分组
def selectSeedFirst(x_aug,y_aug):
    # 初始化一个空字典，用于存放分组后的数据
    grouped_data = {}

    # 遍历数据和标签，根据标签进行分组
    for x, y in zip(x_aug, y_aug):
        if y not in grouped_data:
            grouped_data[y] = []  # 如果标签对应的键不存在，创建一个空列表
        grouped_data[y].append(x)  # 将数据添加到对应标签的列表中

    # 现在 grouped_data 中的每个键对应一个类别，值是该类别对应的数据列表

    # 打印每个子类的样本数量
    for label, data_list in grouped_data.items():
        print(f'Label {label}: {len(data_list)} samples')


