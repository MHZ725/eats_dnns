import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from approach.configArt import shuffle_data
from augment_data.aug_mnist import augMnist
import numpy as np
from keras.models import load_model
from config.modelconfig import label_class,get_model_path,mnist,LeNet5,cifar_10,vgg16
from approach.configArt import printWithColor
"""
   进行数据增广后，挑选优秀种子
"""



#将测试集按照标签进行分组
def selectSeedFirst(x_aug,y_aug):
    # 初始化一个空字典，用于存放分组后的数据
    grouped_data = {}

    # 遍历数据和标签，根据标签进行分组
    for x, y in zip(x_aug, y_aug):
        if y not in grouped_data:
            grouped_data[y] = []   # 如果标签对应的键不存在，创建一个空列表
        grouped_data[y].append(x)  # 将数据添加到对应标签的列表中

    # 现在 grouped_data 中的每个键对应一个类别，值是该类别对应的数据列表

    # 打印每个子类的样本数量
    for label, data_list in grouped_data.items():
        print(f'Label {label}: {len(data_list)} samples')
    # 打印每个子类的样本数量
    class_samples_count = [len(data_list) for label, data_list in grouped_data.items()]
    sorted_data = [len(grouped_data[label]) for label in range(10)]
    print("排序后")
    print(sorted_data)
    
    # 返回包含每个类别样本数量的数组
    return sorted_data,np.array(grouped_data[0]),np.array(grouped_data[1]),np.array(grouped_data[2]),np.array(grouped_data[3]),np.array(grouped_data[4]),\
           np.array(grouped_data[5]),np.array(grouped_data[6]),np.array(grouped_data[7]),np.array(grouped_data[8]),np.array(grouped_data[9]),
           
#分组后，从每组中选择最优数据
def selectSeedWithProb(class_samples,select_set,candidate_set,dataset,models):
    model=get_model_path(dataset,models)
    ori_model=load_model(model)
    pred_test_prob=ori_model.predict(class_samples)
    y_test_psedu = np.argmax(pred_test_prob, axis=1)
    #print(pred_test_prob[:10])
    # 计算每个数组的元素值最大的前三个元素
    max_values_top3 = np.partition(-pred_test_prob, 3, axis=1)[:, :3]

    # 计算每个数组的方差
    variances = np.var(max_values_top3, axis=1)

    # 获取按照方差从小到大排序的数组索引
    sorted_indices = np.argsort(variances)
    
    #根据方差对预测概率列表，数据集（图片像素值）列表，伪标签列表
    sorted_class_samples = class_samples[sorted_indices]
    sorted_test_psedu=y_test_psedu[sorted_indices]
    sorted_test_prob = pred_test_prob[sorted_indices]

    #print(sorted_test_prob[:10].shape)
    #print(sorted_class_samples[:10].shape)
    #返回挑选种子和未挑选种子的预测概率数组、图片像素值数组、伪标签数组
    return sorted_test_prob[:select_set],sorted_test_prob[select_set:candidate_set+select_set],sorted_class_samples[:select_set],\
            sorted_class_samples[select_set:candidate_set+select_set],sorted_test_psedu[:select_set],sorted_test_psedu[select_set:candidate_set+select_set]


#种子选择
def selectSeed(x_aug,y_aug,select_set,candidate_set,dataset,model):
    printWithColor("{},{}".format(dataset,model),"blue")
    #进行分类
    class_samples_count,class_0_samples,class_1_samples,class_2_samples,class_3_samples,class_4_samples,\
        class_5_samples,class_6_samples,class_7_samples,class_8_samples,class_9_samples, = selectSeedFirst(x_aug,y_aug)
    
    #定义一个字典，用于存放每个类别的样本集合
    class_samples_dict = {
        0: (class_0_samples,class_samples_count[0]),
        1: (class_1_samples,class_samples_count[1]),
        2: (class_2_samples,class_samples_count[2]),
        3: (class_3_samples,class_samples_count[3]),
        4: (class_4_samples,class_samples_count[4]),
        5: (class_5_samples,class_samples_count[5]),
        6: (class_6_samples,class_samples_count[6]),
        7: (class_7_samples,class_samples_count[7]),
        8: (class_8_samples,class_samples_count[8]),
        9: (class_9_samples,class_samples_count[9]),
    }

    selected_test_prob=[]           #10个类别选择种子的预测概率
    no_selected_candidate_prob=[]   #10个类别未选择种子的预测概率
    selected_class_samples=[]       #10个类别选择种子的像素值
    no_selected_class_samples=[]    #10个类别未选择种子的像素值
    selected_test_psedu=[]          #10个类别选择种子的伪标签
    no_selected_candidate_psedu=[]  #10个类别未选择种子的伪标签

    selected_real_label=[]          #10个类别选择种子的真实标签
    no_selected_real_label=[]       #10个类别未选择种子的真实标签


    printWithColor("seed select begin","red")
    #得到每个类别选择种子和未选择种子的预测概率、图片像素值、伪标签、
    for class_label, (class_samples,sample_num) in class_samples_dict.items():

        array1, array2,array3, array4,array5, array6 = selectSeedWithProb(class_samples,select_set,candidate_set,dataset,model)
        selected_test_prob.extend(array1)
        no_selected_candidate_prob.extend(array2)
        selected_class_samples.extend(array3)
        no_selected_class_samples.extend(array4)
        selected_test_psedu.extend(array5)
        no_selected_candidate_psedu.extend(array6)
        
        selected_real_label.extend([class_label]*select_set)
        no_selected_real_label.extend([class_label]*candidate_set)



    # 将列表转成numpy    得到8个numpy数组   返回这8个数组
    selected_test_prob=np.array(selected_test_prob)
    no_selected_candidate_prob=np.array(no_selected_candidate_prob)
    selected_class_samples=np.array(selected_class_samples)
    no_selected_class_samples=np.array(no_selected_class_samples)
    selected_test_psedu=np.array(selected_test_psedu)
    no_selected_candidate_psedu=np.array(no_selected_candidate_psedu)
    selected_real_label=np.array(selected_real_label)
    no_selected_real_label=np.array(no_selected_real_label)


    return selected_test_prob,no_selected_candidate_prob,selected_class_samples,no_selected_class_samples,\
        selected_test_psedu,no_selected_candidate_psedu,selected_real_label,no_selected_real_label


if __name__ == '__main__':
    aug=augMnist()
    x_aug,y_aug=aug.load_aug_data("None",use_cache=False)
    x_aug,y_aug=shuffle_data(x_aug,y_aug)
    dataset=mnist
    model=LeNet5
    selectSeed(x_aug,y_aug,100,100,dataset,LeNet5)


    
    
    