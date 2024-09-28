import os

#设置模型常用参数
label_class=10  #10分类任务

mnist="mnist"
cifar_10="cifar"
fashion="fashion"
svhn="svhn"

LeNet5 = "LeNet5"
LeNet1 = "LeNet1"
resNet20 = "resNet20"
vgg16 = "vgg16"
vgg19 = "vgg19"

dataset_name=[mnist,cifar_10,fashion,svhn]
model_data={
    mnist:[LeNet5,LeNet1],
    fashion:[LeNet5,resNet20],
    cifar_10:[vgg16,LeNet5],
    svhn:[LeNet5,vgg16]
}

pair_list = ["mnist_LeNet5", "mnist_LeNet1", "fashion_resNet20", "fashion_LeNet5", "svhn_LeNet5", "svhn_vgg16",
             "cifar_LeNet5", "cifar_vgg16"]

def get_model_path(datasets, model_name):
    dic = {"mnist_LeNet5": './model_dataset/model_mnist_LeNet5.hdf5',
           "mnist_LeNet1": "./model_dataset/model_mnist_LeNet1.hdf5", 
           "fashion_resNet20": "./model_dataset/model_fashion_resNet20.hdf5",
           "fashion_LeNet5": "./model_dataset/lenet5_fashion_mnist.h5",
           "cifar_vgg16": "./model_dataset/vgg16_cifar10.h5",
           "cifar_LeNet5": "./model_dataset/lenet5_cifar_1.h5",
           "svhn_vgg16": "./model_dataset/model_svhn_vgg16.hdf5",
           "svhn_LeNet5": "./model_dataset/model_svhn_LeNet5.hdf5",
           }
    return dic[datasets + "_" + model_name]