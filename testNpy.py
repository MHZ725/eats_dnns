import numpy as np
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from augment_data.aug_mnist import augMnist
from keras.models import load_model
from config.modelconfig import label_class,get_model_path,mnist,LeNet5

test_x= np.load(r"D:\MHZ\南航资料\newArt\aug_new\svhn\All_test_y.npy")
test_y= np.load(r"D:\MHZ\南航资料\newArt\aug_new\svhn\All_test_y.npy")


#print(test_x.size)
#print(test_x[0].shape)
#print(test_x[0])
#print(test_x.shape)

print(test_x.shape)
print(test_y.shape)

print(test_x[:10])
print(test_y[:10])
"""

model=get_model_path(mnist,LeNet5)
ori_model=load_model(model)
pred_test_prob=ori_model.predict(test_x)
y_test_psedu = np.argmax(pred_test_prob, axis=1)

print(pred_test_prob[102])
print(y_test_psedu[102])
print(test_y[102])
print(pred_test_prob[111])
print(y_test_psedu[111])
print(test_y[111])
print(pred_test_prob[1111])
print(y_test_psedu[1111])
print(test_y[1111])
print(pred_test_prob[1123])
print(y_test_psedu[1123])
print(test_y[1123])
print(pred_test_prob[1026])
print(y_test_psedu[1026])
print(test_y[1026])
print(pred_test_prob[10200])
print(y_test_psedu[10200])
print(test_y[10200])
"""