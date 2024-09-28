# 测试
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from approach.configArt import shuffle_data
from augment_data.aug_mnist import augMnist
from augment_data.aug_cifar10 import augCifar
from augment_data.aug_fashion import augFashion
from augment_data.aug_svhn import augSvhn
import numpy as np
from keras.models import load_model
from config.modelconfig import label_class,get_model_path,mnist,LeNet5,cifar_10,vgg16,fashion,resNet20,vgg19,svhn,LeNet1
from approach.configArt import printWithColor
from approach.selectSeed import selectSeed
from approach.calDistance import calculate_distance_art
from termcolor import colored

#from RTS.BestSolution import dac_rank

from ATS.ATS import ATS
import time
from selection_method.nc_cover import CovRank
from selection_method.nc_cover import get_cov_initer

from experment.invalidData import noise_data,corruption_data,poison_data,repeat_data
from selection_method.rank_method.CES.condition import CES_ranker
import tensorflow as tf

def fault_detection(y, y_psedu):
    fault_num = np.sum(y != y_psedu)
    #print("fault num : {}".format(fault_num))

    diverse_fault_num = diverse_errors_num(y, y_psedu)
    #print("diverse_fault_num  : {}/{}".format(diverse_fault_num, 90))
    return fault_num, diverse_fault_num

def diverse_errors_num(y_s, y_psedu):
    fault_pair_arr = []
    fault_idx_arr = []
    for ix, (y_s_temp, y_psedu_temp) in enumerate(zip(y_s, y_psedu)):
        if y_s_temp == -1:
            continue
        elif y_s_temp == y_psedu_temp:
            continue
        else:
            key = (y_s_temp, y_psedu_temp)
            if key not in fault_pair_arr:
                fault_pair_arr.append(key)
                fault_idx_arr.append(ix)
    return len(fault_idx_arr)

def get_psedu_label(m, x):
    pred_test_prob = m.predict(x)
    y_test_psedu = np.argmax(pred_test_prob, axis=1)
    return y_test_psedu

def get_tests(x_dau, y_dau):
    test_size=y_dau.size
    x_sel, y_sel = x_dau[:test_size // 2], y_dau[:test_size // 2]
    x_val, y_val = x_dau[test_size // 2:], y_dau[test_size // 2:]
    return x_sel, y_sel, x_val, y_val
    
def get_tests1(x_dau, y_dau):
    test_size=y_dau.size
    x_sel, y_sel = x_dau[:test_size // 4], y_dau[:test_size // 4]
    x_val, y_val = x_dau[test_size // 4:], y_dau[test_size // 4:]
    return x_sel, y_sel, x_val, y_val

def Class_balance(real_label):
    unique_labels, counts = np.unique(real_label, return_counts=True)
    total_samples = len(real_label)

    # 打印结果
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100
        print_color_file(f"Class {label}: Count = {count}, Percentage = {percentage:.2f}%","yellow")

def print_color_file(string,colors):
    print(colored(string,colors))
    with open(filename, 'a') as file:
        # 将输出内容写入文件
        file.write(string+'\n')
    
def shuffle_data3(X, Y, Z):
    if len(X) != len(Y):
        raise ValueError("size X not eq Y")
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X, Y, Z = X[shuffle_indices], Y[shuffle_indices], Z[shuffle_indices]
    return X, Y, Z


if __name__ == '__main__':
    ##获取数据

    
    for j in range(8):
        if j==0:
            continue
            ex_dataset=mnist
            ex_model=LeNet5
            aug=augMnist()
            data_num=40000
            x_train=aug.x_train
            y_train=aug.y_train
        elif j==1:
            continue
            ex_dataset=mnist
            ex_model=LeNet1
            aug=augMnist()
            data_num=40000
            x_train=aug.x_train
            y_train=aug.y_train
        elif j==2:
            continue
            ex_dataset=fashion
            ex_model=resNet20
            aug=augFashion()
            data_num=40000
            x_train=aug.x_train
            y_train=aug.y_train
        elif j==3:
            continue
            ex_dataset=fashion
            ex_model=LeNet5
            aug=augFashion()
            data_num=40000
            x_train=aug.x_train
            y_train=aug.y_train
        elif j==4:
            ##   内存分配bug
            ex_dataset=cifar_10
            ex_model=vgg16
            aug=augCifar()
            data_num=40000
            x_train=aug.x_train
            y_train=aug.y_train
            tf.keras.backend.clear_session()
        elif j==5:
            continue
            ##  内存分配bug
            ex_dataset=cifar_10
            ex_model=LeNet5
            aug=augCifar()
            data_num=40000
            x_train=aug.x_train
            y_train=aug.y_train
        elif j==6:
            
            ##  内存分配bug
            ex_dataset=svhn
            ex_model=vgg16
            aug=augSvhn()
            data_num=104128
            x_train=aug.x_train
            y_train=aug.y_train
        else:
            continue
            ex_dataset=svhn
            ex_model=LeNet5
            aug=augSvhn()
            data_num=104128
            x_train=aug.x_train
            y_train=aug.y_train
        select_list=[0.01,0.025,0.05,0.075,0.1]
        for select_rate in select_list:
            filename='resnew\output_'+ex_dataset+'_'+ex_model+'_'+str(select_rate)+'.txt'
            print_color_file("{},{},{}".format(ex_dataset,ex_model,select_rate),"blue")
            for i in range(5):  #每次实验执行5次
                x_aug,y_aug=aug.load_aug_data("All",use_cache=False)
                x_aug,y_aug=shuffle_data(x_aug,y_aug)
                if ex_dataset == svhn:
                    x_aug,y_aug,x_val,y_val=get_tests1(x_aug,y_aug)
                else:
                    x_aug,y_aug,x_val,y_val=get_tests(x_aug,y_aug)

                #mnist  80000    40000张图片 1%  2.5%  5%  7.5%  10%
                #svhn   208256   104128
                size=(int)(data_num*0.8)
                mopname="None"

                if i==1:
  
                    mopname="noise_data"
                    x_invalid,y_invalid=noise_data(x_aug,y_aug)
                    x_aug=x_aug[:size]
                    y_aug=y_aug[:size]
                    x_aug=np.concatenate((x_aug,x_invalid),axis=0)
                    y_aug=np.concatenate((y_aug,y_invalid),axis=0)
                elif i==2:
                    
                    mopname="corruption_data"
                    x_invalid,y_invalid=corruption_data(x_aug,y_aug)
                    x_aug=x_aug[:size]
                    y_aug=y_aug[:size]
                    x_aug=np.concatenate((x_aug,x_invalid),axis=0)
                    y_aug=np.concatenate((y_aug,y_invalid),axis=0)
                elif i==3:
                    
                    mopname="poison_data"
                    x_invalid,y_invalid=poison_data(x_aug,y_aug)
                    x_aug=x_aug[:size]
                    y_aug=y_aug[:size]   
                    x_aug=np.concatenate((x_aug,x_invalid),axis=0)
                    y_aug=np.concatenate((y_aug,y_invalid),axis=0)
                elif i==4:
                    mopname="repeat_data"
                    x_invalid,y_invalid=repeat_data(x_aug,y_aug)
                    x_aug=x_aug[:size]
                    y_aug=y_aug[:size]
                    x_aug=np.concatenate((x_aug,x_invalid),axis=0)
                    y_aug=np.concatenate((y_aug,y_invalid),axis=0)


                x_aug,y_aug=shuffle_data(x_aug,y_aug)
                best_seed_num=data_num*select_rate/10 #单个类别  总共挑选的10*best_seed_num
                total_select_set=4000  #10%
                select_set=best_seed_num/2
                candidate_set=select_set*2
                print_color_file("select rate:{},count:{},mop_name:{}".format(select_rate,i,mopname),"yellow")
            

                """
                ##ours
                time_ours1=time.time()
                print_color_file("ours approach","blue")
                selected_test_prob,candidate_test_prob,selected_class_samples,candidate_class_samples,\
                    selected_test_psedu,candidate_test_psedu,selected_real_label, candidate_real_label=selectSeed(x_aug,y_aug,int(select_set),int(candidate_set),ex_dataset,ex_model)  #设置 数据集和模型
                printWithColor("开始计算距离","blue")
                #清洗数据
                candidate_test_prob,candidate_real_label,candidate_test_psedu=shuffle_data3(candidate_test_prob,candidate_real_label,candidate_test_psedu)

                selected_test_prob_new,selected_real_label_new,selected_test_psedu_new=calculate_distance_art(selected_test_prob,candidate_test_prob,selected_class_samples,candidate_class_samples,
                    selected_test_psedu,candidate_test_psedu,selected_real_label, candidate_real_label)
                time_ours2=time.time()
                print_color_file("my approach result:","blue")
                fault_num, diverse_fault_num=fault_detection(selected_real_label_new,selected_test_psedu_new)
                print_color_file("ours class banlance:","red")
                Class_balance(selected_real_label_new)
                print_color_file("ours fault_num:{}".format(fault_num),"red")
                print_color_file("ours diverse_fault_num:{}/{}".format(diverse_fault_num,90),"red")
                print_color_file("ours time: {:.2f}".format(time_ours2-time_ours1),"red")
                
                # 其他方法的 Python 代码
                #RTS
                
                time_RTS1=time.time()
                model=get_model_path(ex_dataset,ex_model)
                ori_model=load_model(model)
                nb_classes=10
                print_color_file("RTS","blue")
                #select_pro = ori_model.predict(x_select)
                select_pro = ori_model.predict(x_aug)
                select_lable = np.argmax(select_pro, axis=1)  # 测试样本的预测标签

                train_pro = ori_model.predict(x_train)
                rank_list=dac_rank(x_aug,nb_classes, ex_dataset,select_pro, train_pro,x_train, y_train)
                xs, ys, ys_psedu = x_aug[rank_list], y_aug[rank_list], select_lable[rank_list]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_RTS2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("RTS result:","blue")
                print_color_file("RTS class banlance:","red")
                Class_balance(ys_num)
                print_color_file("RTS fault_num:{}".format(fault_num_ats),"red")
                print_color_file("RTS diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("RTS time: {:.2f}".format(time_RTS2-time_RTS1),"red")
                
            
                #ATS
                
                time_ATS1=time.time()
                model=get_model_path(ex_dataset,ex_model)
                ori_model=load_model(model)
                nb_classes=10
                ats = ATS()
                print_color_file("ATS","blue")
                y_sel_psedu = get_psedu_label(ori_model, x_aug)
                div_rank, _, _ = ats.get_priority_sequence(x_aug, y_sel_psedu, nb_classes, ori_model, th=0.001)
                xs, ys, ys_psedu = x_aug[div_rank], y_aug[div_rank], y_sel_psedu[div_rank]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_ATS2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("ATS result:","blue")
                print_color_file("ATS class banlance:","red")
                Class_balance(ys_num)
                print_color_file("ATS fault_num:{}".format(fault_num_ats),"red")
                print_color_file("ATS diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("ATS time: {:.2f}".format(time_ATS2-time_ATS1),"red")
                
                #map_p
                time_maxp1=time.time()
                model=get_model_path(ex_dataset,ex_model)
                ori_model=load_model(model)
                print_color_file("Max_p","blue")
                pred_test_prob = ori_model.predict(x_aug)
                y_sel_psedu = np.argmax(pred_test_prob, axis=1)
                metrics = np.max(pred_test_prob, axis=1)
                max_p_rank = np.argsort(metrics)
                xs, ys, ys_psedu = x_aug[max_p_rank], y_aug[max_p_rank], y_sel_psedu[max_p_rank]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_maxp2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("Max_p result:","blue")
                print_color_file("Max_p class banlance:","red")
                Class_balance(ys_num)
                print_color_file("Max_p fault_num:{}".format(fault_num_ats),"red")
                print_color_file("Max_p diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("Max_p time: {:.2f}".format(time_maxp2-time_maxp1),"red")

                #DeepGini
                time_DeepGini1=time.time()
                model=get_model_path(ex_dataset,ex_model)
                ori_model=load_model(model)
                print_color_file("DeepGini","blue")
                pred_test_prob = ori_model.predict(x_aug)
                y_sel_psedu = np.argmax(pred_test_prob, axis=1)
                metrics = np.sum(pred_test_prob ** 2, axis=1)
                rank_lst = np.argsort(metrics)
                xs, ys, ys_psedu = x_aug[rank_lst], y_aug[rank_lst], y_sel_psedu[rank_lst]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_DeepGini2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("DeepGini result:","blue")
                print_color_file("DeepGini class banlance:","red")
                Class_balance(ys_num)
                print_color_file("DeepGini fault_num:{}".format(fault_num_ats),"red")
                print_color_file("DeepGini diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("DeepGini time: {:.2f}".format(time_DeepGini2-time_DeepGini1),"red")
                
                
                
                #神经元覆盖方法
                #NAC NBC SNAC TKNC
                #NAC
                print_color_file("NAC","blue")
                time_nac1=time.time()
                model_path=get_model_path(ex_dataset,ex_model)
               
                ori_model=load_model(model_path)
                cov_initer = get_cov_initer(x_train,y_train,ex_dataset,ex_model)
                cov_ranker = CovRank(cov_initer, model_path, x_aug, y_aug)
                _,_,_,_,rank,_ =cov_ranker.cal_nac_cov(False)

                pred_test_prob = ori_model.predict(x_aug)
                y_sel_psedu = np.argmax(pred_test_prob, axis=1)
                xs, ys, ys_psedu = x_aug[rank], y_aug[rank], y_sel_psedu[rank]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_nac2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("NAC result:","blue")
                print_color_file("NAC class banlance:","red")
                Class_balance(ys_num)
                print_color_file("NAC fault_num:{}".format(fault_num_ats),"red")
                print_color_file("NAC diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("NAC time: {:.2f}".format(time_nac2-time_nac1),"red")

                #NBC
                print_color_file("NBC","blue")
                time_nbc1=time.time()
                model_path=get_model_path(ex_dataset,ex_model)
                cov_initer = get_cov_initer(x_train,y_train,ex_dataset,ex_model)
                cov_ranker = CovRank(cov_initer, model_path, x_aug, y_aug)
                _,_,_,_,rank,_ =cov_ranker.cal_nbc_cov(False)

                pred_test_prob = ori_model.predict(x_aug)
                y_sel_psedu = np.argmax(pred_test_prob, axis=1)
                xs, ys, ys_psedu = x_aug[rank], y_aug[rank], y_sel_psedu[rank]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_nbc2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("NBC result:","blue")
                print_color_file("NBC class banlance:","red")
                Class_balance(ys_num)
                print_color_file("NBC fault_num:{}".format(fault_num_ats),"red")
                print_color_file("NBC diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("NBC time: {:.2f}".format(time_nbc2-time_nbc1),"red")
                
                #SNAC
                print_color_file("SNAC","blue")
                time_snac1=time.time()
                model_path=get_model_path(ex_dataset,ex_model)
                cov_initer = get_cov_initer(x_train,y_train,ex_dataset,ex_model)
                cov_ranker = CovRank(cov_initer, model_path, x_aug, y_aug)
                _,_,_,_,rank,_ =cov_ranker.cal_snac_cov(False)
                
                ori_model=load_model(model_path)
                pred_test_prob = ori_model.predict(x_aug)
                y_sel_psedu = np.argmax(pred_test_prob, axis=1)
                xs, ys, ys_psedu = x_aug[rank], y_aug[rank], y_sel_psedu[rank]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_snac2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("SNAC result:","blue")
                print_color_file("SNAC class banlance:","red")
                Class_balance(ys_num)
                print_color_file("SNAC fault_num:{}".format(fault_num_ats),"red")
                print_color_file("SNAC diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("SNAC time: {:.2f}".format(time_snac2-time_snac1),"red")
                """
                
                ## CES
                print_color_file("CES","blue")
                time_ces1=time.time()
                model_path=get_model_path(ex_dataset,ex_model)
                ori_model=load_model(model_path)
                rank=CES_ranker().run(ori_model, x_aug, data_num*select_rate)
                print(rank)
                model_path=get_model_path(ex_dataset,ex_model)
                
                pred_test_prob = ori_model.predict(x_aug)
                y_sel_psedu = np.argmax(pred_test_prob, axis=1)
                xs, ys, ys_psedu = x_aug[rank], y_aug[rank], y_sel_psedu[rank]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_ces2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("CES result:","blue")
                print_color_file("CES class banlance:","red")
                Class_balance(ys_num)
                print_color_file("CES fault_num:{}".format(fault_num_ats),"red")
                print_color_file("CES diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("CES time: {:.2f}".format(time_ces2-time_ces1),"red")
                """
                #LSA
                print_color_file("LSA","blue")
                time_lsa1=time.time()
                model_path=get_model_path(ex_dataset,ex_model)
                cov_initer = get_cov_initer(x_train,y_train,ex_dataset,ex_model)
                cov_ranker = CovRank(cov_initer, model_path, x_aug, y_aug)
                model_path=get_model_path(ex_dataset,ex_model)
                ori_model=load_model(model_path)
                pred_test_prob = ori_model.predict(x_aug)
                y_sel_psedu = np.argmax(pred_test_prob, axis=1)
                
                lsc = cov_initer.get_lsc()
                rate = lsc.fit(x_aug, y_sel_psedu)
                rank_lst = lsc.rank_2()
                rank=np.array(rank_lst)
                xs, ys, ys_psedu = x_aug[rank], y_aug[rank], y_sel_psedu[rank]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_lsa2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("LSA result:","blue")
                print_color_file("LSA class banlance:","red")
                Class_balance(ys_num)
                print_color_file("LSA fault_num:{}".format(fault_num_ats),"red")
                print_color_file("LSA diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("LSA time: {:.2f}".format(time_lsa2-time_lsa1),"red")
                """
                
                
                #RS
                print_color_file("RS","blue")
                time_rs1=time.time()
                model_path=get_model_path(ex_dataset,ex_model)
                ori_model=load_model(model_path)
                pred_test_prob = ori_model.predict(x_aug)
                y_sel_psedu = np.argmax(pred_test_prob, axis=1)
                xs,ys,ys_psedu=shuffle_data3(x_aug,y_aug,y_sel_psedu)
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_rs2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("RS result:","blue")
                print_color_file("RS class banlance:","red")
                Class_balance(ys_num)
                print_color_file("RS fault_num:{}".format(fault_num_ats),"red")
                print_color_file("RS diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("RS time: {:.2f}".format(time_rs2-time_rs1),"red")
                

 
                """
                #TKNC
                print_color_file("TKNC","blue")
                time_tknc1=time.time()
                model_path=get_model_path(ex_dataset,ex_model)
                cov_initer = get_cov_initer(x_train,y_train,ex_dataset,ex_model)
                cov_ranker = CovRank(cov_initer, model_path, x_aug, y_aug)
                _,_,rank,_,_,_ =cov_ranker.cal_tknc_cov()

                print(rank)
                model_path=get_model_path(ex_dataset,ex_model)
                ori_model=load_model(model_path)
                pred_test_prob = ori_model.predict(x_aug)
                y_sel_psedu = np.argmax(pred_test_prob, axis=1)
                xs, ys, ys_psedu = x_aug[rank], y_aug[rank], y_sel_psedu[rank]
                xs_num, ys_num, ys_psedu_num = xs[:int(data_num*select_rate)], ys[:int(data_num*select_rate)], ys_psedu[:int(data_num*select_rate)]
                time_tknc2=time.time()
                fault_num_ats, diverse_fault_num_ats = fault_detection(ys_num, ys_psedu_num)
                print_color_file("TKNC result:","blue")
                print_color_file("TKNC class banlance:","red")
                Class_balance(ys_num)
                print_color_file("TKNC fault_num:{}".format(fault_num_ats),"red")
                print_color_file("TKNC diverse_fault_num:{}/{}".format(diverse_fault_num_ats,90),"red")
                print_color_file("TKNC time: {:.2f}".format(time_tknc2-time_tknc1),"red")
                """
        #--------------------------------------------------------------------------------------------------------------------------------------
            ## 总共bug
            #fault_num_ats, diverse_fault_num_ats = fault_detection(ys, ys_psedu)
            #printWithColor("fault_num:{}".format(fault_num_ats),"red")
            #printWithColor("diverse_fault_num:{}".format(diverse_fault_num_ats),"red")


            ##其他对比方法
