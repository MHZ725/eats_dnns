#计算距离

import numpy as np
from itertools import combinations
import time

#
def remove_add_data(m1, m2, n1, n2):

    # Select an index to remove
    index_to_remove = 0  # You can change this index as needed

    # Remove the element at the selected index from m1 and m2
    removed_element1 = m1[index_to_remove]
    removed_element2 = m2[index_to_remove]

    m1 = np.delete(m1, index_to_remove)
    m2 = np.delete(m2, index_to_remove,axis=0)

    n1 = np.append(n1,removed_element1)
    n2 = np.append(n2, [removed_element2], axis=0)

    return m1, m2, n1, n2

def calculate_dimension_differences(selected_test_prob, candidate_test_prob):

    n = 10
    k = 3  # 选择的维度数量

    # 生成所有可能的组合
    all_combinations = list(combinations(range(n), k))

    # 计算每个组合中各个维度的差值
    dimension_differences = []

    distance=0

    for combination in all_combinations:
        # 定义两个二维向量
        if(candidate_test_prob[combination[0]]<0.1 or candidate_test_prob[combination[1]]<0.1 or candidate_test_prob[combination[2]]<0.1):
            continue;
        m1 = np.array([selected_test_prob[combination[0]],selected_test_prob[combination[1]]])
        m2 = np.array([selected_test_prob[combination[0]],selected_test_prob[combination[2]]])
        m3 = np.array([selected_test_prob[combination[1]],selected_test_prob[combination[2]]])

        n1 = np.array([candidate_test_prob[combination[0]],candidate_test_prob[combination[1]]])
        n2 = np.array([candidate_test_prob[combination[0]],candidate_test_prob[combination[2]]])
        n3 = np.array([candidate_test_prob[combination[1]],candidate_test_prob[combination[2]]])
    
        # 计算夹角的sin值
        sin_theta_1 = np.abs(np.cross(m1,n1) / (np.linalg.norm(m1) * np.linalg.norm(n1))*(selected_test_prob[combination[2]]-candidate_test_prob[combination[2]]))
        sin_theta_2 = np.abs(np.cross(m2,n2) / (np.linalg.norm(m2) * np.linalg.norm(n2))*(selected_test_prob[combination[1]]-candidate_test_prob[combination[1]]))
        sin_theta_3 = np.abs(np.cross(m3,n3) / (np.linalg.norm(m3) * np.linalg.norm(n3))*(selected_test_prob[combination[0]]-candidate_test_prob[combination[0]]))

        differences=sin_theta_1+sin_theta_2+sin_theta_3
        distance=distance+differences

        dimension_differences.append(differences)

    return distance

def calculate_distance(selected_test_prob,candidate_test_prob,selected_class_samples,candidate_class_samples,
        selected_test_psedu,candidate_test_psedu,selected_real_label, candidate_real_label,select_set_size):

        #选择集
        selected_grouped_prob={}
        selected_grouped_label={}
        selected_grouped_samples={}
        for a,b,c,x in zip(selected_test_prob,selected_real_label,selected_class_samples,selected_test_psedu):
            if x not in selected_grouped_prob:
                selected_grouped_prob[x] = []
                selected_grouped_samples[x] = []
                selected_grouped_label[x] = []
            selected_grouped_prob[x].append(a)
            selected_grouped_samples[x].append(c)
            selected_grouped_label[x].append(b)
            
        #selected_grouped_prob=np.array(selected_grouped_prob)
        #selected_grouped_samples=np.array(selected_grouped_samples)
        #selected_grouped_label=np.array(selected_grouped_label)

        #候选集
        candidate_grouped_prob={}
        candidate_grouped_label={}
        candidate_grouped_samples={}
        for a,b,c,x in zip(candidate_test_prob,candidate_real_label,candidate_class_samples,candidate_test_psedu):
            if x not in candidate_grouped_prob:
                candidate_grouped_prob[x] = []
                candidate_grouped_samples[x] = []
                candidate_grouped_label[x] = []
            candidate_grouped_prob[x].append(a)
            candidate_grouped_samples[x].append(c)
            candidate_grouped_label[x].append(b)
        #统计数量
        for label, data_list in candidate_grouped_prob.items():
            print(f'Label {label}: {len(data_list)} samples')

        #candidate_grouped_prob_1=np.array(candidate_grouped_prob)
        #selected_grouped_samples=np.array(selected_grouped_samples)
        #candidate_grouped_label=np.array(candidate_grouped_label)
        #print(candidate_grouped_prob_1[0])

        # 最大化 最小距离
        
        for i in range(0,10):
            min_distances=[]
            for candidate_element in candidate_grouped_prob[i]:
                minDistance=100
                for selected_element in selected_grouped_prob[i]:
                    distance=calculate_dimension_differences(selected_element,candidate_element)
                    if(minDistance>distance): 
                        minDistance=distance
                min_distances.append(minDistance)

            #将选中最小距离中，最大的距离放入选择集中   按顺序排列
            sorted_indices = np.argsort(min_distances)[::-1]
            
            sorted_candidate_grouped_prob = [candidate_grouped_prob[i][j] for j in sorted_indices]
            sorted_candidate_grouped_label = [candidate_grouped_label[i][j] for j in sorted_indices]
            sorted_candidate_sample=[candidate_grouped_samples[i][j] for j in sorted_indices]
            """
            sorted_candidate_grouped_prob = np.array(candidate_grouped_prob)[sorted_indices]
            sorted_candidate_grouped_label =np.array(candidate_grouped_label)[sorted_indices]
            sorted_candidate_sample=np.array(candidate_grouped_samples)[sorted_indices]
            """
            candidate_grouped_prob[i]=sorted_candidate_grouped_prob      
            candidate_grouped_label[i]=sorted_candidate_grouped_label
            candidate_grouped_samples[i]=sorted_candidate_sample

            size=int(len(candidate_grouped_prob[i])/2+1)
            test_psedu= np.full(size, i)
            selected_test_prob=np.concatenate((selected_test_prob,np.array(sorted_candidate_grouped_prob)[:size]),axis=0)
            selected_real_label=np.concatenate((selected_real_label,np.array(sorted_candidate_grouped_label)[:size]))
            selected_test_psedu=np.concatenate((selected_test_psedu,test_psedu))
            selected_class_samples=np.concatenate((selected_class_samples,np.array(sorted_candidate_sample)[:size]),axis=0)
            

        return selected_test_prob,selected_real_label,selected_test_psedu,selected_class_samples

            



           

    #return 

    

def calculate_distance_art(selected_test_prob,candidate_test_prob,selected_class_samples,candidate_class_samples,
        selected_test_psedu,candidate_test_psedu,selected_real_label, candidate_real_label):
    select_set_size=selected_test_prob.size*2   #选择集总共图像
    #  selected_test_prob.shape         (1500,10)
    #  selected_class_samples.shape     (1500,32,32,3)
    #  selected_test_psedu.shape        (1500,1)
    #  selected_real_label.shape        (1500,1)


    selected_test_prob,selected_real_label,selected_test_psedu,selected_class_samples=calculate_distance(selected_test_prob,candidate_test_prob,selected_class_samples,candidate_class_samples,
        selected_test_psedu,candidate_test_psedu,selected_real_label, candidate_real_label,select_set_size)

    return selected_test_prob,selected_real_label,selected_test_psedu,selected_class_samples
