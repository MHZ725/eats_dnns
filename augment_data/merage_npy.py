import numpy as np

# 用于存储所有数据的列表
all_data_x = []
all_data_y = []

# 假设你有三个.npy文件: file1.npy, file2.npy, file3.npy
files_to_merge_x = [r"aug_new\svhn\BL_test_x.npy", 
                    r"aug_new\svhn\BR_test_x.npy", 
                    r"aug_new\svhn\CT_test_x.npy",
                    r"aug_new\svhn\RT_test_x.npy",
                    r"aug_new\svhn\SF_test_x.npy",
                    r"aug_new\svhn\SR_test_x.npy",
                    r"aug_new\svhn\ZM_test_x.npy",
                    r"aug_new\svhn\None_test_x.npy"]

files_to_merge_y = [r"aug_new\svhn\BL_test_y.npy", 
                    r"aug_new\svhn\BR_test_y.npy", 
                    r"aug_new\svhn\CT_test_y.npy",
                    r"aug_new\svhn\RT_test_y.npy",
                    r"aug_new\svhn\SF_test_y.npy",
                    r"aug_new\svhn\SR_test_y.npy",
                    r"aug_new\svhn\ZM_test_y.npy",
                    r"aug_new\svhn\None_test_y.npy"]

# 逐个加载.npy文件并将其添加到all_data列表中
for file_path in files_to_merge_x:
    data = np.load(file_path)
    all_data_x.append(data)

for file_path in files_to_merge_y:
    data = np.load(file_path)
    all_data_y.append(data)
#使用numpy.concatenate将所有数据合并成一个数组
merged_data_x = np.concatenate(all_data_x, axis=0)  # 指定轴为0表示按行合并
merged_data_y = np.concatenate(all_data_y, axis=0)  # 指定轴为0表示按行合并

# 保存合并后的数据为一个新的.npy文件
np.save("All_test_x.npy", merged_data_x)
np.save("All_test_y.npy", merged_data_y)
