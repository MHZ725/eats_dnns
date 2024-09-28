import numpy as np
# 混洗数据
def shuffle_data(X, Y, seed=None):
    if len(X) != len(Y):
        raise ValueError("size X not eq Y")
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X, Y = X[shuffle_indices], Y[shuffle_indices]
    return X, Y
# RP 随机挑选重复数据
def repeat_data(x_ori_select, y_ori_select, ratio=0.2, seed=0):
    dup_num = int(len(x_ori_select) * ratio)
    x_ori_select, y_ori_select = shuffle_data(x_ori_select, y_ori_select, seed)
    x_temp, y_temp = x_ori_select[:dup_num], y_ori_select[:dup_num]
    assert len(x_temp) == dup_num
    return x_temp, y_temp

# 不相关数据
def irrelevant_data(len_x_ori_select, x_extra, ratio=0.2, seed=0):
    num = int(ratio * len_x_ori_select)
    print("numb", num)
    np.random.seed(seed)
    x_extra = x_extra[np.random.permutation(len(x_extra))]
    x_temp = x_extra[:num]
    y_temp = np.array([-1] * len(x_temp))
    assert len(x_temp) == num
    return x_temp, y_temp

# 敌意样本 adv
def adv_data(len_x_ori_select, x_extra, y_extra, ratio=0.2, seed=0):
    num = int(ratio * len_x_ori_select)
    if num >= len(x_extra):
        raise ValueError(num, len(x_extra), "ADV 不够了")
    x_extra, y_extra = shuffle_data(x_extra, y_extra, seed=seed)
    x_temp = x_extra[:num]
    y_temp = y_extra[:num]
    assert len(x_temp) == num
    return x_temp, y_temp

# 噪声 NP
# ratio 噪声数据比例
def noise_data(x_ori_select, y_ori_select, ratio=0.2, seed=1):
    dup_num = int(len(x_ori_select) * ratio)
    x_ori_select, y_ori_select = shuffle_data(x_ori_select, y_ori_select, seed)
    x_temp, y_temp = x_ori_select[:dup_num], y_ori_select[:dup_num]
    x_dup = []
    np.random.seed(seed)
    seed_list = np.random.randint(0, 100000, len(x_temp))
    for x, seed in zip(x_temp, seed_list):
        x2 = add_noise(x, seed)
        x_dup.append(x2)
    assert len(x_dup) == dup_num
    return np.array(x_dup), y_temp

# 数据破损
def corruption_data(x_ori_select, y_ori_select, ratio=0.2, seed=1):
    dup_num = int(len(x_ori_select) * ratio)
    x_ori_select, y_ori_select = shuffle_data(x_ori_select, y_ori_select, seed)
    x_temp, y_temp = x_ori_select[:dup_num], y_ori_select[:dup_num]
    x_cor = []
    np.random.seed(seed)
    seed_list = np.random.randint(0, 100000, len(x_temp))
    for x, seed in zip(x_temp, seed_list):
        x2 = add_corrup(x, seed)
        x_cor.append(x2)
    y_cor = np.array([-1] * len(x_temp))
    assert len(x_cor) == dup_num
    return np.array(x_cor), y_cor

# 数据投毒
def poison_data(x_ori_select, y_ori_select, ratio=0.5, seed=1):
    dup_num = int(len(x_ori_select) * ratio)
    x_ori_select, y_ori_select = shuffle_data(x_ori_select, y_ori_select, seed)
    x_temp, y_temp = x_ori_select[:dup_num], y_ori_select[:dup_num]
    x_cor = []
    np.random.seed(seed)
    seed_list = np.random.randint(0, 100000, len(x_temp))
    for x, seed in zip(x_temp, seed_list):
        x2 =add_poison(x, seed)
        x_cor.append(x2)
    y_cor = np.array([-1] * len(x_temp))
    assert len(x_cor) == dup_num
    return np.array(x_cor), y_cor

# 投毒

def add_poison(img, seed):
    x = img.copy()
    img_shape = x.shape
    corp_size = img_shape[0] // 4
    np.random.seed(seed)
    corp_position = np.random.randint(0, img_shape[0] - corp_size, size=2)
    np.random.seed(seed)
    x_poison = np.random.rand(corp_size, corp_size, img_shape[2])
    x[corp_position[0]:corp_position[0] + corp_size,
    corp_position[1]:corp_position[1] + corp_size,
    :] = x_poison
    return x

# 破损

def add_corrup(img, seed):
    x = img.copy()
    img_shape = x.shape
    corp_size = int(img_shape[0] // 1.5)
    np.random.seed(seed)
    corp_position = np.random.randint(0, img_shape[0] - corp_size, size=2)
    x[
    corp_position[0]:corp_position[0] + corp_size,
    corp_position[1]:corp_position[1] + corp_size,
    :] = 0
    return x

# 噪声

def add_noise(img, seed):
    x = img.copy()
    row, col, ch = x.shape
    mean = 0
    var = 0.001
    sigma = var ** 0.5
    np.random.seed(seed)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = x + gauss
    noisy = np.clip(noisy, 0, 1.0)
    return noisy.astype('float32').reshape(x.shape)
