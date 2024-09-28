import matplotlib.pyplot as plt
import numpy as np

# 数据  mnist-lenet5
ours_data = np.array([(0, 0), (1, 352), (2.5, 397), (5, 416), (7.5, 435), (10, 429)])
rts_data = np.array([(0, 0), (1, 291), (2.5, 348), (5, 374), (7.5, 391), (10, 385)])
ats_data = np.array([(0, 0), (1, 291), (2.5, 389), (5, 413), (7.5, 427), (10, 429)])
maxp_data = np.array([(0, 0), (1, 258), (2.5, 323), (5, 376), (7.5, 402), (10, 419)])
gini_data = np.array([(0, 0), (1, 253), (2.5, 312), (5, 362), (7.5, 396), (10, 415)])
nbc_data = np.array([(0, 0), (1, 99), (2.5, 162), (5, 240), (7.5, 268), (10, 295)])
snac_data = np.array([(0, 0), (1, 76), (2.5, 151), (5, 235), (7.5, 260), (10, 298)])
ces_data = np.array([(0, 0), (1, 79), (2.5, 174), (5, 247), (7.5, 298), (10, 310)])
rs_data = np.array([(0, 0), (1, 81), (2.5, 151), (5, 236), (7.5, 283), (10, 311)])

# 计算理论值的面积
theoretical_value = 450
theoretical_area = theoretical_value * (10 / 100)  # 假设理论值的宽度是10%

# 计算曲线下方的面积与理论值的面积的比例
def calculate_ratio(data):
    x = data[:, 0] / 100  # 将百分比转换为小数
    y = data[:, 1]
    area_value = np.trapz(y, x)
    ratio = area_value / theoretical_area
    return ratio

# 打印比例值
print(f'Area Ratio for OURS: {calculate_ratio(ours_data):.4f}')
print(f'Area Ratio for RTS: {calculate_ratio(rts_data):.4f}')
print(f'Area Ratio for ATS: {calculate_ratio(ats_data):.4f}')
print(f'Area Ratio for Max_p: {calculate_ratio(maxp_data):.4f}')
print(f'Area Ratio for DeepGini: {calculate_ratio(gini_data):.4f}')
print(f'Area Ratio for NBC: {calculate_ratio(nbc_data):.4f}')
print(f'Area Ratio for SNAC: {calculate_ratio(snac_data):.4f}')
print(f'Area Ratio for CES: {calculate_ratio(ces_data):.4f}')
print(f'Area Ratio for RS: {calculate_ratio(rs_data):.4f}')

# 画曲线
def plot_curve(data, label,color=None):
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, label=label, marker='o', linestyle='-',color=color)

# 画450的水平线，使用中等蓝色以提高可见度
plt.axhline(y=theoretical_value, color='mediumblue', linestyle='--', label='Theo')

# 画曲线，为每条曲线指定颜色
plot_curve(rts_data, 'RTS', 'seagreen')  
plot_curve(ats_data, 'ATS', 'darkorange')  
plot_curve(maxp_data, 'Max_p', 'saddlebrown') 
plot_curve(gini_data, 'Gini', 'slateblue')  
plot_curve(nbc_data, 'NBC', 'violet') 
plot_curve(snac_data, 'SNAC', 'gold') 
plot_curve(ces_data, 'CES', 'teal') 
plot_curve(rs_data, 'RS', 'black')  
plot_curve(ours_data, 'OURS', 'crimson') 
# 添加标签和标题
plt.xlabel('Percentage of Selected Tests(%)')
plt.ylabel('Diverise Errors')
plt.title('mnist-lenet5')

# 添加网格
#plt.grid(True, linestyle='--', alpha=0.7)

# 显示图例
plt.legend(ncol=2)

# 调整刻度大小
plt.tick_params(axis='both', which='major', labelsize=12)

# 显示图表
plt.show()
