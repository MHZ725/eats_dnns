import matplotlib.pyplot as plt
import numpy as np

# 数据  svhn-vgg16
ours_data = np.array([(0, 0), (1, 431), (2.5, 438), (5, 441), (7.5, 447), (10, 447)])
rts_data = np.array([(0, 0), (1, 411), (2.5, 433), (5, 441), (7.5, 445), (10, 443)])
ats_data = np.array([(0, 0), (1, 424), (2.5, 440), (5, 445), (7.5, 447), (10, 446)])
maxp_data = np.array([(0, 0), (1, 345), (2.5, 406), (5, 431), (7.5, 443), (10, 439)])
gini_data = np.array([(0, 0), (1, 326), (2.5, 382), (5, 429), (7.5, 439), (10, 439)])
nbc_data = np.array([(0, 0), (1, 0), (2.5, 0), (5, 0), (7.5, 0), (10, 0)])
snac_data = np.array([(0, 0), (1, 0), (2.5, 0), (5, 0), (7.5, 0), (10, 0)])
ces_data = np.array([(0, 0), (1, 262), (2.5, 344), (5, 410), (7.5, 426), (10, 439)])
rs_data = np.array([(0, 0), (1, 307), (2.5, 379), (5, 429), (7.5, 442), (10, 443)])

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
plt.title('svhn-vgg16')

# 显示图例
plt.legend(ncol=2)

# 调整刻度大小
plt.tick_params(axis='both', which='major', labelsize=12)

# 显示图表
plt.show()
