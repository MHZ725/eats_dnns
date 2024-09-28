import matplotlib.pyplot as plt
import numpy as np

# 数据  svhn-lenet5
ours_data = np.array([(0, 0), (1, 443), (2.5, 450), (5, 450), (7.5, 450), (10, 450)])
rts_data = np.array([(0, 0), (1, 425), (2.5, 447), (5, 450), (7.5, 450), (10, 450)])
ats_data = np.array([(0, 0), (1, 442), (2.5, 450), (5, 450), (7.5, 450), (10, 450)])
maxp_data = np.array([(0, 0), (1, 435), (2.5, 450), (5, 450), (7.5, 450), (10, 450)])
gini_data = np.array([(0, 0), (1, 439), (2.5, 449), (5, 450), (7.5, 450), (10, 450)])
nbc_data = np.array([(0, 0), (1, 249), (2.5, 349), (5, 405), (7.5, 424), (10, 438)])
snac_data = np.array([(0, 0), (1, 298), (2.5, 391), (5, 429), (7.5, 435), (10, 446)])
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
plt.title('svhn-lenet5')

# 显示图例
plt.legend(ncol=2)

# 调整刻度大小
plt.tick_params(axis='both', which='major', labelsize=12)

# 显示图表
plt.show()
