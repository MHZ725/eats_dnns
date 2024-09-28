import matplotlib.pyplot as plt
import numpy as np

# 数据  cifar10-lenet5
ours_data = np.array([(0, 0), (1, 396), (2.5, 439), (5, 444), (7.5, 446), (10, 448)])
rts_data = np.array([(0, 0), (1, 335), (2.5, 398), (5, 434), (7.5, 434), (10, 442)])
ats_data = np.array([(0, 0), (1, 378), (2.5, 440), (5, 445), (7.5, 447), (10, 447)])
maxp_data = np.array([(0, 0), (1, 359), (2.5, 422), (5, 442), (7.5, 444), (10, 448)])
gini_data = np.array([(0, 0), (1, 345), (2.5, 420), (5, 437), (7.5, 445), (10, 447)])
nbc_data = np.array([(0, 0), (1, 195), (2.5, 260), (5, 288), (7.5, 309), (10, 328)])
snac_data = np.array([(0, 0), (1, 276), (2.5, 363), (5, 404), (7.5, 415), (10, 428)])
ces_data = np.array([(0, 0), (1, 263), (2.5, 336), (5, 365), (7.5, 400), (10, 414)])
rs_data = np.array([(0, 0), (1, 270), (2.5, 359), (5, 401), (7.5, 415), (10, 429)])

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
plt.title('cifar10-lenet5')

# 显示图例
plt.legend(ncol=2)

# 调整刻度大小
plt.tick_params(axis='both', which='major', labelsize=12)

# 显示图表
plt.show()
