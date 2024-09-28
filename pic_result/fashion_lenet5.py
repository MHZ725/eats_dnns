import matplotlib.pyplot as plt
import numpy as np

# 数据  fashion-lenet5
ours_data = np.array([(0, 0), (1, 306), (2.5, 341), (5, 348), (7.5, 345), (10, 361)])
rts_data = np.array([(0, 0), (1, 281), (2.5, 332), (5, 352), (7.5, 354), (10, 371)])
ats_data = np.array([(0, 0), (1, 313), (2.5, 354), (5, 371), (7.5, 370), (10, 375)])
maxp_data = np.array([(0, 0), (1, 272), (2.5, 307), (5, 329), (7.5, 331), (10, 346)])
gini_data = np.array([(0, 0), (1, 264), (2.5, 307), (5, 297), (7.5, 297), (10, 301)])
nbc_data = np.array([(0, 0), (1, 126), (2.5, 173), (5, 216), (7.5, 245), (10, 259)])
snac_data = np.array([(0, 0), (1, 150), (2.5, 205), (5, 252), (7.5, 282), (10, 303)])
ces_data = np.array([(0, 0), (1, 133), (2.5, 196), (5, 228), (7.5, 257), (10, 294)])
rs_data = np.array([(0, 0), (1, 163), (2.5, 208), (5, 261), (7.5, 288), (10, 296)])

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
plt.title('fashion-lenet5')

# 显示图例
plt.legend(ncol=2)

# 调整刻度大小
plt.tick_params(axis='both', which='major', labelsize=12)

# 显示图表
plt.show()
