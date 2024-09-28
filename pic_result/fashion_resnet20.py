import matplotlib.pyplot as plt
import numpy as np

# 数据  fashion-resNet20
ours_data = np.array([(0, 0), (1, 314), (2.5, 349), (5, 361), (7.5, 361), (10, 362)])
rts_data = np.array([(0, 0), (1, 283), (2.5, 322), (5, 347), (7.5, 348), (10, 359)])
ats_data = np.array([(0, 0), (1, 277), (2.5, 345), (5, 355), (7.5, 356), (10, 360)])
maxp_data = np.array([(0, 0), (1, 245), (2.5, 304), (5, 341), (7.5, 348), (10, 355)])
gini_data = np.array([(0, 0), (1, 243), (2.5, 290), (5, 336), (7.5, 345), (10, 352)])
nbc_data = np.array([(0, 0), (1, 123), (2.5, 168), (5, 218), (7.5, 223), (10, 245)])
snac_data = np.array([(0, 0), (1, 126), (2.5, 170), (5, 220), (7.5, 229), (10, 251)])
ces_data = np.array([(0, 0), (1, 169), (2.5, 241), (5, 294), (7.5, 321), (10, 337)])
rs_data = np.array([(0, 0), (1, 167), (2.5, 237), (5, 290), (7.5, 310), (10, 335)])

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
plt.title('fashion-resNet20')

# 显示图例
plt.legend(ncol=2)

# 调整刻度大小
plt.tick_params(axis='both', which='major', labelsize=12)

# 显示图表
plt.show()
