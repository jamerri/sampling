from scipy.interpolate import griddata, LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
from matplotlib import axis
import matplotlib.ticker as ticker
from pyheatmap.heatmap import HeatMap
from PIL import Image

__author__ = 'Li Dechang'
__data__ = '2020/3/19'

'''参数'''
x_min = 0
x_max = 5.5   # h 5.5
x_cell_size = 0.185  # h 0.28  0.056
y_min = 0
y_max = 4.5  # h 4.5
y_cell_size = 0.15  # h 0.23  0.046
start_line = 1
x_column = 0
# z_column = 1
y_column = 1
c_column = 2
vx_column = 3
vy_column = 4
vz_column = 5

file_path = 'F:\\pyCharm code\\row_1.5.txt'  # 没有变动的原始文件

'''处理原始数据'''
with open(file_path, 'r', encoding='utf-8') as file_object:
    raw_data = file_object.readlines()[start_line:]  # 从有数据的那一行开始
for i in range(len(raw_data)):
    data = raw_data[i].split(',')  # 分隔的符号根据原始文件形式而定
    data[-1] = data[-1].rstrip()
    for j in range(len(data)):  # 把列表元素从字符串转换成浮点数
        data[j] = float(data[j])
    raw_data[i] = data
np.set_printoptions(suppress=True)
raw_data = np.array(raw_data)
# print(raw_data)

'''坐标平移'''
raw_y = raw_data[:, 1]  # x坐标
raw_x = raw_data[:, 2]    # y坐标
# raw_z = raw_data[:, 3]  # z坐标
raw_d = raw_data[:, 4:8]  # 浓度及以后的数据
# m_z = min(raw_z)
m_y = min(raw_y)
m_x = min(raw_x)
raw_y = raw_data[:, 1] + (-m_y)
raw_x = raw_data[:, 2] + (-m_x)  # 偏移后的x坐标
# raw_z = raw_data[:, 3] + (-m_z)

'''处理后的数据，这样处理原始文件不变'''
raw_data = np.column_stack((raw_x, raw_y, raw_d))  # 平移后的原始数据，但是之前读进程序的原始数据不变
# raw_data = np.column_stack((raw_x, raw_z, raw_d))  # 合并


'''生成空网格'''
x = np.arange(x_min + 0.5 * x_cell_size, x_max + 0.5 * x_cell_size, x_cell_size)
print(len(x))
# print(len(x))
# x_r = x[::-1]  # 倒序
y = np.arange(y_min + y_cell_size * 0.5, y_max + y_cell_size * 0.5, y_cell_size)
print(len(y))
# print(len(y))
# y_r = y[::-1]
empty_grid = []
for i in y:  # y_r
    for j in x:  # x_r
        empty_grid.append([round(j, 5), round(i, 5)])
empty_grid = np.array(empty_grid)
# print(empty_grid)

'''进行插值，把原始数据集插值到空网格坐标节点上'''
grid_c = griddata(raw_data[:, :2], raw_data[:, c_column], empty_grid, method='linear')  # 插值
grid_v_x = griddata(raw_data[:, :2], raw_data[:, vx_column], empty_grid, method='linear')
grid_v_y = griddata(raw_data[:, :2], raw_data[:, vy_column], empty_grid, method='linear')
grid_v_z = griddata(raw_data[:, :2], raw_data[:, vz_column], empty_grid, method='linear')
interpolation_data = np.column_stack((empty_grid, grid_c, grid_v_x, grid_v_y, grid_v_z))  # 合并
np.set_printoptions(suppress=True)


'''找到nan，并处理nan'''
row_c = list(raw_data[:, 2])
min_c = min(row_c)
max_c = max(row_c)
cc = interpolation_data[:, 2]
xv = interpolation_data[:, 3]
yv = interpolation_data[:, 4]
zv = interpolation_data[:, 5]
cc[np.isnan(cc)] = min_c
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
zv[np.isnan(zv)] = 0
interpolation_c = interpolation_data[:, 2]  # 取出插值后的值
# interpolation_c = interpolation_c[::-1]
interpolation_c = np.array(interpolation_c).reshape(len(y), len(x))

'''写出插值后的数据'''
np.savetxt('1.5_900_interpolation_data.txt', interpolation_data, fmt="%.5f", delimiter=' ')
# np.savetxt('h_interpolation_c.txt', interpolation_c, fmt="%.5f", delimiter=' ')


'''绘制热力图'''
# x_grid, y_grid = np.meshgrid(x, y)
plt.figure()
'''坐标标签'''
plt.xlabel('X' + '/m')
plt.ylabel('Y/m')

# plt.xlim(0, 109)  # 截取显示范围
# plt.ylim(0, 90)

'''坐标映射，换一次模拟图改一次坐标映射'''
yLabel = ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5']

yLabel = np.array(yLabel)
xLable = ['0', '0.6', '1.2', '1.8', '2.4', '3.0', '3.6', '4.2', '4.8', '5.5']
xLable = np.array(xLable)
plt.yticks(np.arange(0, 95, 10), yLabel)
plt.xticks(np.arange(0, 110, 12.1), xLable)

'''颜色bar调整'''
color_1 = '#ff0000'  # 红色
color_2 = '#00ff00'  # 绿色
color_3 = '#0000CD'  # 蓝色
color_4 = '#ffff00'  # 黄色
color_5 = '#00bfff'  # 深天蓝
color_6 = '#4169E1'  # 宝蓝色
color_7 = '#1E90FF'  # 闪蓝色
color_8 = '#6495ED'  # 矢车菊色
color_9 = '#7fff00'  # 淡绿色
color_10 = '#FF9900'  # 橙色
color_11 = '#ff8C00'  # 橙红色
color_12 = '#87CEFA'  # 浅天蓝

cmap2 = col.LinearSegmentedColormap.from_list('own2', [color_3, color_8, color_12, color_2, color_4, color_10,color_11 ,color_1])
cm.register_cmap(cmap=cmap2)
cm.get_cmap('own2')


'''vmin和vmax的具体取值根据具体数值范围而定'''
# norm = matc.BoundaryNorm(np.linspace(0, 4.1, 20), 19)
# d = plt.imshow(interpolation_c, cmap=plt.get_cmap('jet', 19), vmin=0.6, vmax=4, origin='low')
plt.imshow(interpolation_c, cmap='own2',origin='low', vmin=0.6, vmax=4)
# plt.contourf(x_grid, y_grid, interpolation_c, cmap='jet')
plt.colorbar()
plt.show()



