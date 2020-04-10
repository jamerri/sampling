# -*- coding: utf-8 -*-
#@Time    : 2020/3/15 20:44
#@Author  : jamerri

import numpy as np

'''参数'''
step = 0.4
position_x0 = 0.2
position_y0 = 0.2
x_cell_size = 0.27
y_cell_size = 0.21
position_x = []
position_y = []
m = 0
sampling_data = []

file_open_path = 'C:/Users/Lenovo/Desktop/采样数据集/h_interpolation_data.txt'
'''处理原始数据'''
with open(file_open_path, 'r', encoding='utf-8') as file_object:
    raw_data = file_object.readlines()[1:]  # 从有数据的那一行开始
for i in range(len(raw_data)):
    data = raw_data[i].split(' ')
    while '' in data:
        data.remove('')
    for j in range(len(data)):  # 把列表元素从字符串转换成浮点数
        data[j] = float(data[j])
    raw_data[i] = data

raw_data = np.array(raw_data)
np.set_printoptions(suppress=True)
# print(raw_data)
lists = raw_data.reshape((20,20,6))
# print(lists)

'''螺旋轨迹'''
position_x.append(position_x0)
position_y.append(position_y0)

a = 0

for i in range(21):
    if i % 4 == 0:  # 向右移动
        for j in range(13 - 2 * a):
                position_x.append(position_x[m] + step)
                position_y.append(position_y[m])
                m = m + 1
    elif i % 4 == 1:  # 向上移动
        for j in range(10 - 2 * a):
                position_x.append(position_x[m])
                position_y.append(position_y[m] + step)
                m = m + 1
    elif i % 4 == 2:  # 向左移动
        for j in range(12 - 2 * a):
                position_x.append(position_x[m] - step)
                position_y.append(position_y[m])
                m = m + 1
    elif i % 4 == 3:  # 向下移动
        for j in range(9 - 2 * a):
                position_x.append(position_x[m])
                position_y.append(position_y[m] - step)
                m = m + 1
        a = a + 1

position_x = np.array(position_x)
position_y = np.array(position_y)
np.set_printoptions(suppress=True)

'''转化为网格'''
for i in range(143):
    row = position_x[i]/x_cell_size
    line = position_y[i]/y_cell_size
    if row or line == 20:
        row = row - 1
        line = line - 1
    sampling_data.append(lists[int(line)][int(row)])
sampling_data = np.array(sampling_data)
np.set_printoptions(suppress=True)
# print(sampling_data)

'''写入txt'''
over_data = np.array(sampling_data)
np.savetxt("C:/Users/Lenovo/Desktop/采样数据集/2h_spiral.txt", over_data,fmt='%5f',delimiter=',')
over_position_x = np.array(position_x)
over_position_y = np.array(position_y)
np.savetxt("C:/Users/Lenovo/Desktop/采样数据集/2h_spiral_position.txt",(over_position_x,over_position_y),fmt='%5f',delimiter=',')
# print(position_x)
# print(position_y)
# print(m)