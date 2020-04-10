# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 15:24
# @Author  : jamerri

import numpy as np
import random
import math

'''参数'''
step = 0.4
position_x0 = 0.2
position_y0 = 0.2
x_cell_size = 0.27
y_cell_size = 0.21
X_MAX = 5.5
X_MIN = 0
Y_MAX = 4.5
Y_MIN = 0
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
lists = raw_data.reshape((20, 20, 6))
# print(lists)

'''碰撞检测'''


def check_boundary_x(x):
    if x < X_MIN:
        x = X_MIN + abs(x - X_MIN)
        return x
    elif x > X_MAX:
        x = X_MAX - abs(x - X_MAX)
        return x
    return x

def check_boundary_y(y):
    if y < Y_MIN:
        y = Y_MIN + abs(y - Y_MIN)
        return y
    elif y > Y_MAX:
        y = Y_MAX - abs(y - Y_MAX)
        return y
    return y

'''随机轨迹'''
position_x.append(position_x0)
position_y.append(position_y0)
for i in range(143):
    yaw = random.uniform(-math.pi, math.pi)
    degree = math.degrees(yaw)
    add_x = math.cos(degree) * step
    add_y = math.sin(degree) * step
    X = add_x + position_x[m]
    Y = add_y + position_y[m]
    X_real = check_boundary_x(X)
    Y_real = check_boundary_y(Y)
    position_x.append(X_real)
    position_y.append(Y_real)
    m = m + 1
    # print(yaw, degree)
    # print(add_x, add_y)
    # print(X, Y)
    # print(X_real, Y_real)
    # print(position_x,position_y)
position_x = np.array(position_x)
position_y = np.array(position_y)
np.set_printoptions(suppress=True)

# print(position_x)
# print(position_y)

'''转化为网格'''
for i in range(143):
    row = position_x[i]/x_cell_size
    line = position_y[i]/y_cell_size
    if row or line == 20:
        row = row - 1
        line = line - 1
    if row or line == 21:
        row = row - 2
        line = line - 2
    sampling_data.append(lists[int(line)][int(row)])
sampling_data = np.array(sampling_data)
np.set_printoptions(suppress=True)
# print(sampling_data)

'''写入txt'''
over_data = np.array(sampling_data)
np.savetxt("C:/Users/Lenovo/Desktop/采样数据集/1h_random.txt", over_data,fmt='%5f',delimiter=',')
over_position_x = np.array(position_x)
over_position_y = np.array(position_y)
np.savetxt("C:/Users/Lenovo/Desktop/采样数据集/1h_random_position.txt",(over_position_x,over_position_y),fmt='%5f',delimiter=',')
# print(position_x)
# print(position_y)
# print(m)
# print(sampling_data)
