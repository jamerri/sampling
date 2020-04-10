# -*- coding: utf-8 -*-
#@Time    : 2020/3/29 21:27
#@Author  : jamerri

# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 13:54
# @Author  : jamerri

import numpy as np
import function as fuc
import random
from sklearn.model_selection import KFold
from conf import conf
import math
import time
from datetime import datetime

start = time.time()

'定义为全局变量'
conf = conf.KernelDMVW()

map = []
D_total = []
D_train = []
K_NLPD_value = []
parameter_kernel_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
parameter_wind_scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
np.set_printoptions(suppress=True)

'''定义坐标点信息类'''


class Point:
    def __init__(self, x, y, c, wind_x, wind_y, wind_z):
        self.x = x
        self.y = y
        self.c = c
        self.wind_x = wind_x
        self.wind_y = wind_y
        self.wind_z = wind_z


'''定义大网格类'''


class Block:
    def __init__(self, sets):
        self.point_set = sets


'''从txt读取900个点'''
file_open_path = 'C:/Users/jamerri/Desktop/实验数据/1.5_900_interpolation_data.txt'
'''处理原始数据'''
raw_data = np.loadtxt(file_open_path)
for i in range(900):
    D_total.append(
        Point(raw_data[i][0], raw_data[i][1], raw_data[i][2], raw_data[i][3], raw_data[i][4], raw_data[i][5]))
# print(raw_data)

'''设置训练集'''


def get_train_set(blc, b_len, b_bre, c_num):
    """
    设置训练集
    blc：大网格的数量
    b_len：一个大网格的长度
    b_bre：一个大网格的宽度
    c_num：每个网格采样的数量
    """
    tmp = []
    index_dict = {-1, }
    b_total = len(D_total) / blc  # 每个网格点的数量
    blc = int(np.math.sqrt(blc))
    for i in range(blc):
        for j in range(blc):  # 第i行第j列的网格
            for n in range(c_num):
                index = -1
                while index in index_dict:
                    r = np.random.randint(0, b_total)
                    ln = r // b_bre
                    col = r - ln * b_bre
                    index = (i * b_len + ln) * (blc * b_bre) + (j * b_bre + col)  # 计算采样点的索引
                index_dict.add(index)
                tmp.append(D_total[index])

    return tmp


D_train = get_train_set(25, 6, 6, 24)

'''设置测试集'''
D_test = [i for i in D_total if i not in D_train]

print(len(D_train))
print(len(D_test))

# '''写D_train训练集到txt'''
# raw_D_train_c_filed = []
# for i in range(len(D_train)):
#     raw_D_train_c_filed.append(
#         [D_train[i].x, D_train[i].y, D_train[i].c, D_train[i].wind_x, D_train[i].wind_y, D_train[i].wind_z])
#
# fuc.write_D_train_data(raw_D_train_c_filed)
# '''写D_test训练集到txt'''
# raw_D_test_c_filed = []
# for i in range(len(D_test)):
#     raw_D_test_c_filed.append(
#         [D_test[i].x, D_test[i].y, D_test[i].c, D_test[i].wind_x, D_test[i].wind_y, D_test[i].wind_z])
#
# fuc.write_D_test_data(raw_D_test_c_filed)

'''10折交叉验证'''
D_train = random.sample(D_train, len(D_train))
if len(D_train) == 25:
    del D_train[4], D_train[8], D_train[12], D_train[16], D_train[20]
print(len(D_train))
# print(D_train)

for u in range(len(parameter_wind_scale)):
    for m in range(len(parameter_kernel_size)):
        ws = parameter_wind_scale[u]
        kz = parameter_kernel_size[m]
        kf = KFold(n_splits=10)
        NLPD_value = []
        for train, test in kf.split(D_train):
            train_position_x = []
            train_position_y = []
            train_wind_x = []
            train_wind_y = []
            train_c = []
            for i in range(len(train)):
                cnt_train = train[i]
                # print(cnt_train)
                train_position_x.append(D_train[cnt_train].x)
                train_position_y.append(D_train[cnt_train].y)
                train_c.append(D_train[cnt_train].c)
                train_wind_x.append(D_train[cnt_train].wind_x)
                train_wind_y.append(D_train[cnt_train].wind_y)
            calculate_speed = fuc.calculateSpeed_direction(train_wind_x, train_wind_y)[0]
            calculate_direction = fuc.calculateSpeed_direction(train_wind_x, train_wind_y)[1]
            mean_value = fuc.mean_value_and_variance_value_KernelDMVW(train_position_x, train_position_y, train_c, calculate_speed, calculate_direction, kz, ws)[0]
            variance_value = fuc.mean_value_and_variance_value_KernelDMVW(train_position_x, train_position_y, train_c, calculate_speed, calculate_direction, kz, ws)[1]
            # print(mean_value)
            # print(variance_value)
            sub_n = []
            for j in range(len(test)):
                cnt_test = test[j]
                # print(cnt_test)
                test_position_x = D_train[cnt_test].x
                test_position_y = D_train[cnt_test].y
                test_c = D_train[cnt_test].c
                number = fuc.find_number(test_position_x, test_position_y)
                sub_n.append(math.log(variance_value[number]) + pow((mean_value[number] - test_c), 2) / variance_value[
                            number])
            # print(len(sub_n))
            NLPD_value_total = np.array(sub_n).sum() / (2 * len(test)) + 0.5 * math.log(2 * math.pi)
            NLPD_value.append(NLPD_value_total)
        mean_NLPD_value = np.mean(np.array(NLPD_value))
        K_NLPD_value.append(mean_NLPD_value)
print(K_NLPD_value)
print(len(K_NLPD_value))
min_NLPD_value = np.min(np.array(K_NLPD_value))
Num = np.argmin(K_NLPD_value)
empty = []
for i in parameter_wind_scale:
    for j in parameter_kernel_size:
        empty.append([i, j])
print(empty[Num])
print('min_NLPD_value = ', min_NLPD_value)
#
# kernel_size_num = Num % len(parameter_kernel_size)
# wind_scale_num = int(Num / len(parameter_kernel_size))
# print('min_NLPD_value = ', min_NLPD_value)
# print('the best kernel size = ', parameter_kernel_size[kernel_size_num - 1])
# print('the best wind scale = ', parameter_wind_scale[wind_scale_num])

end = time.time()
print('runtime:%s second' % (end - start))
