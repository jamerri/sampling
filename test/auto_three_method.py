# -*- coding: utf-8 -*-
#@Time    : 2020/3/29 23:03
#@Author  : jamerri

import numpy as np
from conf import conf
import function as fuc
import math
import time

start = time.time()

'定义为全局变量'
conf = conf.KernelDMVW()

map = []
D_total = []
NLPD1 = []
NLPD2 = []
NLPD3 = []

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


'''从txt读取9900个点'''
file_open_path = 'C:/Users/jamerri/Desktop/实验数据/1.5_9900_interpolation_data.txt'
'''处理原始数据'''
raw_data = np.loadtxt(file_open_path)
for i in range(9900):
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


for q in range(0, 5):
    D_train = []
    D_train = get_train_set(100, 9, 11, 20)
    print(len(D_train))
    '''设置测试集'''
    D_test = [i for i in D_total if i not in D_train]
    print(len(D_test))

    '''写D_train训练集到txt'''
    raw_D_train_c_filed = []
    for i in range(len(D_train)):
        raw_D_train_c_filed.append(
            [D_train[i].x, D_train[i].y, D_train[i].c, D_train[i].wind_x, D_train[i].wind_y, D_train[i].wind_z])

    fuc.write_D_train_data(raw_D_train_c_filed)
    '''写D_test训练集到txt'''
    raw_D_test_c_filed = []
    for i in range(len(D_test)):
        raw_D_test_c_filed.append(
            [D_test[i].x, D_test[i].y, D_test[i].c, D_test[i].wind_x, D_test[i].wind_y, D_test[i].wind_z])

    fuc.write_D_test_data(raw_D_test_c_filed)

    '''储存训练集数据到数组'''
    train_position_x = []
    train_position_y = []
    train_wind_x = []
    train_wind_y = []
    train_c = []

    '''训练kernelDMV方法'''
    kz1 = 0.4
    for i in range(len(D_train)):
        train_position_x.append(D_train[i].x)
        train_position_y.append(D_train[i].y)
        train_c.append(D_train[i].c)
    mean_value = fuc.mean_value_and_variance_value(train_position_x, train_position_y, train_c, kz1)[0]
    variance_value = fuc.mean_value_and_variance_value(train_position_x, train_position_y, train_c, kz1)[1]

    '''用测试集计算NLPD指标'''
    sub_n = []
    for j in range(len(D_test)):
        test_position_x = D_test[j].x
        test_position_y = D_test[j].y
        test_c = D_test[j].c
        number = fuc.find_number(test_position_x, test_position_y)
        sub_n.append(math.log(variance_value[number]) + pow((mean_value[number] - test_c), 2) / variance_value[
            number])
        # print(len(sub_n))
    NLPD1.append(np.array(sub_n).sum() / (2 * len(D_test)) + 0.5 * math.log(2 * math.pi))

    '''训练kernelDMVW方法'''
    kz2 = 0.4
    ws2 = 0.1
    for i in range(len(D_train)):
        train_position_x.append(D_train[i].x)
        train_position_y.append(D_train[i].y)
        train_c.append(D_train[i].c)
        train_wind_x.append(D_train[i].wind_x)
        train_wind_y.append(D_train[i].wind_y)
    calculate_speed = fuc.calculateSpeed_direction(train_wind_x, train_wind_y)[0]
    calculate_direction = fuc.calculateSpeed_direction(train_wind_x, train_wind_y)[1]
    mean_value = fuc.mean_value_and_variance_value_KernelDMVW(train_position_x, train_position_y, train_c, calculate_speed,
                                                 calculate_direction, kz2, ws2)[0]
    variance_value = fuc.mean_value_and_variance_value_KernelDMVW(train_position_x, train_position_y, train_c, calculate_speed,
                                                 calculate_direction, kz2, ws2)[1]

    '''用测试集计算NLPD指标'''
    sub_n = []
    for j in range(len(D_test)):
        test_position_x = D_test[j].x
        test_position_y = D_test[j].y
        test_c = D_test[j].c
        number = fuc.find_number(test_position_x, test_position_y)
        sub_n.append(math.log(variance_value[number]) + pow((mean_value[number] - test_c), 2) / variance_value[
            number])
        # print(len(sub_n))
    NLPD2.append(np.array(sub_n).sum() / (2 * len(D_test)) + 0.5 * math.log(2 * math.pi))

    '''训练kernelDMVW_pro方法'''
    kz3 = 0.4  # 最优带宽参数
    ws3 = 0.1  # 最优拉伸系数参数
    bt3 = 0.1  # 最优改进风速宽参数
    for i in range(len(D_train)):
        train_position_x.append(D_train[i].x)
        train_position_y.append(D_train[i].y)
        train_c.append(D_train[i].c)
        train_wind_x.append(D_train[i].wind_x)
        train_wind_y.append(D_train[i].wind_y)
    calculate_speed = fuc.calculateSpeed_direction(train_wind_x, train_wind_y)[0]
    calculate_direction = fuc.calculateSpeed_direction(train_wind_x, train_wind_y)[1]
    mean_value = fuc.mean_value_and_variance_value_KernelDMVW_pro(train_position_x, train_position_y, train_c, calculate_speed,
                                                     calculate_direction, kz3, ws3, bt3)[0]
    variance_value = fuc.mean_value_and_variance_value_KernelDMVW_pro(train_position_x, train_position_y, train_c, calculate_speed,
                                                     calculate_direction, kz3, ws3, bt3)[1]

    '''用测试集计算NLPD指标'''
    sub_n = []
    for j in range(len(D_test)):
        test_position_x = D_test[j].x
        test_position_y = D_test[j].y
        test_c = D_test[j].c
        number = fuc.find_number(test_position_x, test_position_y)
        sub_n.append(math.log(variance_value[number]) + pow((mean_value[number] - test_c), 2) / variance_value[
            number])
        # print(len(sub_n))
    NLPD3.append(np.array(sub_n).sum() / (2 * len(D_test)) + 0.5 * math.log(2 * math.pi))
print(NLPD1)
print(NLPD2)
print(NLPD3)

'''把NLPD1写入TXT'''
fuc.write_NLPD1(NLPD1)
fuc.write_NLPD2(NLPD2)
fuc.write_NLPD3(NLPD3)

end = time.time()
print('runtime:%s second' % (end - start))