# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 16:05
# @Author  : jamerri

import numpy as np
from conf import conf
from datetime import datetime
import math

'定义为全局变量'
conf = conf.KernelDMVW()

"""写D_train到TXT"""


def write_D_train_data(train_txt):
    train_data_w = np.array(train_txt)
    time1 = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    np.savetxt('C:/Users/jamerri/Desktop/实验数据/D_train-' + time1 + '.txt', train_data_w, fmt="%.5f", delimiter=' ')


"""写D_train到TXT"""


def write_D_test_data(test_txt):
    test_data_w = np.array(test_txt)
    time2 = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    np.savetxt('C:/Users/jamerri/Desktop/实验数据/D_test-' + time2 + '.txt', test_data_w, fmt="%.5f", delimiter=' ')


"""写NLPD到TXT"""


def write_NLPD(NLPD_txt):
    NLPD_w = np.array(NLPD_txt)
    time3 = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    np.savetxt('C:/Users/jamerri/Desktop/实验数据/D_NLPD-' + time3 + '.txt', NLPD_w, fmt="%.5f", delimiter=' ')


'''网格中心点函数'''


def center_grid():
    center_grid = []
    x_grid = np.arange(conf.min_x + conf.cell_size_x / 2, conf.max_x, conf.cell_size_x)
    y_grid = np.arange(conf.min_y + conf.cell_size_y / 2, conf.max_y, conf.cell_size_y)
    for i in y_grid:
        for j in x_grid:
            center_grid.append([round(i, 2), round(j, 2)])
    conf.center_points = np.array(center_grid)
    return conf.center_points


# print(center_grid())

'''生成kernel DM函数'''


def mean_value_and_variance_value(X, Y, C, KZ):
    # 网格中心坐标
    conf.x_center = center_grid()[:, 1]
    conf.y_center = center_grid()[:, 0]
    weight = np.zeros(conf.x_center.shape)  # 一个传感器所对应的所有网格的权重
    concentration_weight = np.zeros(conf.x_center.shape)  # 一个传感器所对应的所有网格的浓度权重
    conf.kernel_size = KZ
    # 传感器有关信息
    conf.position_x = np.array(X)
    conf.position_y = np.array(Y)
    conf.concentrations = np.array(C)
    s = (len(conf.position_x), len(conf.x_center))
    sum_weight = np.zeros(s)  # 所有传感器对应的所有网格的权重形成的一个数组
    sum_concentration_weight = np.zeros(s)  # 所有传感器对应的所有网格的浓度权重形成的一个数组

    for i in range(len(conf.position_x)):
        x = conf.position_x[i]
        y = conf.position_y[i]
        concentration = conf.concentrations[i]
        for j in range(len(conf.x_center)):
            center_x = conf.x_center[j]
            center_y = conf.y_center[j]
            # 判断距离
            rotation_distance = pow((x - center_x), 2) + pow((y - center_y), 2)
            if rotation_distance > pow((conf.radius_multiple * conf.kernel_size), 2):
                weight[j] = 0
                concentration_weight[j] = 0
            else:
                # 前面常数部分
                conf.norm_fact = 1 / (np.sqrt(2 * np.pi) * conf.kernel_size)
                # 后面指数部分
                conf.exp_fact = rotation_distance / (2 * pow(conf.kernel_size, 2))
                weight[j] = conf.norm_fact * np.exp(- conf.exp_fact)
                concentration_weight[j] = weight[j] * concentration
        sum_weight[i] = np.array(weight)
        sum_concentration_weight[i] = np.array(concentration_weight)
    weight_value = sum_weight.sum(axis=0)
    concentration_weight_value = sum_concentration_weight.sum(axis=0)
    confidence_value = np.zeros(weight_value.shape)
    mean_value = np.zeros(weight_value.shape)
    conf.sigma_omega = conf.confidence_scale * (1 / (np.sqrt(2 * np.pi) * conf.kernel_size))
    r_0 = sum(conf.concentrations) / len(conf.position_x)
    # r_0 = 0
    for k in range(len(weight_value)):
        confidence = 1 - np.exp(-weight_value[k] / (conf.sigma_omega * conf.sigma_omega))
        confidence_value[k] = np.array(confidence)
        if weight_value[k] == 0:
            mean_value[k] = r_0
        else:
            mean = (confidence * (concentration_weight_value[k] / weight_value[k])) + ((1 - confidence) * r_0)
            mean_value[k] = np.array(mean)
    variance_factor = np.zeros(conf.position_x.shape)
    # contributions = np.zeros(mean_value.shape)
    temporary = np.zeros(mean_value.shape)
    # sum_variance_factor = np.zeros(sum_weight.shape)
    sum_temporary = np.zeros(sum_weight.shape)
    for h in range(len(conf.position_x)):
        # 网格形状（x 和 y方向网格个数）
        number_cell_x = int(np.ceil((conf.max_x - conf.min_x) / conf.cell_size_x))
        number_cell_y = int(np.ceil((conf.max_y - conf.min_y) / conf.cell_size_y))
        # 进行向上取整求出离传感器最近的网格位置
        number_x = int(np.ceil((conf.position_x[h] - conf.min_x) / conf.cell_size_x))
        number_y = int(np.ceil((conf.position_y[h] - conf.min_y) / conf.cell_size_y))
        number = int((number_y - 1) * number_cell_x + number_x - 1)  # 算出位置后减1,因为检索从0开始
        variance_factor[h] = (concentration - mean_value[number]) ** 2
        for n in range(len(mean_value)):
            temporary[n] = sum_weight[h][n] * variance_factor[h]
        sum_temporary[h] = np.array(temporary)
    variance_contributions = variance_factor.sum(axis=0)
    temporary_value = sum_temporary.sum(axis=0)
    v_0 = variance_contributions / len(conf.position_x)
    # v_0 = 0
    variance_value = np.zeros(weight_value.shape)
    for l in range(len(weight_value)):
        if weight_value[l] == 0:
            variance_value[l] = v_0
        else:
            variance_value[l] = confidence_value[l] * temporary_value[l] / weight_value[l] + (
                    1 - confidence_value[l]) * v_0
    return mean_value, variance_value

def mean_value_and_variance_value_KernelDMVW(X, Y, C, W_V, W_D, KZ, WS):
    # 网格中心坐标
    conf.x_center = center_grid()[:, 1]
    conf.y_center = center_grid()[:, 0]
    weight = np.zeros(conf.x_center.shape)  # 一个传感器所对应的所有网格的权重
    concentration_weight = np.zeros(conf.x_center.shape)  # 一个传感器所对应的所有网格的浓度权重
    conf.kernel_size = KZ
    conf.wind_scale = WS
    # 传感器有关信息
    conf.position_x = np.array(X)
    conf.position_y = np.array(Y)
    conf.concentrations = np.array(C)
    conf.wind_speeds = np.array(W_V)
    conf.wind_directions = np.array(W_D)
    # conf.wind_directions = np.zeros(len(conf.position_x))  # 无风向生成对应的零矩阵
    s = (len(conf.position_x), len(conf.x_center))
    sum_weight = np.zeros(s)  # 所有传感器对应的所有网格的权重形成的一个数组
    sum_concentration_weight = np.zeros(s)  # 所有传感器对应的所有网格的浓度权重形成的一个数组
    for i in range(len(conf.position_x)):
        x = conf.position_x[i]
        y = conf.position_y[i]
        concentration = conf.concentrations[i]
        # 风的角度换算为弧度制
        wind_direction_rad = np.deg2rad(conf.wind_directions[i])
        # 旋转矩阵
        conf.rotation = np.array([(np.cos(wind_direction_rad), -np.sin(wind_direction_rad)),
                                  (np.sin(wind_direction_rad), np.cos(wind_direction_rad))])
        conf.a = conf.kernel_size + conf.wind_scale * conf.wind_speeds[i]
        conf.b = conf.kernel_size / (1 + (conf.wind_scale * conf.wind_speeds[i]) / conf.kernel_size)
        # 旋转坐标轴后的传感器坐标
        rotation_x = x * np.cos(wind_direction_rad) + y * np.sin(wind_direction_rad)
        rotation_y = y * np.cos(wind_direction_rad) - x * np.sin(wind_direction_rad)
        for j in range(len(conf.x_center)):
            center_x = conf.x_center[j]
            center_y = conf.y_center[j]
            # 旋转坐标轴后的网格中心坐标
            rotation_center_x = center_x * np.cos(wind_direction_rad) + center_y * np.sin(wind_direction_rad)
            rotation_center_y = center_y * np.cos(wind_direction_rad) - center_x * np.sin(wind_direction_rad)
            # 判断距离
            rotation_distance = pow((rotation_center_x - rotation_x), 2) / pow(conf.radius_multiple * conf.a, 2) + \
                                pow((rotation_center_y - rotation_y), 2) / pow(conf.radius_multiple * conf.b, 2)
            if rotation_distance > 1:
                weight[j] = 0
                concentration_weight[j] = 0
            else:
                # 基础的协方差矩阵
                conf.covariance_matrix = np.array([(conf.a * conf.a, 0), (0, conf.b * conf.b)])

                # 旋转后的协方差矩阵
                conf.rotation_covariance = np.dot(np.dot(conf.rotation, conf.covariance_matrix),
                                                  np.transpose(conf.rotation))

                # 旋转后的协方差矩阵内的元素
                sigma_x_sq = conf.rotation_covariance[0][0]
                sigma_x = np.sqrt(sigma_x_sq)
                sigma_y_sq = conf.rotation_covariance[1][1]
                sigma_y = np.sqrt(sigma_y_sq)
                sigma_xy = conf.rotation_covariance[0][1]

                rho = sigma_xy / (sigma_x * sigma_y)
                # 前面常数部分
                conf.norm_fact = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))
                # 后面指数部分
                conf.exp_fact = (((center_x - x) ** 2 / sigma_x_sq) + ((center_y - y) ** 2 / sigma_y_sq) -
                                 ((2 * rho * (center_x - x) * (center_y - y)) / (sigma_x * sigma_y)))
                weight[j] = conf.norm_fact * np.exp(-(0.5 / (1 - rho ** 2)) * conf.exp_fact)
                concentration_weight[j] = weight[j] * concentration
        sum_weight[i] = np.array(weight)
        sum_concentration_weight[i] = np.array(concentration_weight)
    weight_value = sum_weight.sum(axis=0)
    concentration_weight_value = sum_concentration_weight.sum(axis=0)
    confidence_value = np.zeros(weight_value.shape)
    mean_value = np.zeros(weight_value.shape)
    conf.sigma_omega = conf.confidence_scale * (1 / (np.sqrt(2 * np.pi) * conf.kernel_size))
    r_0 = sum(conf.concentrations) / len(conf.position_x)
    # r_0 = 0
    for i in range(len(weight_value)):
        confidence = 1 - np.exp(-weight_value[i] / (conf.sigma_omega * conf.sigma_omega))
        confidence_value[i] = np.array(confidence)
        if weight_value[i] == 0:
            mean_value[i] = r_0
        else:
            mean = (confidence * (concentration_weight_value[i] / weight_value[i])) + ((1 - confidence) * r_0)
            mean_value[i] = np.array(mean)
    variance_factor = np.zeros(conf.position_x.shape)
    temporary = np.zeros(mean_value.shape)
    sum_temporary = np.zeros(sum_weight.shape)
    for i in range(len(conf.position_x)):
        # 网格形状（x 和 y方向网格个数）
        number_cell_x = int(np.ceil((conf.max_x - conf.min_x) / conf.cell_size_x))
        number_cell_y = int(np.ceil((conf.max_y - conf.min_y) / conf.cell_size_y))
        # 进行向上取整求出离传感器最近的网格位置
        number_x = int(np.ceil((conf.position_x[i] - conf.min_x) / conf.cell_size_x))
        number_y = int(np.ceil((conf.position_y[i] - conf.min_y) / conf.cell_size_y))
        number = int((number_y - 1) * number_cell_x + number_x - 1)  # 算出位置后减1,因为检索从0开始
        variance_factor[i] = pow((conf.concentrations[i] - mean_value[number]), 2)
        for j in range(len(mean_value)):
            temporary[j] = sum_weight[i][j] * variance_factor[i]
        sum_temporary[i] = np.array(temporary)
    variance_contributions = variance_factor.sum(axis=0)
    temporary_value = sum_temporary.sum(axis=0)
    # np.set_printoptions(suppress=True)  # 设置成不以科学计数法输出
    v_0 = variance_contributions / len(conf.position_x)
    # v_0 = 0
    variance_value = np.zeros(weight_value.shape)
    for i in range(len(weight_value)):
        if weight_value[i] == 0:
            variance_value[i] = v_0
        else:
            variance_value[i] = confidence_value[i] * temporary_value[i] / weight_value[i] + (
                        1 - confidence_value[i]) * v_0
    return mean_value, variance_value

def mean_value_and_variance_value_KernelDMVW_pro(X, Y, C, W_V, W_D, KZ, WS, BT):
    # 网格中心坐标
    conf.x_center = center_grid()[:, 1]
    conf.y_center = center_grid()[:, 0]
    weight = np.zeros(conf.x_center.shape)  # 一个传感器所对应的所有网格的权重
    concentration_weight = np.zeros(conf.x_center.shape)  # 一个传感器所对应的所有网格的浓度权重
    conf.kernel_size = KZ
    conf.wind_scale = WS
    conf.wind_speed_factor = BT
    # 传感器有关信息
    conf.position_x = np.array(X)
    conf.position_y = np.array(Y)
    conf.concentrations = np.array(C)
    conf.wind_speeds = np.array(W_V)
    conf.wind_directions = np.array(W_D)
    # conf.wind_directions = np.zeros(len(conf.position_x))  # 无风向生成对应的零矩阵
    s = (len(conf.position_x), len(conf.x_center))
    sum_weight = np.zeros(s)  # 所有传感器对应的所有网格的权重形成的一个数组
    sum_concentration_weight = np.zeros(s)  # 所有传感器对应的所有网格的浓度权重形成的一个数组
    for i in range(len(conf.position_x)):
        x = conf.position_x[i]
        y = conf.position_y[i]
        concentration = conf.concentrations[i]
        # 风的角度换算为弧度制
        wind_direction_rad = np.deg2rad(conf.wind_directions[i])
        # 旋转矩阵
        conf.rotation = np.array([(np.cos(wind_direction_rad), -np.sin(wind_direction_rad)),
                                  (np.sin(wind_direction_rad), np.cos(wind_direction_rad))])
        conf.a = conf.kernel_size + conf.wind_scale * conf.wind_speeds[i]
        conf.b = conf.kernel_size / (1 + (conf.wind_scale * conf.wind_speeds[i]) / conf.kernel_size)
        # 旋转坐标轴后的传感器坐标
        rotation_x = x * np.cos(wind_direction_rad) + y * np.sin(wind_direction_rad)
        rotation_y = y * np.cos(wind_direction_rad) - x * np.sin(wind_direction_rad)
        for j in range(len(conf.x_center)):
            center_x = conf.x_center[j]
            center_y = conf.y_center[j]
            # 旋转坐标轴后的网格中心坐标
            rotation_center_x = center_x * np.cos(wind_direction_rad) + center_y * np.sin(wind_direction_rad)
            rotation_center_y = center_y * np.cos(wind_direction_rad) - center_x * np.sin(wind_direction_rad)
            # 判断距离
            rotation_distance = pow((rotation_center_x - rotation_x), 2) / pow(conf.radius_multiple * conf.a, 2) + \
                                pow((rotation_center_y - rotation_y), 2) / pow(conf.radius_multiple * conf.b, 2)
            if rotation_distance > 1:
                weight[j] = 0
                concentration_weight[j] = 0
            else:
                # 基础的协方差矩阵
                conf.covariance_matrix = np.array([(conf.a * conf.a, 0), (0, conf.b * conf.b)])

                # 旋转后的协方差矩阵
                conf.rotation_covariance = np.dot(np.dot(conf.rotation, conf.covariance_matrix),
                                                  np.transpose(conf.rotation))

                # 旋转后的协方差矩阵内的元素
                sigma_x_sq = conf.rotation_covariance[0][0]
                sigma_x = np.sqrt(sigma_x_sq)
                sigma_y_sq = conf.rotation_covariance[1][1]
                sigma_y = np.sqrt(sigma_y_sq)
                sigma_xy = conf.rotation_covariance[0][1]

                rho = sigma_xy / (sigma_x * sigma_y)
                # 前面常数部分
                conf.norm_fact = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))
                # 改进的风速修正方法(旋转坐标轴后风向与x轴正方向一致）
                distance = np.sqrt(
                    pow((rotation_center_x - rotation_x), 2) + pow((rotation_center_y - rotation_y), 2))  # 椭圆内点与传感器距离
                c_angle = (rotation_center_x - rotation_x) / distance  # 椭圆内点与中心点连线与风向的余弦值

                # 前面常数部分
                conf.norm_fact = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))
                # 后面指数部分
                conf.exp_fact = (((center_x - x) ** 2 / sigma_x_sq) + ((center_y - y) ** 2 / sigma_y_sq) -
                                 ((2 * rho * (center_x - x) * (center_y - y)) / (sigma_x * sigma_y)))
                w = (conf.norm_fact * np.exp(-(0.5 / (1 - rho ** 2)) * conf.exp_fact)) * (
                            1 + conf.wind_speed_factor * conf.wind_speeds * c_angle)
                if w < 0:
                    weight[j] = 0
                    concentration_weight[j] = 0
                else:
                    weight[j] = w
                    concentration_weight[j] = w * concentration
        sum_weight[i] = np.array(weight)
        sum_concentration_weight[i] = np.array(concentration_weight)
    weight_value = sum_weight.sum(axis=0)
    concentration_weight_value = sum_concentration_weight.sum(axis=0)
    confidence_value = np.zeros(weight_value.shape)
    mean_value = np.zeros(weight_value.shape)
    conf.sigma_omega = conf.confidence_scale * (1 / (np.sqrt(2 * np.pi) * conf.kernel_size))
    r_0 = sum(conf.concentrations) / len(conf.position_x)
    # r_0 = 0
    for i in range(len(weight_value)):
        confidence = 1 - np.exp(-weight_value[i] / (conf.sigma_omega * conf.sigma_omega))
        confidence_value[i] = np.array(confidence)
        if weight_value[i] == 0:
            mean_value[i] = r_0
        else:
            mean = (confidence * (concentration_weight_value[i] / weight_value[i])) + ((1 - confidence) * r_0)
            mean_value[i] = np.array(mean)
    variance_factor = np.zeros(conf.position_x.shape)
    temporary = np.zeros(mean_value.shape)
    sum_temporary = np.zeros(sum_weight.shape)
    for i in range(len(conf.position_x)):
        # 网格形状（x 和 y方向网格个数）
        number_cell_x = int(np.ceil((conf.max_x - conf.min_x) / conf.cell_size_x))
        number_cell_y = int(np.ceil((conf.max_y - conf.min_y) / conf.cell_size_y))
        # 进行向上取整求出离传感器最近的网格位置
        number_x = int(np.ceil((conf.position_x[i] - conf.min_x) / conf.cell_size_x))
        number_y = int(np.ceil((conf.position_y[i] - conf.min_y) / conf.cell_size_y))
        number = int((number_y - 1) * number_cell_x + number_x - 1)  # 算出位置后减1,因为检索从0开始
        variance_factor[i] = pow((conf.concentrations[i] - mean_value[number]), 2)
        for j in range(len(mean_value)):
            temporary[j] = sum_weight[i][j] * variance_factor[i]
        sum_temporary[i] = np.array(temporary)
    variance_contributions = variance_factor.sum(axis=0)
    temporary_value = sum_temporary.sum(axis=0)
    # np.set_printoptions(suppress=True)  # 设置成不以科学计数法输出
    v_0 = variance_contributions / len(conf.position_x)
    # v_0 = 0
    variance_value = np.zeros(weight_value.shape)
    for i in range(len(weight_value)):
        if weight_value[i] == 0:
            variance_value[i] = v_0
        else:
            variance_value[i] = confidence_value[i] * temporary_value[i] / weight_value[i] + (
                        1 - confidence_value[i]) * v_0
    return mean_value, variance_value

def find_number(X, Y):
    conf.position_x = X
    conf.position_y = Y
    # 网格形状（x 和 y方向网格个数）
    number_cell_x = int(np.ceil((conf.max_x - conf.min_x) / conf.cell_size_x))
    number_cell_y = int(np.ceil((conf.max_y - conf.min_y) / conf.cell_size_y))
    # 进行向上取整求出离传感器最近的网格位置
    number_x = int(np.ceil((conf.position_x - conf.min_x) / conf.cell_size_x))
    number_y = int(np.ceil((conf.position_y - conf.min_y) / conf.cell_size_y))
    number = int((number_y - 1) * number_cell_x + number_x - 1)  # 算出位置后减1,因为检索从0开始
    return number

def calculateSpeed_direction(V_X, V_Y):
    ux_speed = np.array(V_Y)  # y轴
    uy_speed = np.array(V_X)  # x轴
    # ux_speed = read_data_set()[:, 5]  # z轴
    calculate_speed = np.zeros(len(uy_speed))
    calculate_direction = np.zeros(len(uy_speed))
    for i in range(len(ux_speed)):
        speed = np.sqrt(ux_speed[i] ** 2 + uy_speed[i] ** 2)
        if ux_speed[i] > 0 and uy_speed[i] > 0:
            direction = math.atan2(ux_speed[i], uy_speed[i]) / np.pi * 180
        elif ux_speed[i] > 0 and uy_speed[i] < 0:
            direction = 180 - (math.atan2(ux_speed[i], abs(uy_speed[i])) / np.pi * 180)
        elif ux_speed[i] < 0 and uy_speed[i] > 0:
            direction = 360 - (math.atan2(abs(ux_speed[i]), abs(uy_speed[i])) / np.pi * 180)
        else:
            direction = 180 + (math.atan2(abs(ux_speed[i]), abs(uy_speed[i])) / np.pi * 180)
        calculate_speed[i] = speed
        calculate_direction[i] = direction
    return calculate_speed, calculate_direction