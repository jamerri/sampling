import numpy as np
import math


train = np.zeros(10,9)
def center_grid():
    pass

def calculate_mape():
    pass

def calculaite_NLPD():
    parameter = np.linspace(0.1, 2.0, 20)
    '''为后面找点做准备'''
    index_grid = center_grid().tolist()  # 把中心网格array数组转换成列表，便于找到数据点索引
    for i in range(len(list(index_grid))):
        index_grid[i][0] = round(index_grid[i][0], 5)
        index_grid[i][1] = round(index_grid[i][1], 5)
        '''容器数组'''
    para_npld = np.zeros(len(parameter))  # 每一行是一个参数对应的10个NLPD的均值
    ten_nlpd = np.zeros(10)   # 存放10折的NLPD

    for i in parameter:
        for j in range(len(train)):   # 控制折数
            for k in range(len(train[j])):  # 控制每折中的训练集
                # measurement_positions_x = train[j].x
                # measurement_positions_y = train[j].y
                # measurement_concentrations = train[j].c

                # 注意，要想把kernel 嵌入，我觉得还是要把数组做成切片的形式，train[j][k].x这种形式与kernel中循环不符
                v,r=calculate_mape(train[j][k].x, train[j][k].y, train[j][k].c) # 调用kernel, 计算返回 均值地图数组和 方差地图数组

            test = np.zeros(10, 9)  # 测试集实际并非如此，在这只是举个例子，测试集应该是一个三维数组
            nlpd_v = np.zeros(len(test[0][0]))
            nlpd_r = np.zeros(len(test[0][0]))
            for k in range(len(test[j])):  # 控制每折中的测试集
                '''找到每折测试集中的点在预测均值地图和方差地图中对应的r和v'''
                x_index = np.floor(test[j][k].x / para.cell_size) * para.cell_size + 0.5 * para.cell_size
                y_index = np.floor(test[j][k].y / para.cell_size) * para.cell_size + 0.5 * para.cell_size
                index_position = [round(x_index, 5), round(y_index, 5)]  # 每个数据点对应中心网格得坐标
                # print(index_position[1])
                index = index_grid.index(index_position)  # 数据点所在网格得索引
                nlpd_v[k] = v[index]
                npld_r[k] = r[index]  # 返回一折中所有测试点对应的 v 和 r
                empty_c = []
                empty_c.append(test[j][k].c)  # 把一折中所有的浓度 写入一个列表
            sub_nlpd = math.log(nlpd_v) + pow((np.array(empty_c) - npld_r), 2) / nlpd_v  # 一折测试集数据对应的nlpd中花括号部分
            sub_nlpd = np.array(sub_nlpd)
            sum_sub_nlpd = sub_nlpd.sum(axis=1)  # 花括号内求和
            each_nlpd = 1/(2*len(test[j])) * sum_sub_nlpd + 0.5*math.log(2*np.pi)  # 每一折的NLPD
            ten_nlpd[j] = each_nlpd  # 把计算出的每一折NLPD存放到容器中，一共会存入10个
        mean_ten_nlpd = sum(ten_nlpd)/10  # 一个参数对应的10折求和
        para_npld[i] = mean_ten_nlpd    # 存放每个参数对应的平均nlpd








                # print(index_grid[i][1])




calculaite_NLPD()