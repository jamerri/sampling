# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 16:04
# @Author  : jamerri

"""生成kernel类"""


class KernelDMVW(object):
    def __init__(self):
        # 初始化网格长度
        self.min_x = 0
        self.max_x = 5.5
        self.min_y = 0
        self.max_y = 4.5

        # 初始化算法参数
        self.cell_size_x = 0.185  # 0.185  # 0.05
        self.cell_size_y = 0.15  # 0.15  # 0.05
        self.kernel_size = None
        self.sigma_omega = None
        self.confidence_scale = 1
        self.wind_scale = None
        self.wind_speed_factor = None
        self.radius_multiple = 3

        # 创建网格
        self.cell_gride = None

        # 网格中心坐标
        self.center_points = None
        self.x_center = None
        self.y_center = None

        # 初始化测量向量（传感器信息）
        self.sensor_positions = None
        self.position_x = None
        self.position_y = None
        self.concentrations = None
        self.wind_directions = None
        self.wind_speeds = None

        # 初始化旋转椭圆的参数
        self.a = None
        self.b = None

        # 初始化地图
        # Omega
        self.weight_map = None
        # R（权重地图）
        self.concentration_weight_map = None
        # alpha（置信地图）
        self.confidence_map = None
        # r（平均地图）
        self.mean_map = None
        # v（方差地图）
        self.variance_map = None
