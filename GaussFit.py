# 用户名：lenovo
# 时间：2022/8/10  9:35
# 用途：
import numpy as np
import math


def gauss_A(loc):
    """
    计算高斯拟合中的系数矩阵
    :param loc: 某一类的全体坐标集合[x,y1,f]
    :return:gaussian拟合对应线性方程组的系数矩阵
    """
    A = np.array([
        [sum([(i[0] ** 2 + i[1] ** 2) ** 2 for i in loc]), sum([i[0] * (i[0] ** 2 + i[1] ** 2) for i in loc]),
         sum([i[1] * (i[0] ** 2 + i[1] ** 2) for i in loc]), sum([(i[0] ** 2 + i[1] ** 2) for i in loc])],
        [sum([(i[0] ** 2 + i[1] ** 2) * i[0] for i in loc]), sum([i[0] ** 2 for i in loc]),
         sum([i[0] * i[1] for i in loc]), sum([i[0] for i in loc])],
        [sum([(i[0] ** 2 + i[1] ** 2) * i[1] for i in loc]), sum([i[0] * i[1] for i in loc]),
         sum([i[1] ** 2 for i in loc]), sum([i[1] for i in loc])],
        [sum([i[0] ** 2 + i[1] ** 2 for i in loc]), sum([i[0] for i in loc]), sum([i[1] for i in loc]),
         sum([1 for i in loc])]
    ])
    return A


def gauss_b(info):
    """
    计算高斯拟合中的常数项矩阵
    :param info: 某一类的全体坐标以及像素点集合[x,y1,f]
    :return:gaussian拟合对应线性方程组的常数项
    """
    b = np.array([sum([(i[0] ** 2 + i[1] ** 2) * math.log(i[2]) for i in info]),
                  sum([i[0] * math.log(i[2]) for i in info]),
                  sum([i[1] * math.log(i[2]) for i in info]),
                  sum([math.log(i[2]) for i in info])]).reshape(-1, 1)
    # try:
    #     b = np.array([sum([(i[0] ** 2 + i[1] ** 2) * math.log(i[2]) for i in info]),
    #                   sum([i[0] * math.log(i[2]) for i in info]),
    #                   sum([i[1] * math.log(i[2]) for i in info]),
    #                   sum([math.log(i[2]) for i in info])]).reshape(-1, 1)
    # except ValueError:
    #     print(f'info={info}')
    return b


def compute_center_Gauss(picture:np.ndarray):
    """
    粒子圆心估计的高斯拟合方法(when b function is error, please check if the image get inverse)
    :param picture: 需要进行寻找中心的子图
    :return:参数构成的字典，参数key值分别为：sigma, x, y1, a
    """
    # 初始信息组合
    m, n = picture.shape
    loc = np.array([
        [i, j] for i in range(m) for j in range(n)
    ])
    info = np.zeros([m * n, 3])
    y = np.array([picture[i[0], i[1]] for i in loc])
    info[:, 0] = loc[:, 0]
    info[:, 1] = loc[:, 1]
    info[:, 2] = y
    # 坐标归一化
    x_max = info[:, 0].max()
    x_min = info[:, 0].min()
    y_max = info[:, 1].max()
    y_min = info[:, 1].min()
    X = (info[:, 0] - x_min) / (x_max - x_min)
    Y = (info[:, 1] - y_min) / (y_max - y_min)
    # 归一化后的坐标组合
    info2 = np.zeros([m * n, 3])
    info2[:, 0] = X
    info2[:, 1] = Y
    info2[:, 2] = info[:, 2]
    A = gauss_A(info2)
    b = gauss_b(info2)
    x = np.linalg.solve(A, b)
    x = [x[0,0], x[1,0], x[2,0], x[3,0]]
    sigma = np.sqrt(1 / (2 * abs(x[0])))
    x0 = -x[1] / (2 * x[0])
    y0 = -x[2] / (2 * x[0])
    X = x0 * (x_max - x_min) + x_min
    Y = y0 * (y_max - y_min) + y_min
    Sigma = np.sqrt((y_max - y_min) * (x_max - x_min)) * sigma
    para_A = np.exp(x[3] - (x[2] ** 2 + x[1] ** 2) / (4 * x[0]))
    return X, Y, Sigma, para_A


def gaussian(A, x0, y0, sigma, size_x, size_y):
    """
    制造在某一参数下Gaussian函数对应的离散子列
    :param A: 参数A
    :param x0: 参数均值x
    :param y0: 参数均值y
    :param sigma: 参数方差
    :param size_x: 离散点的尺寸，每个离散点为整数点
    :param size_y: 离散点的尺寸，每个离散点为整数点
    :return: 对应的Gaussian函数在离散点列处的值，结果为矩阵
    """
    f = np.zeros([size_x, size_y])
    for i in range(size_x):
        for j in range(size_y):
            f[i, j] = A * np.exp(-(i - x0) ** 2 / (2 * sigma ** 2) - (j - y0) ** 2 / (2 * sigma ** 2))
    return f


def compute_gauss_error(picture:np.ndarray, x0:int, y0:int, sigma:float, para_A:float):
    """
    判断gaussian拟合结果的好坏
    :param picture: 被拟合子图
    :param x0: 拟合参数x0
    :param y0: 拟合参数y0
    :param sigma: 拟合参数sigma
    :param para_A: 拟合参数A
    :return:均方根误差
    """
    m = picture.shape[0]
    n = picture.shape[1]
    f = gaussian(para_A, x0, y0, sigma, m, n)
    rmse = np.sqrt(sum(sum(np.power(f - picture, 2))) / (m * n))
    return rmse
