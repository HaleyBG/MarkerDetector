# # 用户名：lenovo
# # 时间：2022/8/10  9:34
# # 用途：
import numpy as np


def expandImage(image, line_l, row_l):
    '''
    图像延拓（填充0延拓）
    :param image: 被延拓的图像
    :param line_l: 行单边延拓数
    :param row_l: 列单边延拓数
    :return: 延拓后的函数
    '''
    m = image.shape[0]
    n = image.shape[1]
    temp = np.zeros([int(m + 2 * line_l), int(n + 2 * row_l)])
    temp[int(line_l):int(line_l + m), int(row_l):int(row_l + n)] = image
    return temp


def corr_p_p(p0, p1):
    img0 = p0.reshape(p0.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
    img1 = p1.reshape(p1.size, order='C')
    corr = np.corrcoef(img0, img1)[0, 1]
    return corr