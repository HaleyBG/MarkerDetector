# 用户名：lenovo
# 时间：2022/8/8  10:38
# 用途：
import numpy as np
import cv2
import fundmental as fun


def corv(s, k):
    """
    1d卷积函数，被atrous调用
    :param s: 一维信号,长度l(ndarray)
    :param k: 卷积核(奇数)(ndarray)
    :return: Line:卷积结果,长度l
    """
    line = np.convolve(s, k, mode='valid')
    return line


def fullZero2(k, lk):
    """
    优化后的卷积核补零函数
    :param k:
    :param lk:
    :return:
    """
    K = np.zeros([lk, 2])
    K[:, 0] = k
    temp = K.flatten()
    res = temp[0:2 * lk - 1]
    return res


def atrous_cv(ai, k):
    k1 = k.reshape((-1, 1))
    k2 = k.reshape((1, -1))
    K = np.kron(k1, k2)
    Trous = cv2.filter2D(ai, -1, K, borderType=cv2.BORDER_REFLECT)
    return Trous


def wavelet(a, A):
    """
    小波图像计算函数
    :param a: 尺度i的近似图像
    :param A: 尺度i+1的近似图像
    :return: W:尺度i+1的经过阈值化后的小波图像
    """
    w = a - A
    (m, n) = w.shape
    # 计算小波系数的阈值
    temp = np.median(np.abs(np.ones((m, n)) * np.median(w) - w))
    t = temp * 3 / 0.67
    W = fun.hardvalsmall(w, t)
    return W


def releimage(setw):
    """
    相关图像计算函数(内含阈值化)
    :param setw: 小波图像集合（阈值化后的）
    :return: temp：小波图像的乘积（相关系数矩阵）
    """
    (h, m, n) = setw.shape
    temp = np.ones((m, n))
    for i in range(h):
        temp *= setw[i]
    return temp


def waveletprocess2(Image, J=3, k=np.array([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])):
    """
    小波变换处理函数,小波相似图片只是由最后一张小波图制作
    :param Image: 需要处理的图像(不需要进行归一化，累乘会使像素值非常小，像素值彼此之间差距过大）
    :param k: 小波变换所用核
    :param J: 小波变换的尺度
    :return: 小波相关图像
    """
    if J == 1:
        print('J的值需要大于1！')
        return
    Image = (Image * 255).astype(np.uint8)
    # 图像黑白反转,结果是反转后元素为0-255的图像
    inImage = fun.Img_in(Image)
    a = inImage.astype(np.uint8)
    m, n = a.shape
    # Ai记录所有近似图像
    Ai = np.zeros((J, m, n))
    Ai[0] = a
    # 计算近似图像组
    for i in range(J):
        if i == 0:
            continue
        else:
            Ai[i] = atrous_cv(Ai[i - 1], k)
            lk = len(k)
            k = fullZero2(k, lk)
    # 计算小波图像组
    Wi = np.zeros((J - 1, m, n))
    for i in range(J - 1):
        Wi[i] = wavelet(Ai[i], Ai[i + 1])
        # fun.draw(Wi[i], 'the %sst pic' % (i))

    # 计算相关图像
    wave_image = Wi[J - 2]
    return wave_image, Wi[0]

if __name__ == '__main__':
    '''
    结合使用小波图像1和2合成corr
    '''
    image = cv2.imread('../enddata/3.20.1.1.jpg', 0)
    wavedir = waveletprocess2(image, 4)
    # fun.draw(wavedir['corr'], 'wavelet1 plan')
    sigma1 = np.std(wavedir[1])
    sigma2 = np.std(image)
    threshold = 2 * sigma1
    res = fun.hardval2(wavedir['corr'], threshold)
