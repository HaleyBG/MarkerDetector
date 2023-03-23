import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import fundmental
import math


def img_hist(img: np.ndarray):
    """
    used to generate a histogram
    :param img: gray image
    :return:
    """
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()


def readname(filePath='/home/lthpc/Documents/Programs/WaveletDetectionChange/mrccut_old/small/'):
    name = os.listdir(filePath)
    return name


def two_value(p, ld):
    """
    图像二值化
    :param p: img needed 2 value
    :param ld: 阈值
    :return: P:阈值化后的image
    """
    p[p < ld] = 0
    p[p >= ld] = 1
    return p


def make_ori_template(r_img: int, ri: int):
    template = np.ones((r_img * 2 + 1, r_img * 2 + 1)).astype(np.float32)
    template = cv2.circle(template, (r_img, r_img), ri, (0., 0., 0.), -1, 8)
    return template


def ncc_value(img1, img2):
    """
    计算两个图片的相似性，两图尺寸相同
    :param img1:
    :param img2:
    :return:
    """
    img1 = img1.reshape(img1.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
    img2 = img2.reshape(img2.size, order='C')
    # np.corrcoef is return 2*2 shape metrix
    ncc = np.corrcoef(img1, img2)[0, 1]
    return ncc


def normalize(img: np.ndarray):
    max_img = img.max()
    min_img = img.min()
    img_norm = ((img - min_img) / (max_img - min_img)).astype(np.float32)
    return img_norm


def dis_p(point1, point2):
    """
    计算两点间距离
    :param point1:
    :param point2:
    :return:
    """
    dis = np.linalg.norm(point1 - point2, ord=None, axis=None, keepdims=False)
    return dis


def roundness(label_img):
    """
    计算圆度
    :type label_img: np.ndarray
    :return:
    """
    contours, hierarchy = cv2.findContours(np.array(label_img, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(contours[0])*4*math.pi
    b = math.pow(cv2.arcLength(contours[0],True), 2)
    if b == 0:
        return 0
    return a/b


def find_local_peak(img, m, n, m_w, n_w):
    sub_img= img[m:m + m_w, n:n + n_w]
    min_val, max_val, min_l, max_l = cv2.minMaxLoc(sub_img)
    peak_m = m + max_l[1]
    peak_n = n + max_l[0]
    return peak_m, peak_n


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(f"---new foler {path} ---")
    else:
        print(f"---There is this folder!---")

if __name__ == '__main__':
    # ori_img = cv2.imread('10111_19.jpg', 0)
    # hist = cv2.calcHist(ori_img, [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.xlim([0, 256])
    # plt.show()

    template = make_ori_template(4)
    fundmental.draw(template)
