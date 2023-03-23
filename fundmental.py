# 用户名：lenovo
# 时间：2022/8/8  10:37
# 用途：
import numpy as np
import cv2

def hardvalsmall(w,t):
    '''
    硬阈值函数, 抹平小值
    :param w: 小波平面矩阵(ndarray)
    :param t: 阈值(float)
    :return: W:阈值后的小波平面矩阵(ndarray)
    '''
    w[w < t] = 0
    return w

def hardval2(p,ld):
    '''
    图像二值化，要求：矩阵元素大于零，若有负数，先求绝对值再进行二值化
    :param p: 相关系数矩阵(ndarray)
    :param ld: 阈值
    :return: P:阈值化后的相关系数矩阵
    '''
    temp = abs(p)
    temp[temp < ld] = 0
    temp[temp >= ld] = 255
    P = temp.astype(np.uint8)
    return P

def onmouse(event, x, y, flags, userdata):
    if event == 1:  # 点击鼠标左键时
        print("Row=", y, "Column=", x, "\n")  #
def draw(p, named="No Name", time=0):
    '''
    作图函数
    :param p: 待作图矩阵(ndarray)
    :param time: 图片存在时间
    :param named: 图像窗口名字，str
    :return:
    '''
    # cv2.namedWindow(named,cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback(named, onmouse,12)
    # cv2.imshow(named, p)
    # cv2.waitKey(time)
    # cv2.destroyAllWindows()
    return 0
def ToOne(img):
    '''
    图像归一化函数
    :param img: 输入图像
    :return:
    image:归一化后图像
    Max:归一化前图像的最大值
    Min:归一化前图像的最小值
    '''
    Max = img.max()
    Min = img.min()
    image = (img - img.min())/(img.max()-img.min())
    res = {'image': image, 'min': Min, 'max': Max, 'oldimage': img}
    return res


def Img_in(img):
    '''
    图像取反
    :param img: 需要取反的图像,像素范围为0——255
    :return: Temp: 取反之后的图像
    '''
    m = img.shape[0]
    n = img.shape[1]
    temp = 255*np.ones((m, n))
    Temp = temp - img
    return Temp


def Img_in2(img):
    '''
    图像取反,the normalized image
    :param img: 需要取反的图像,像素范围为0——1
    :return: Temp: 取反之后的图像
    '''
    m = img.shape[0]
    n = img.shape[1]
    temp = np.ones((m, n), dtype=np.float32)
    Temp = temp - img
    return Temp
#
# def removeNoise(image):
#     '''
#     去除二值图像中的噪声点，注意二值图像的最大值为255
#     如果图像中没有什么细小噪声点，就不要用，不然会将所有点都去除
#     （点与噪声之间没有区分时不建议用）
#     :param image:
#     :return:去除噪声点之后的图像
#     '''
#     imgTmp = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
#     imgTmp1 = 255 - imgTmp
#     kernel = np.ones([5, 5])/25
#     imgTmp2 = cv2.filter2D(imgTmp1, -1, kernel)
#     imgTmp3 = 255 - imgTmp2
#     imgTmp4 = cv2.threshold(imgTmp3, 0, 255, cv2.THRESH_BINARY)
#     img = imgTmp4[1].astype(np.uint8)
#     return img


# def madePicture2(a,loc):
#     '''
#     取子图函数，从a中提取最小的包含loc的矩形子图
#     :param a: 原图像
#     :param loc: 需要提取的元素坐标
#     :return: 子图以及子图区域对应的坐标
#     '''
#     loc = loc.astype(np.int)
#     m = a.shape[0]
#     n = a.shape[1]
#     x_min = loc[:, 0].min()
#     x_max = loc[:, 0].max()
#     y_min = loc[:, 1].min()
#     y_max = loc[:, 1].max()
#     if x_min-1>0 and y_min-1>0 and x_max+2<m and y_max+2<n:
#         sub_loc = np.array(
#             [[i, j] for i in range(x_min-1, x_max + 2) for j in range(y_min-1, y_max + 2)]
#         )
#         sub_p = a[x_min-1:x_max + 2, y_min-1:y_max + 2]  # 防止信息的不完整
#     elif x_min-2>0 and y_max+3<n:
#         sub_loc = np.array(
#             [[i, j] for i in range(x_min - 2, x_max + 1) for j in range(y_min , y_max + 3)]
#         )
#         sub_p = a[x_min-2:x_max + 1, y_min:y_max + 3]  # 防止信息的不完整
#     elif x_max+3<m and y_max+3<n:
#         sub_loc = np.array(
#             [[i, j] for i in range(x_min , x_max + 3) for j in range(y_min , y_max + 3)]
#         )
#         sub_p = a[x_min:x_max + 3, y_min:y_max + 3]  # 防止信息的不完整
#     elif x_max+3<m and y_min-2>0:
#         sub_loc = np.array(
#             [[i, j] for i in range(x_min , x_max + 3) for j in range(y_min-2 , y_max + 1)]
#         )
#         sub_p = a[x_min:x_max + 3, y_min-2:y_max + 1]  # 防止信息的不完整
#     elif x_min-2>0 and y_min-2>0:
#         sub_loc = np.array(
#             [[i, j] for i in range(x_min-2 , x_max +1) for j in range(y_min-2 , y_max + 1)]
#         )
#         sub_p = a[x_min-2:x_max + 1, y_min-2:y_max + 1]  # 防止信息的不完整
#     return sub_p, sub_loc
def drawPointr(a, dicRC, fontsize=False):
    '''
    依据字典中的半径值，将字典中存在的中心点在图像中标记出来
    :param a: 原图像（画布）
    :param dicRC: 中心点以及半径的字典信息
    :param fontsize: 设置显示字体的字号大小，默认不显示
    :return:
    '''
    for i in dicRC:
        x = dicRC[i]['center'][0]
        y = dicRC[i]['center'][1]
        r = int(dicRC[i]['r'] + 2)
        cv2.circle(a, [y, x], r, (10, 10, 255))
        if bool(fontsize):
            cv2.putText(a, '(%s,%s)'%(y, x), [y, x], cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 1)
    draw(a, 'the result of detection')

def removeSmall(img, threshold):
    '''
    :param img: 需要去除小区域的图片
    :param threshold: 小区域最小像素个数
    :return:
    '''
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    contours, hierarch = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < threshold:
            cv2.drawContours(img, [contours[i]], 0, 0, -1)


#
# def pixel_trans_log(img):
#     imginfo = ToOne(img)
#     temp = np.log(imginfo['image']+1)
#     resinfo = ToOne(temp)
#     res = (resinfo['image']*255).astype(np.uint8)
#     return res
# def pixel_trans_times(img):
#     imginfo = ToOne(img)
#     temp = (imginfo['image']+200)**2
#     resinfo = ToOne(temp)
#     res = (resinfo['image']*255).astype(np.uint8)
#     return res
# def pixel_trans_exp(img):
#     imginfo = ToOne(img)
#     temp = np.exp(imginfo['image'])
#     resinfo = ToOne(temp)
#     res = (resinfo['image']*255).astype(np.uint8)
#     return res


def avgImg(image, k):
    '''
    对图像进行滑动平均
    :param image: 需要平均的图片
    :param k: 平均核边长（奇数）
    :return: 平滑之后的图像
    '''
    kernel = np.ones([k, k])/(k*k)
    filterImg = cv2.filter2D(image, -1, kernel)
    filterImgInfo = ToOne(filterImg)
    res = filterImgInfo['image']
    return res
if __name__=='__main__':
    image = cv2.imread('../data1cut_deres.jpg', 0)
    res = avgImg(image, 3)
    draw(res)
