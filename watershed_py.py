'''
Author: Hou Gaoxin 710872687@qq.com
Date: 2023-05-13 14:34:16
LastEditors: Hou Gaoxin 710872687@qq.com
LastEditTime: 2023-10-01 22:50:28
FilePath: /Program_detection_program/watershed_py.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
import fundmental as fun


def watershed_py(oriImg, thresh, f=None):
    '''
    分水岭算法
    :type img: object
    :param oriImg:the original image
    :param img:wavelet image
    :param para:make up the parameter of threshold
    :return:
    '''


    # open operator
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # dilate and erode
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    sure_fg = cv2.erode(opening, kernel, iterations=1)
    # fun.draw(sure_fg, 'erode')
    # fun.draw(sure_bg, 'dilate')

    # get the area map
    # unknown area
    unknown = cv2.subtract(sure_bg, sure_fg)
    # foreground
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    # unknown area
    markers[unknown == 255] = 0
    # the area map
    markers_copy = markers.copy()
    markers_copy[markers == 0] = 150  # 灰色为不确定区域
    markers_copy[markers == 1] = 0  # 黑色是背景
    markers_copy[markers > 1] = 255  # 白色为前景

    # show
    markers_copy = np.uint8(markers_copy)
    # fun.draw(markers_copy, 'markers_copy')
    # fun.draw(oriImg, 'before oriImage')

    # watershed
    oriImg = np.uint8(oriImg * 255)
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_GRAY2BGR)
    # fun.draw(oriImg, 'after oriImage')
    markers = cv2.watershed(oriImg, markers)

    # remove small region
    region = markers.copy()
    region[region != 0] = 255
    region = region.astype(np.uint8)
    # function of remove small region
    contours, hierarch = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 20:
            cv2.drawContours(region, [contours[i]], 0, -1, -1)

    # draw the picture
    copy_img = np.zeros(oriImg.shape, dtype=np.uint8)
    copy_img[markers == -1] = (255,255,255)
    copy_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(copy_img, connectivity=8, ltype=None)
    if f:
        f.write(f'The number of labels is {num_labels}.\n')
    oriImg[markers == -1] = (0, 255, 0)
    # fun.draw(oriImg)
    return oriImg
