# 用户名：lenovo
# 时间：2022/8/10  9:33
# 用途：
import numpy as np
import cv2
from sklearn.cluster import MeanShift
import fundmental as fun
import matplotlib.pyplot as plt
import GaussFit as gs
import templateMatch as tp
import wavelet as wl
# 聚类主要部分
def compute_distance(location):
    '''
    计算距离矩阵函数
    :param location: 需要计算距离矩阵的坐标集合
    :return:distancne:距离矩阵(上三角矩阵），其他位置为0
    '''
    n = location.shape[0]
    distance = np.zeros([n, n])
    for i in range(n-1):
        for j in range(i+1, n):
            distance[i, j]=np.sqrt(sum(np.power(location[i]-location[j], 2)))
    return distance
# def hiClustering(locations):
#     '''
#     关于坐标的层次聚类算法
#     :param locations: 需要进行聚类分析的坐标集合，size = n*2
#     :return:
#     team: 组队情况,其中0值为没有组队成功的点，为孤立点
#     noteam:没有组队的点所在的索引（元组）
#     '''
#     n = locations.shape[0]
#     team = np.zeros([n])
#     i = 1
#     D = compute_distance(locations)
#     D[D == 0] = 1e5
#     #当存在距离足够小（距离小于2的粒子对时才进行合并）
#     while (D.min()<2):
#         X, Y = np.where(D == D.min())
#         for k in range(len(X)):
#             x = X[k]
#             y1 = Y[k]
#             if team[x] == 0 and team[y1] == 0:
#                 team[x] = i;team[y1] = i
#                 i = i+1
#             elif team[x] == 0:
#                 team[x] = team[y1]
#             elif team[y1] == 0:
#                 team[y1] = team[x]
#             else:
#                 team[team == team[x]] = team[y1]
#             D[x, y1] = 1e5
#         noteam = np.array([])
#     if team.min() == 0:
#         print('仍有没有组队的粒子点')
#         noteam = np.where(team == 0)
#     return team, noteam
# def picked(P):
#     '''
#     挑选出小波图的非零点所在的坐标
#     :param P: 小波图
#     :return:
#     '''
#     pick_x, pick_y = np.where(P >= 250)
#     n = pick_y.shape[0]
#     locations = np.zeros([n,2])
#     for i in range(n):
#         locations[i, :] = [pick_x[i], pick_y[i]]
#     return locations

# def TwovalueHiclustering(image, small_region = 3):
#     '''
#     二值图像中亮斑的聚类，其中包括了去除孤立点以及去除小区域点
#     :param image: 二值图像
#     :param small_region: 多小个像素算小区域？该数字以下的区域均被视为小区域
#     :return: 去掉小区域之后的结果；team_dir聚类结果的字典（组成亮斑的所有像素集合分别属于第几组）
#     '''
#     locations = picked(image)
#     team, noteam = hiClustering(locations)
#     if np.array(list(noteam)).size > 0:  # 如果有没组队的点
#         discrete_point_image = image.copy()  # 用来展示孤立点的画布
#         temp = locations[noteam, :][0]  # 没组队点的坐标
#         n = temp.shape[0]  # 没组队点的个数
#         for k in range(n):
#             cv2.circle(discrete_point_image, temp[k, :].astype(np.int), 5, (225, 225, 0))#圈出没组队的点
#         fun.draw(discrete_point_image, "the single point without team")
#         temp = locations[noteam]
#         print('没组队的点坐标为：')
#         print(temp)
#         print('为了程序继续进行，默认把孤立白色粒子作为奇异点，并忽视处理（认为他们不再是白色粒子）')
#         for i in temp:
#             image[int(i[0]), int(i[1])]=0  # 把孤立点从图中抹去
#     else:  # 如果没有孤立点
#         print('所有白色像素点均聚合到某个类中')
#     team_dir = {}
#     num = 0
#     fun.draw(image, 'before preclude the small region')
#     for i in list(set(team)):
#         # 检测是不是为孤立点
#         if i == 0.:
#             continue
#         # 计算粒子半径的方法
#         temp = np.array(list(np.where(team == i)))[0]  # temp为队伍为i的索引,一维索引
#         loc = locations[temp, :]  # loc为本类坐标
#         if len(temp) < small_region:
#             for i in loc:
#                 image[int(i[0]), int(i[1])] = 0
#             continue
#         team_dir[num] = loc
#         num += 1
#     fun.draw(image, 'without small region')
#     return image, team_dir

def hicluster(img, r0):
    x, y = np.where(img > 0)
    local = np.array([x, y]).T
    ms = MeanShift(bandwidth=r0, bin_seeding=True)
    ms.fit(local)

    # 每个点的标签
    labels = ms.labels_

    # 粒子中心的点的集合
    cluster_cennters = ms.cluster_centers_

    # 总共的标签分类
    labels_unique = np.unique(labels)

    # 分类个数
    n_clusters = len(labels_unique)

    clus_team_dir = {}

    for k in range(n_clusters):
        my_menbers = labels == k
        cluster_cennter = [int(cluster_cennters[k][0]), int(cluster_cennters[k][1])]
        clus_team_dir[k] = [cluster_cennter, local[my_menbers]]
    return clus_team_dir





if __name__=='__main__':
    img = cv2.imread('../twovalue.jpg', 0)
    hicluster(img, 6)