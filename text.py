import numpy as np
import cv2
import time
import fundmental as fun
import wavelet as wave
import baseFun as bs
import math
import kd_tree as kd
import GaussFit as gs
import matplotlib.pyplot as plt


# 2Section:2.0，shape方法remove部分cubic使用圆度来进行，不使用模板匹配方法。
# 此代码中有全局变量：img_2value,radius,file
def template_make(img: np.ndarray, scale:int):
    """
    该流程中的img像素是正的（取反操作并入了waveletprocess）
    :param img:
    :return:
    """
    margin = 2  # 2
    # j = 3
    j = scale+1
    threshold_pixel = 0.5
    threshold_shape = 0.75#0.85
    img_m, img_n = img.shape
    threshold_remove = 4 if min(img_n, img_m) < 2000 else 8
    img = bs.normalize(img)

    wave_image, __ = wave.waveletprocess2(Image=img, J=j)
    fun.draw(wave_image, 'wave_image')
    img_2value = fun.hardval2(wave_image, 2)
    fun.removeSmall(img_2value, threshold_remove)
    fun.draw(img_2value, 'img_2value')

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_2value, connectivity=8, ltype=None)

    center_int = centroids.astype(int)
    fid_index = []
    # fid_circle = []
    # 随着后续remove调整前景状态
    for i in range(1, num_labels):
        sub_img = img[stats[i, 1]:stats[i, 1] + stats[i, 3] + 1, stats[i, 0]:stats[i, 0] + stats[i, 2]]
        sub_wave = img_2value[stats[i, 1]:stats[i, 1] + stats[i, 3] + 1, stats[i, 0]:stats[i, 0] + stats[i, 2]]
        mean, dev = cv2.meanStdDev(sub_img)

        # remove1: remove the sub_cubic by pixel，判断sub_img中点，均值mean，方差dev是否会合适
        if sub_img[center_int[i, 1] - stats[i, 1], center_int[i, 0] - stats[
            i, 0]] > threshold_pixel or mean > threshold_pixel or dev < 0.1:
            # img_2value[labels==i]=0
            continue

        # remove2: remove the roundness too small,第一个判断是把十分不合理的从小波图去除，第二个判断更强，表示可以用作模板的patches的要求更高
        round = bs.roundness(sub_wave)
        if round <= threshold_shape:#0.85:
            continue

        fid_index.append(i)

    # 对得到的fid_index进行统计直径
    d = []
    for i in fid_index:
        d.append(max(stats[i, 2], stats[i, 3]))
    # 进一步精确小波图（把像素点个数小于
    if len(d)==0:
        print('The wavelet detailed coefficients does not get proper information.')
        print('Please select option again!')
        return (0,0)
    d_mean = sum(d) / len(d)
    fun.removeSmall(img_2value, d_mean)

    hist, arr = np.histogram(d, bins=10, range=(min(d), max(d)))
    max_index = np.where(hist == max(hist))[0]
    max_index = max_index[0]
    temp = (arr[max_index] + arr[max_index + 1]) / 2
    temp = int(temp)
    d_end = temp + 2 * margin if temp % 2 == 1 else temp + 1 + 2 * margin
    r_end = int((d_end - 1) / 2)


    # 得到最终模板
    add_template = np.zeros(shape=(d_end, d_end), dtype=np.float32)
    num_template = 0
    for i in fid_index:
        # 判断是否是边缘点
        if center_int[i, 1] - r_end < 0 or center_int[i, 0] - r_end < 0 or center_int[i, 1] + r_end + 1 > img_m or \
                center_int[i, 0] + r_end + 1 > img_n:
            continue
        temp = img[center_int[i, 1] - r_end:center_int[i, 1] + r_end + 1,
               center_int[i, 0] - r_end:center_int[i, 0] + r_end + 1]
        # fun.draw(temp, f'temp_of_{i}')
        add_template = cv2.add(temp, add_template)
        num_template += 1

    template = add_template / num_template
    return template, img_2value


def get_ave_pixel(img, x, y, r):
    """
    找某点周围的像素均值
    :param img: 图像
    :param x: 小方块左上角点对应x
    :param y: 小方块左上角点对应y
    :param r: 方块半径
    :return: 方块中心点附近像素均值
    """
    m, n = img.shape
    d = 2 * r + 1
    thre_r = float(r * r) * 0.81
    center_x = x + r
    center_y = y + r
    value = 0.
    count = 0
    for j in range(x, x + d):
        if j > n:
            pass
        for i in range(y, y + d):
            if i > m:
                pass
            if (center_x - j) * (center_x - j) + (center_y - i) * (center_y - i) < thre_r:
                value += img[i, j]
                count += 1
    return value / count


def refine_fid_by_gaussian_distribution(candidate, candidate_corr, candidate_ave_pixel):
    """
    利用ncc*pixel的分布筛选candidate
    :param candidate:坐标集合，[x,y]
    :param candidate_corr:
    :param candidate_ave_pixel:
    :return:new_fid每一维数据分别是(x,y,ncc,avg_pixel,ncc*avg_pixel,index)
    """
    # 构造ncc*pixel
    num = len(candidate)
    new_score = []
    for i in range(num):
        new_score.append(candidate_corr[i] * candidate_ave_pixel[i])

    # new_score是评分的指标

    # 构造new_score的均值与标准差
    avg = 0.
    stdev = 0.
    for i in range(num):
        avg += new_score[i]
        stdev += new_score[i] * new_score[i]
    avg /= num
    stdev = math.sqrt(stdev / num - avg * avg)

    # 开始筛选
    new_fid = []
    fid_index = 0
    for i in range(num):
        if 255 in wave_img[candidate[i][1]:candidate[i][1] + 2 * radius, candidate[i][0]:candidate[i][0] + 2 * radius]:
            thre = avg + 0.5 * stdev
        else:
            thre = avg + 3 * stdev  # avg + 2 * stdev
        if new_score[i] > thre:
            new_fid.append(
                [candidate[i][0], candidate[i][1], candidate_corr[i], candidate_ave_pixel[i], new_score[i], fid_index])
            fid_index += 1
    return new_fid


def distribute_again(fid_info:list,new_new_fid:list=[]):
    """
    判断是否需要继续处理
    去除点分两步
    1.distribute again
    2.contrast
    :param fid_info:初始fid信息
    :param new_new_fid:经过筛选后的fid信息
    :return:是否经过进一步筛选
    """
    # 分析阶段
    new_fid = np.array(fid_info)
    new_fid_std = np.std(new_fid[:, 4])
    new_mean = np.mean(new_fid[:, 4])
    new_his = np.histogram(new_fid[:, 4], bins=15, range=(min(new_fid[:, 4]), max(new_fid[:, 4])))

    # 实验阶段
    if new_fid_std > 0.1:
        return 1
    else:
        return 0


def refine_fid_by_contrast(img:np.ndarray, fid:list, new_fid:list, contrast_list:list):
    """
    通过对比度筛选fid
    :param img: 识别的原图
    :param fid:数据结构：fid=(x,y,ncc,avg_pixel,ncc*avg_pixel,index)
    :param new_fid:输出的新fid
    :return:
    """
    contrast_threshold = 0.7 # 具体值还没确定
    fid_index=0
    for i in range(len(fid)):
        img_temp = img[fid[i][1]:fid[i][1]+2*radius+1,fid[i][0]:fid[i][0]+2*radius+1]
        contrast_i = calculate_contrast(img_temp)
        contrast_list.append(contrast_i)
        if contrast_i>contrast_threshold:
            temp_list = fid[i]
            temp_list[5] = fid_index
            new_fid.append(temp_list)
            fid_index+=1


def calculate_center_pixel(img:np.ndarray):
    """
    计算img的中心被0.7倍大小的方块覆盖区域的平均值
    :param img: 计算的图
    :return: 像素值平均值
    """
    d = img.shape[0]
    center = int((d - 1) / 2)
    sub_cubic_r = int(0.7 * d / 2)
    center_pixel = img[center - sub_cubic_r:center + sub_cubic_r, center - sub_cubic_r:center + sub_cubic_r].sum(
        axis=(0, 1)) / (2 * sub_cubic_r + 1) ** 2
    return center_pixel


def calculate_contrast(img:np.ndarray):
    """
    计算img的对比度,计算策略为：img的中心被0.7倍大小的方块覆盖区域的平均值与img周围像素点之间差的平方和
    :param img:
    :return:对比度
    """
    center_pixel = calculate_center_pixel(img)
    edge_list = []
    edge_list+=list(img[0][:])
    edge_list+=list(img[-1])
    edge_list+=list(img[1:-1,0])
    edge_list+=list(img[1:-1, -1])
    edge_list = np.array(edge_list)
    contrast = sum((edge_list-center_pixel)**2)
    return contrast


def draw_point(img: np.ndarray, cubic_points: list, r: int):
    """
    将cubic的点在画标注
    :param img:
    :param cubic_points: 方格的左上角
    :param r: 2r+1为方格长度
    :return:
    """
    num = len(cubic_points)
    for i in range(num):
        cv2.circle(img, (cubic_points[i][0] + r, cubic_points[i][1] + r), r, (1., 1., 1.))
    fun.draw(img,"in draw_point function")
    return img


def save_normalized_image(img: np.ndarray, file_name, file_path=f'./result/'):
    save_img = (255 * img).astype(np.uint8)
    cv2.imwrite(file_path + file_name, save_img)
    return 0


def chose_fid(img: np.ndarray, point, step, nccs: list, point_high_ncc: list,ncc_high:list):
    """
    判断point对应的cubic是不是含有fid，区分fid与背景噪点
    :param img:
    :param point:(x,y)
    :return:
    """
    # 确定取平均的方向
    m, n = img.shape
    mid_m = int(m / 2)
    mid_n = int(n / 2)
    if point[0] < mid_n:
        point1 = [point[0] + step, point[1]]
    else:
        point1 = [point[0] - step, point[1]]
    if point[1] < mid_m:
        point2 = [point[0], point[1] + step]
    else:
        point2 = [point[0], point[1] - step]
    point3 = [point1[0], point2[1]]

    # 取平均
    avg_subimg = make_avg_image(image=img, points=[point1, point2, point3], step=step)

    # 阈值
    thr = 0.5

    # 计算相似性
    ncc = ncc_value(avg_subimg, img[point[1]:point[1] + step, point[0]:point[0] + step])
    nccs.append(ncc)
    if ncc > 0.5:
        point_high_ncc.append(point)
        ncc_high.append(ncc)
    # 进行判断
    if ncc > thr:
        return 0
    else:
        return 1


def make_avg_image(image: np.ndarray, points: list, step: int):
    """
    将图片中的部分点处对应的cubic取平均,for chose_fid,make the avg_temp
    :param image:归一化了
    :param points:归一化了
    :return:
    """
    sum_img = np.zeros(shape=(step, step), dtype=np.float32)
    num = len(points)
    for i in range(num):
        sub_img = image[points[i][1]:points[i][1] + step, points[i][0]:points[i][0] + step]
        sum_img = cv2.add(sum_img, sub_img)
    avg_img = sum_img / num
    return avg_img


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
    ncc = np.corrcoef(img1, img2)[0, 1]  # 该函数比较两个向量的相似程度，计算的是皮尔逊相关系数
    return ncc


def find_fid_marker(img_ori: np.ndarray, template: np.ndarray, dense: int):
    """
    识别主步骤
    :param img_ori: 识别原图
    :param template: 模板生成部分生成的模板
    :param dense: 标记，没什么用
    :return: fid,np.ndarray,返回识别到的点
    """
    # 归一化与取反操作
    img = bs.normalize(img_ori)
    img = fun.Img_in2(img)

    template = bs.normalize(template)
    template = fun.Img_in2(template)

    # 模板匹配并归一化
    corr = cv2.matchTemplate(img, template, 3)
    corr = bs.normalize(corr)

    img_mean, img_std_dev = cv2.meanStdDev(img)
    corr_mean, corr_std_dev = cv2.meanStdDev(corr)

    img_threshold = img_mean # +0.1*img_std_dev
    corr_threshold = corr_mean + 1.5*corr_std_dev

    corr_m, corr_n = corr.shape
    temp_m, temp_n = template.shape

    global radius
    radius = int((temp_m - 1) / 2)
    # 候选点
    candidate = []
    candidate_index = 0
    candidate_corr = []
    candidate_ave_pixel = []
    img_paint_candidate = fun.Img_in2(img)
    for i in range(0, corr_m, temp_m):
        for j in range(0, corr_n, temp_n):
            peak_m, peak_n = bs.find_local_peak(corr, i, j, temp_m, temp_n)
            if corr[peak_m, peak_n] > corr_threshold and img[peak_m + radius, peak_n + radius] > img_threshold and 255 in wave_img[peak_m:peak_m+2*radius+1, peak_n:peak_n + 2*radius+1]:
                # 此处存放的x,y是每个方格的左上角
                candidate.append([peak_n, peak_m, candidate_index])
                candidate_corr.append(corr[peak_m, peak_n])
                candidate_ave_pixel.append(get_ave_pixel(img, peak_n, peak_m, radius))
                candidate_index += 1
                cv2.circle(img_paint_candidate, (peak_n + radius, peak_m + radius), radius, (1., 1., 1.))
            elif corr[peak_m, peak_n] > corr_threshold and img[peak_m + radius, peak_n + radius] > img_threshold+0.1*img_std_dev:
                # 此处存放的x,y是每个方格的左上角
                candidate.append([peak_n, peak_m, candidate_index])
                candidate_corr.append(corr[peak_m, peak_n])
                candidate_ave_pixel.append(get_ave_pixel(img, peak_n, peak_m, radius))
                candidate_index += 1
                cv2.circle(img_paint_candidate, (peak_n + radius, peak_m + radius), radius, (1., 1., 1.))
    fun.draw(img_paint_candidate, 'candidate')
    del img_paint_candidate

    # step1使用高斯筛选
    fid = candidate
    new_fid = refine_fid_by_gaussian_distribution(fid, candidate_corr, candidate_ave_pixel, )
    del candidate, candidate_corr, candidate_ave_pixel
    fid = new_fid
    # fid元素解释： fid=(x,y,ncc,avg_pixel,ncc*avg_pixel,index)

    # 当高斯筛选流程去除之后，需要加以下步骤完成数据转换
    # candidate = np.array(candidate)
    # fid = np.zeros((len(candidate), 6), dtype=int)
    # fid[:,:2] = candidate[:,:2]
    # fid[:,5] = candidate[:,2]
    # fid = fid.tolist()

    # 判断是否要对高斯筛选结果进一步筛选,判断变量为temp，若temp，则加两个筛选过程，否则不额外筛选。
    new_fid = []# new_fid没啥用,可以删除
    temp = distribute_again(fid)  #,new_fid)  # temp为是否运行refine_fid_by_contrast的指标
    if temp:
        new_fid = []
        contrast_list_show=[]
        refine_fid_by_contrast(img, fid, new_fid, contrast_list_show)
        fid = new_fid

    # step2若没有判断条件的代码
    # new_fid = []
    # contrast_list_show = []
    # refine_fid_by_contrast(img, fid, new_fid, contrast_list_show)
    # fid = new_fid

    # 作图保存数据
    img_paint = fun.Img_in2(img)
    draw_point(img_paint, fid, radius)
    del img_paint

    # 去除重复标记
    img_paint_remove_repeat = fun.Img_in2(img)
    dist_thr = 2 * radius
    node = kd.Node()
    new_fid = []
    kd.construct(d=2, data=fid.copy(), node=node, layer=0)
    for i in range(len(fid)):
        if fid[i][3] < 0:
            continue
        L = []  # 用来保存该点的最近邻
        kd.search(node=node, p=fid[i], L=L, K=5)

        for j in range(len(L)):
            if kd.distance(fid[i], L[j]) < dist_thr:
                fid[L[j][5]][3] = -1
        new_fid.append(fid[i])
        cv2.circle(img_paint_remove_repeat, [fid[i][0] + radius, fid[i][1] + radius], radius, (1., 1., 1.))
        kd.clear_flag(node=node)
    fun.draw(img_paint_remove_repeat,"remove repeat")
    fid = new_fid
    del img_paint_remove_repeat

    # step3去除背景识别点
    new_fid = []
    nccs = []
    point_high=[]
    ncc_high=[]
    for i in range(len(fid)):
        if chose_fid(img, fid[i], 2 * radius + 1, nccs, point_high, ncc_high):
            new_fid.append(fid[i])
    fid = np.array(new_fid)
    fid_xy = fid[:, :2].astype(int)
    score_xy = fid[:,2]

    # fid = np.array(fid)
    return fid_xy


def location_fid(img:np.ndarray, location:np.ndarray, width:int):
    """
    标记点定位阶段
    :param img: 定位的原图片
    :param location: 识别阶段的坐标点,都为左上角
    :param width: 进行圆心refine的子图的大小
    :return:refined_xy, 定位后fids的坐标
    """
    location = location.astype(int)
    img_inv = fun.Img_in(img)
    refined_xy = []
    score_xy = []
    for i in range(len(location)):
        sub_img = img_inv[location[i, 1]:location[i, 1]+width, location[i, 0]:location[i, 0]+width]
        if 0 in sub_img:
            sub_img[sub_img==0]=2

        row, colum, sigma, para_a = gs.compute_center_Gauss(sub_img)
        real_row = row + location[i, 1]
        real_colum = colum + location[i, 0]
        refined_xy.append(np.array([real_colum, real_row], dtype=int))
        score = gs.compute_gauss_error(sub_img, row, colum, sigma, para_a)
        score_xy.append(score)
    score_xy = np.array(score_xy)
    refined_xy = np.array(refined_xy)
    return refined_xy, score_xy


def score_ncc(img:np.ndarray, template:np.ndarray, location_xy:np.ndarray):
    """
    为每个坐标进行打分由于每个点的坐标进行过细化，导致部分点的识别点有部分在图外
    因此需要对原图的边界进行处理，并且所有的坐标索引都需要增加相应的常数，这里，
    单边边界宽度为8。
    :param img:原图
    :param template:模板图
    :param location_xy:坐标集合，坐标点为中点
    :return:打分数组,顺序与location_xy相同
    """
    infor_list = []
    img_big = np.zeros((img.shape[0]+16, img.shape[1]+16))
    img_big[8:img.shape[0]+8, 8:img.shape[1]+8]=img
    for i in range(len(location_xy)):
        x = int(location_xy[i,0])+8
        y = int(location_xy[i,1])+8
        sub_img = img_big[y-radius:y+radius+1, x-radius:x+radius+1]
        corr = bs.ncc_value(sub_img, template)
        infor_list.append(corr)
    infor_list = np.array(infor_list)
    return infor_list


def score_contrast(img:np.ndarray, location_xy:np.ndarray):
    """
    计算location_xy坐标的对比度打分
    :param img: 原图
    :param location_xy:需要打分的坐标
    :return: ndarray, 打分数组
    """
    img_big = np.zeros((img.shape[0]+16, img.shape[1]+16))
    img_big[8:img.shape[0]+8, 8:img.shape[1]+8] = img
    contrast_list = []
    for i in range(len(location_xy)):
        x = location_xy[i,0]+8
        y = location_xy[i,1]+8
        sub_img = img_big[y-radius:y+radius+1, x-radius:x+radius+1]
        contrast = calculate_contrast(sub_img)
        contrast_list.append(contrast)
    contrast_list = np.array(contrast_list)
    contrast_list = fun.ToOne(contrast_list)['image']
    return contrast_list


def score_wavelet(wavelet_img:np.ndarray, location_xy:np.ndarray):
    """
    计算小波是否显著，小波评分list元素为0或者1
    :param wavelet_img: 原图
    :param location_xy: 坐标，方块的中点
    :return: 小波评分list
    """
    score_wavelet = []
    for i in range(len(location_xy)):
        x = location_xy[i, 0]
        y = location_xy[i, 1]
        if wavelet_img[y, x]>200:
            score_wavelet.append(1)
        else:
            score_wavelet.append(0)
    score_wavelet = np.array(score_wavelet)
    return score_wavelet


def score_pixel(img:np.ndarray, location_xy:np.ndarray):
    """
    像素打分模式为像素值取反之后归一化，取值范围为0~1.
    :param img: 原图
    :param location_xy: 标记点中心点坐标
    :return: 依据像素点打分的list
    """
    img_big = np.zeros((img.shape[0]+16, img.shape[1]+16))
    img_big[8:img.shape[0]+8, 8:img.shape[1]+8] = img
    pixel_list = []
    for i in range(len(location_xy)):
        x = location_xy[i,0]+8
        y = location_xy[i,1]+8
        sub_img = img_big[y-radius:y+radius+1, x-radius:x+radius+1]
        p = calculate_center_pixel(sub_img)
        pixel_list.append(p)
    pixel_list = np.array(pixel_list)
    pixel_list = 1. - pixel_list
    return pixel_list


if __name__ == '__main__':
    # file_path = input('Please enter the path of images used to detect.')
    # result_folder = input('Please entry the path to save result of file.')
    file_path = './mrccut/small/deep3/'
    # file_path = './mrccut/text/'
    # file_path = './mrccut/small/deep4/'
    # file_path = './mrccut/big/deep4/'
    # file_path = './mrccut/resize_data/'
    file_name = bs.readname(file_path)
    result_folder = f"./result_{time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time()))}"
    bs.mkdir(result_folder)
    detail_file = open(f"{result_folder}/fiducial_information.txt", "a")
    detail_file.write(f"File Generation Time：{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}\n")
    fi = open(f"{result_folder}/general_information.txt", "a")
    fi.write(f"File Generation Time：：{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}\n")
    global file
    for file in file_name:
        detail_file.write(f"===============================\n")
        detail_file.write(f"This is {file} turn \n")
        fi.write(f"===============================\n")
        fi.write(f"This is {file} turn \n")
        print('================================')
        print(f'this is {file} turn...')
        img = cv2.imread(file_path + file, 0)
        if img is None:
            print(f'{file}不存在该路径！')
            break
        fun.draw(img,"the input image")
        print(f'the shape of original image is {img.shape}')
        # 通过数据集中胶体金颗粒的密集程度进行选择性放缩
        dense = eval(input(f'Is the size of fids in {file} is relatively small? If yes, please input 1, otherwise input 0.'))
        if dense:
            size = 2500
        else:
            size = 2000

        m, n = img.shape
        min_shape = min(m, n)
        # check if it is empty and too big to detect
        resize_index = 0
        if img is None:
            print('there is no figure!')
            break
        elif min_shape > size:
            for i in range(2, 7, 2):
                if int(min_shape / i) > size:
                    continue
                else:
                    mul_para = i
                    resize_index = 1
                    break
            # print(f"mul_para={mul_para}")
            img = cv2.resize(img, dsize=(int(n / mul_para), int(m / mul_para)), interpolation=cv2.INTER_AREA)
        fun.draw(img, 'after resize')
        print('The scale of detailed coefficients should be provide for a better results. ')
        print('The scale usually be chosen as 2 or 3. You can input 2 to test it.')
        scale = eval(input('The scale of detailed coefficients you want is: '))
        start = time.time()
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img1 = fun.ToOne(img)['image']


        global wave_img
        template1, wave_img = template_make(img, scale)
        fun.draw(wave_img, "output wave_image of template_make")
        fun.draw(template1, "template made from template_make")
        try:
            temp_m, __ = template1.shape
        except AttributeError:
            print(f'Now we will passed it.')
            continue
        global radius
        radius = int((temp_m - 1) / 2)
        fid= find_fid_marker(img, template1, dense)
        # 识别时间估计
        end = time.time()
        print(f'the detection time is {end - start}')
        fi.write(f"Detection time of {file} is {end-start}\n")
        # 定位阶段
        img255 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        start = time.time()
        refined_xy, score_error = location_fid(img, fid, len(template1))
        end = time.time()
        print(f'the location time is {end-start}')
        fi.write(f'the location time is {end-start}\n')

        print(f'The number of marker is {len(refined_xy)}')
        fi.write(f'the number of detected fiducial is {len(refined_xy)}\n')

        # 标记识别点
        for i in range(len(refined_xy)):
            cv2.circle(img255, refined_xy[i], radius, (0,0,255))
            detail_file.write(str(refined_xy[i])+'\n')
        fun.draw(img255, "the result after refined step")
        cv2.imwrite(f'{result_folder}/end_{file}', img255)

        # 还原大小
        if resize_index:
            temp_m, temp_n = template1.shape
            template = cv2.resize(template1, dsize=(mul_para * temp_n, mul_para * temp_m),
                                  interpolation=cv2.INTER_LINEAR)
        fun.draw(template1,"the real of template")
        fi.write(f'the shape of template is {template1.shape}\n')

    # fi.write(f"文件关闭==========时间：{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}\n")
    fi.close()