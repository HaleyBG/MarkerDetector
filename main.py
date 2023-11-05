'''
@Project ：Program_detection 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：Haley
@Date    ：2023/5/5 下午3:32
@Usage   :
'''

import numpy as np
import cv2
import time
import fundmental as fun
import wavelet as wave
import baseFun as bs
import kd_tree as kd
import GaussFit as gs
import os
import matplotlib.pyplot as plt
import mrc2jpg as mj
import sys
import mrcfile as mf
import math
import argparse

def template_make(img: np.ndarray, scale: int):
    """
    :param img:
    :return:
    """
    margin = 2  # the margin of template of fiducial marker
    # j = 3
    j = scale + 1
    threshold_pixel = 0.5
    threshold_shape = 0.75  # 0.85
    img_m, img_n = img.shape
    # threshold_remove = 4 if min(img_n, img_m) < 2000 else 8
    threshold_remove = 4 if min(img_n, img_m) < 1500 else 8
    img = bs.normalize(img)

    wave_image, __ = wave.waveletprocess2(Image=img, J=j)
    wave_ori = fun.hardval2(wave_image, 2)
    img_2value = wave_ori.copy()
    fun.removeSmall(img_2value, threshold_remove)
    if save_img:
        if wave_ori.dtype == 'float32':
            wave_ori_temp = (wave_ori * 255).astype(np.uint8)
            cv2.imwrite(f"{result_folder}/wave_original_img_{save_name}", wave_ori_temp)
        else:
            cv2.imwrite(f"{result_folder}/wave_original_img_{save_name}", wave_ori)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_2value, connectivity=8, ltype=None)

    center_int = centroids.astype(int)
    fid_index = []
    for i in range(1, num_labels):
        sub_img = img[stats[i, 1]:stats[i, 1] + stats[i, 3] + 1, stats[i, 0]:stats[i, 0] + stats[i, 2]]
        sub_wave = img_2value[stats[i, 1]:stats[i, 1] + stats[i, 3] + 1, stats[i, 0]:stats[i, 0] + stats[i, 2]]
        mean, dev = cv2.meanStdDev(sub_img)

        # remove1: remove the sub_cubic by pixel
        if sub_img[center_int[i, 1] - stats[i, 1], center_int[i, 0] - stats[
            i, 0]] > threshold_pixel or mean > threshold_pixel or dev < 0.05:
            continue

        # remove2: remove the roundness too small
        round = bs.roundness(sub_wave)
        # print(f"round={round}")
        if round <= threshold_shape: 
            continue

        fid_index.append(i)

    d = []
    for i in fid_index:
        d.append(max(stats[i, 2], stats[i, 3]))

    if len(d) == 0:
        print('The wavelet detailed coefficients do not get proper information.')
        print('Please select the scale again.')
        return (0, 0, 0)
    d_mean = sum(d) / len(d)
    fun.removeSmall(img_2value, int(d_mean * 0.8))
    if show_img:
        fun.draw(wave_image, 'wave_image')
        fun.draw(img_2value, 'img_2value')
    if save_img:
        if img_2value.dtype == 'float32':
            wave_img2_temp = (img_2value * 255).astype(np.uint8)
            cv2.imwrite(f'{result_folder}/wave_img_{save_name}', wave_img2_temp)
        else:
            cv2.imwrite(f'{result_folder}/wave_img_{save_name}', img_2value)

    hist, arr = np.histogram(d, bins=10, range=(min(d), max(d)))
    max_index = np.where(hist == max(hist))[0]
    max_index = max_index[0]
    diameter_float = (arr[max_index] + arr[max_index + 1]) / 2
    temp = int(diameter_float)
    d_end = temp + 2 * margin if temp % 2 == 1 else temp + 1 + 2 * margin
    r_end = int((d_end / 2) + 0.5)

    # Get the template 
    add_template = np.zeros(shape=(2 * r_end + 1, 2 * r_end + 1), dtype=np.float32)
    num_template = 0
    for i in fid_index:
        if center_int[i, 1] - r_end < 0 or center_int[i, 0] - r_end < 0 or center_int[i, 1] + r_end + 1 > img_m or \
                center_int[i, 0] + r_end + 1 > img_n:
            continue
        temp = img[center_int[i, 1] - r_end:center_int[i, 1] + r_end + 1,
               center_int[i, 0] - r_end:center_int[i, 0] + r_end + 1]
        # fun.draw(temp, f'temp_of_{i}')
        add_template = cv2.add(temp, add_template)
        num_template += 1

    if num_template==0:
        return 0,0,0
    template = add_template / num_template
    wavelet_image = img_2value
    return template, wavelet_image, d_end


def template_average(template: np.ndarray):
    """
    This function used to average template to make the center point more center
    :param template:
    :return:
    """
    # dividing height and width by 2 to get the center of the image
    height, width = template.shape[:2]

    # get the center coordinates of the image to create the 2D rotation matrix
    center = ((width - 1) / 2, (height - 1) / 2)

    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)

    # rotate the image using cv2.warpAffine
    rotated_image1 = cv2.warpAffine(src=template, M=rotate_matrix, dsize=(width, height))
    rotated_image2 = cv2.warpAffine(src=rotated_image1, M=rotate_matrix, dsize=(width, height))
    rotated_image3 = cv2.warpAffine(src=rotated_image2, M=rotate_matrix, dsize=(width, height))

    template_new = np.add(template, rotated_image1)
    template_new = np.add(template_new, rotated_image2)
    template_new = np.add(template_new, rotated_image3)
    template_new = template_new / 4
    return template_new


def get_ave_pixel(img, seed_x, seed_y, r):
    """
    calculate the average of pixel
    :param img: 
    :param seed_x: The x of coordinate in upper left corner(square)
    :param seed_y: The y of coordinate in upper left corner(square)
    :param r: radius of square
    :return: average of square around the center pixel
    """
    # m, n = img.shape
    d = 2 * r + 1
    thre_r = float(r * r) * 0.36  # 0.9r
    center_x = seed_x + r
    center_y = seed_y + r
    value = 0.
    count = 0
    for x in range(seed_x, seed_x + d):
        for y in range(seed_y, seed_y + d):
            if (center_x - x) * (center_x - x) + (center_y - y) * (center_y - y) < thre_r:
                value += img[y, x]
                count += 1
    return value / count


def refine_fid_by_gaussian_distribution_markerauto_wave(candidate):
    """
    Use ncc*pixel to filter candidates with detailed coefficients
    :param candidate:information of candidates: [x,y1,ncc,pixel,none,index]
    :return:new_information of candidates: (x,y1,ncc,avg_pixel,ncc*avg_pixel,index)
    """
 
    num = len(candidate)
    new_score = []  
    for i in range(num):
        new_score.append(candidate[i][2] * candidate[i][3])

    new_score_np = np.array(new_score)
    avg = np.mean(new_score_np)
    stdev = np.std(new_score_np)

    thre = avg - 0.5 * stdev

    if show_plt:
        temp, _ = np.histogram(new_score, bins=50)
        max_temp = max(temp)
        max_temp = int(1.1*max_temp)
        fig, axes = plt.subplots()
        axes.hist(new_score, bins=50)
        axes.vlines(thre, 0, max_temp, linestyles='dashed', colors='red', label=r'$\mu_{np}-0.5\sigma_{np}$')
        axes.set_xlabel(r"NCC$\times$pixel")
        axes.set_ylabel(f"Number")
        axes.set_title(f"Distribution of candidates in {file_name.split('.')[0]}")
        axes.legend()
        fig.show()

    new_fid = []
    fid_index = 0
    for i in range(num):
        if new_score[i] > thre:
            new_fid.append(
                [candidate[i][0], candidate[i][1], candidate[i][2], candidate[i][3], new_score[i], fid_index])
            fid_index += 1
    return new_fid


def refine_fid_by_gaussian_distribution_markerauto_no_wave(candidate):
    """
    Use ncc*pixel to filter candidates without detailed coefficients
    :param candidate:information of candidates: [x,y1,ncc,pixel,none,index]
    :return:new_information of candidates: (x,y1,ncc,avg_pixel,ncc*avg_pixel,index)
    """
    # generate ncc*pixel
    num = len(candidate)
    new_score = []  
    for i in range(num):
        new_score.append(candidate[i][2] * candidate[i][3])

    new_score_np = np.array(new_score)
    avg = np.mean(new_score_np)
    stdev = np.std(new_score_np)

    thre = avg + 3 * stdev

    if show_plt:
        temp, _ = np.histogram(new_score, bins=50)
        max_temp = max(temp)
        max_temp = int(1.1*max_temp)
        fig, axes = plt.subplots()
        axes.hist(new_score, bins=50)
        axes.vlines(thre, 0, max_temp, linestyles='dashed', colors='red', label=r'$\mu_{np}+3\sigma_{np}$')
        axes.set_xlabel(r"NCC$\times$pixel")
        axes.set_ylabel(f"Number")
        axes.set_title(f"Distribution of candidates without information in {file_name.split('.')[0]}")
        axes.legend()
        fig.show()

    new_fid = []
    fid_index = 0
    for i in range(num):
        if new_score[i] > thre:
            new_fid.append(
                [candidate[i][0], candidate[i][1], candidate[i][2], candidate[i][3], new_score[i], fid_index])
            fid_index += 1
    return new_fid


def draw_point(img: np.ndarray, cubic_points: list, r: int, color: tuple = (0, 255, 0)):

    if not cubic_points.any():
        return 0
    num = len(cubic_points)
    if img.dtype == 'float32':
        for i in range(num):
            cv2.circle(img, (cubic_points[i][0], cubic_points[i][1]), r, (1., 1., 1.), 2)
    else:
        for i in range(num):
            cv2.circle(img, (cubic_points[i][0], cubic_points[i][1]), r, color, 2)
    if show_img:
        fun.draw(img, "in draw_point function")


def remove_by_ncc_in_end(fid):
    """
    filter by ncc in the end
    """
    global hyperparameter_ncc
    new_fid = []
    ncc_threshold = hyperparameter_ncc
    index = 0
    for i in range(len(fid)):
        if fid[i][2]>=ncc_threshold:
            temp = fid[i]
            temp[5] = index
            index += 1
            new_fid.append(temp)
    return new_fid

def markerauto_work_flow(img_ori: np.ndarray, template_ori: np.ndarray):
    """
    markerauto
    :param img_ori:
    :param template_ori:
    :return:
    """
    img_draw = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2RGB)

    img = bs.normalize(img_ori)
    img = fun.Img_in2(img)

    template = bs.normalize(template_ori)
    template = fun.Img_in2(template)

    corr = cv2.matchTemplate(img, template, 3)
    corr = bs.normalize(corr)

    # ======================
    # Candidiate generation
    start_time = time.time()
    img_mean, img_std_dev = cv2.meanStdDev(img)
    corr_mean, corr_std_dev = cv2.meanStdDev(corr)

    img_threshold = int(img_mean.squeeze()*10)/10
    # img_threshold = int(img_mean*10)/10
    corr_threshold = int((corr_mean.squeeze() + 2 * corr_std_dev.squeeze())*10)/10
    # corr_threshold = int((corr_mean + 2 * corr_std_dev)*10)/10

    idiameter = int(2 * radius_int + 1)

    candidate_no_wave = []
    candidate_wave = []
    no_index = 0
    wave_index = 0
    img_draw_temp = img_draw.copy()
    corr_m, corr_n = corr.shape
    remove_point = []
    for i in range(0, corr_m, idiameter):
        for j in range(0, corr_n, idiameter):
            peak_m, peak_n = bs.find_local_peak(corr, i, j, idiameter, idiameter)
            if corr[peak_m, peak_n] > corr_threshold and get_ave_pixel(img, peak_n, peak_m, radius_int) > img_threshold:
                if 255 in wave_img[peak_m:peak_m + 2 * radius_int + 1, peak_n:peak_n + 2 * radius_int + 1]:
                    candidate_wave.append(
                        [peak_n, peak_m, corr[peak_m, peak_n], get_ave_pixel(img, peak_n, peak_m, radius_int), 1,
                         wave_index])
                    
                    wave_index += 1
                    cv2.circle(img_draw_temp, (peak_n + radius_int, peak_m + radius_int), radius_int,
                               (0, 255, 0), 2)
                else:
                    candidate_no_wave.append(
                        [peak_n, peak_m, corr[peak_m, peak_n], get_ave_pixel(img, peak_n, peak_m, radius_int), 1,
                         no_index])
                    # fids = [x, y1, corr, pixel, none, index]
                    no_index += 1
                    cv2.circle(img_draw_temp, (peak_n + radius_int, peak_m + radius_int), radius_int,
                               (0, 0, 255), 2)
            else:
                remove_point.append([peak_n, peak_m])

    if save_img:
        if img_draw_temp.dtype == 'float32':
            img_draw_temp = (255 * img_draw_temp).astype(np.uint8)
            cv2.imwrite(f'{result_folder}/candidate1_generation_{save_name}', img_draw_temp)
        else:
            cv2.imwrite(f'{result_folder}/candidate1_generation_{save_name}', img_draw_temp)

    fid_no_wave = candidate_no_wave
    fid_wave = candidate_wave
    end_time = time.time()
    information_file.write(f"The time of the second module is {end_time - start_time}.\n")
    information_file.write(
        f"The number of fiducial markers in the second module is {len(candidate_wave) + len(candidate_no_wave)}\n")

    # ==============================
    # step1 Gaussian distribution
    start_time = time.time()
    new_fid_no_wave = refine_fid_by_gaussian_distribution_markerauto_no_wave(fid_no_wave)
    new_fid_temp = new_fid_no_wave + fid_wave
    new_fid = refine_fid_by_gaussian_distribution_markerauto_wave(new_fid_temp)
    # Explain the composition of fid: fid=(x,y1,ncc,avg_pixel,ncc*avg_pixel,index)
    if save_img:
        img_draw_temp = img_draw.copy()
        location_xy2 = np.array(new_fid)[:, :2].astype(int) + radius_int
        draw_point(img_draw_temp, location_xy2, radius_int, color=(0, 255, 0))

        if img_draw_temp.dtype == 'float32':
            img_draw_temp = (255 * img_draw_temp).astype(np.uint8)
            cv2.imwrite(f'{result_folder}/candidate2_gauss_{save_name}', img_draw_temp)
        else:
            cv2.imwrite(f'{result_folder}/candidate2_gauss_{save_name}', img_draw_temp)
    end_time = time.time()
    information_file.write(
        f"The time of the third module is {end_time - start_time}\n")
    information_file.write(f"The number of fidicual markers in the third module is {len(new_fid)}\n")
    fid = new_fid


    # ==========================
    # step2 NCC filter in the end
    new_fid = remove_by_ncc_in_end(fid=fid)
    fid = new_fid
    # ========================
    # Remove repeated kd tree
    candidate_index_location = 5  # mark the INDEX of fid
    start_time = time.time()
    img_draw_temp = img_draw.copy()

    dist_thr = diameter_int  # *1.414
    node = kd.Node()
    new_fid = []
    kd.construct(d=2, data=fid.copy(), node=node, layer=0)
    for i in range(len(fid)):
        if fid[i][4] < 0:
            continue
        L = []  # To save neighborhood
        kd.search(node=node, p=fid[i], L=L, K=5)
        for j in range(len(L)):
            if kd.distance(fid[i], L[j]) < dist_thr:
                fid[int(L[j][candidate_index_location])][4] = -1
        new_fid.append(fid[i])
        cv2.circle(img_draw_temp, [fid[i][0] + radius_int, fid[i][1] + radius_int], radius_int, (0, 255, 0), 2)
        kd.clear_flag(node=node)
    fid = new_fid
    if show_img:
        fun.draw(img_draw_temp, "remove repeat")
    if save_img:
        if img_draw_temp.dtype == 'float32':
            img_draw_temp = (255 * img_draw_temp).astype(np.uint8)
            cv2.imwrite(f'{result_folder}/candidate3_repeat_{save_name}', img_draw_temp)
        else:
            cv2.imwrite(f'{result_folder}/candidate3_repeat_{save_name}', img_draw_temp)
    end_time = time.time()

    fid = np.array(fid)
    return fid


def location_fid(img: np.ndarray, location: np.ndarray, width: int):
    """
    location the fiducial markers
    :param img: original image 
    :param location: The coordinates of the upper left corner
    :param width: The size of the subgraph for circle center refinement
    :return:refined_xy, coordinate after refinement
    """
    location = location.astype(int)
    img_inv = fun.Img_in(img)
    refined_xy = []
    score_xy = []
    for i in range(len(location)):
        sub_img = img_inv[location[i, 1]:location[i, 1] + width, location[i, 0]:location[i, 0] + width]
        if 0 in sub_img:
            sub_img[sub_img == 0] = 2

        row, colum, sigma, para_a = gs.compute_center_Gauss(sub_img)
        real_row = row + location[i, 1]
        real_colum = colum + location[i, 0]
        refined_xy.append(np.array([real_colum, real_row], dtype=int))
        score = gs.compute_gauss_error(sub_img, row, colum, sigma, para_a)
        score_xy.append(score)
    score_xy = np.array(score_xy)
    refined_xy = np.array(refined_xy)
    return refined_xy, score_xy


def main(root_dir, projection, agle, dense, scale, result_folder):
    global wave_img, radius_int, diameter_int, show_img, save_img, show_plt, fids_file, information_file
    mean = np.mean(projection)
    std = np.std(projection)
    projection = projection.copy()
    projection[projection>mean+4*std] = mean + 4*std
    projection[projection<mean-4*std] = mean - 4*std

    ori_img = cv2.normalize(projection, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    show_img = 0
    show_plt = 0

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    python_file_name = os.path.basename(__file__)

    # information_file used to save information
    information_file = open(f"{result_folder}/general_information.txt", "a")
    information_file.write(f"Time：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")
    information_file.write(f"The program file is {python_file_name}.\n")
    information_file.write(f"The input mrc_file is {file_name} ... \n")
    information_file.write(f"The selected angle is {agle}.\n")
    information_file.write(f"The scale={scale}; dense={dense}\n")
    print(f'The input mrc_file is {file_name} ...')
    fids_file = open(f"{result_folder}/fiducial_{file_name.split('.')[0]}.txt", "w")

    if ori_img is None:
        print(f'There is no figure {file_name}!')
        sys.exit(0)
    if show_img:
        fun.draw(ori_img, "The input image")
    information_file.write(f"The shape of the projection is {ori_img.shape}.\n")

    if dense:
        size = 2500
    else:
        size = 2000
    # size = 2000
    m, n = ori_img.shape
    min_shape = min(m, n)
    # check if it is empty and too big to detect
    resize_index = 0
    if min_shape > size:
        for i in range(2, 10, 2):
            if int(min_shape / i) > size:
                continue
            else:
                mul_para = i  # parameter of resize
                resize_index = 1
                break
        img_resized = cv2.resize(ori_img, dsize=(int(n / mul_para), int(m / mul_para)),
                                 interpolation=cv2.INTER_AREA)
    else:
        img_resized = ori_img
        mul_para = 1

    if show_img:
        fun.draw(img_resized, 'After resize')

    # detection start
    print("The detection step begins...")
    start = time.time()
    img = cv2.GaussianBlur(img_resized, (5, 5), 0)
    img1 = fun.ToOne(img)

    # Template generation
    start_template = time.time()
    template1, wave_img, diameter_int = template_make(img, scale)
    if not diameter_int:
        return 0
    end_template = time.time()
    information_file.write(f"The time of the first module is {end_template-start_template}.\n")
    try:
        m, n = template1.shape
    except AttributeError:
        print(f'Now we will passed it.')
        return 0
    template1 = template_average(template1)
    ori_template = cv2.resize(template1, dsize=(int(mul_para * m), int(mul_para * n)),
                              interpolation=cv2.INTER_AREA)
    if show_img:
        fun.draw(wave_img, "output wave_image of template_make")
        fun.draw(template1, "template made from template_make")
    if save_img:
        if ori_template.dtype == 'float32':
            template_temp = (255 * ori_template).astype(np.uint8)
            cv2.imwrite(f'{result_folder}/template_{save_name}', template_temp)
        else:
            cv2.imwrite(f'{result_folder}/template_{save_name}', ori_template)

    # diameter_int = diameter_temp * mul_para
    radius_int = int(diameter_int / 2 + 0.5)

    # statistic of NCC pixel and contrast to get the threshold
    fid = markerauto_work_flow(img, template1)
    end = time.time()
    information_file.write(f"The number of detected fiducial markers in detection step is {len(fid)}.\n")
    information_file.write(f"The time of detection step is {end-start}\n")
    print(f"The time of detection step is {end-start}")
    # reture the location of original image
    ori_fid = fid * mul_para

    # location the fids
    print("The fiducial marker localization step begins...")
    start = time.time()
    temp_ori_m = ori_template.shape[0]
    fid, _ = location_fid(ori_img, ori_fid, temp_ori_m)
    end = time.time()
    print(f"The time of fiducial markers localization step is {end-start}")
    information_file.write(f"The time of fiducial markers localization step is {end-start}\n")
    # ori_img_draw = cv2.equalizeHist(ori_img)
    ori_img_draw = cv2.cvtColor(ori_img,cv2.COLOR_GRAY2BGR)
    for i in range(len(fid)):
        fids_file.write(f"{fid[i]}\n")
        cv2.circle(ori_img_draw, fid[i], int(radius_int*mul_para), (0, 255, 0), 2)

    cv2.imwrite(f"./{result_folder}/end_{save_name}", ori_img_draw)
    fids_file.close()
    information_file.close()
    return 1


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Information Input")
    parser.add_argument("mrc_dir", type=str, metavar="MRC file path")
    parser.add_argument("fixed_projection", type=int, metavar="Index of projection to detect" )
    parser.add_argument("--dense", type=int, default=0, metavar="Mark whether the number of fiducial markers is dense enough (at least greater than 50)")
    parser.add_argument("--scale", type=int, default=2, metavar="The scale of wavelet transform")
    parser.add_argument("--threshold_ncc", type=float, default=0.55, metavar="The threshold of the Template Matching at the end of MarkerDetector")
    parser.add_argument("--save_all_figure", type=int, default=0, metavar="Mark if save all figure")
    args = parser.parse_args()
    root_dir = args.mrc_dir

    save_img = args.all_figure

    if args.fixed_projection == -1:
        detect_one_projection = 0
    else:
        detect_one_projection = 1
        detect_which_projection = args.fixed_projection
    dense = args.dense
    scale = args.scale
    hyperparameter_ncc = args.threshold_ncc

    file_data = mf.open(root_dir).data
    num_projection = file_data.shape[0]
for angle in range(num_projection):
    if detect_one_projection:
        if angle != detect_which_projection:
            continue
    projection = file_data[angle].copy()
    result_folder_root = f"./Result"
    if not os.path.exists(result_folder_root):
        os.mkdir(result_folder_root)
    result_folder = f"{result_folder_root}/result_{angle}"
    file_name = root_dir.split('/')[-1]
    save_name = file_name.split('.')[0]+'.jpg'
    key = main(root_dir=root_dir,projection=projection, agle=angle, dense=dense, scale=scale, result_folder = result_folder)
    if not key:
        print('High-quality templates suitable for this dataset are difficult to extract. ')
