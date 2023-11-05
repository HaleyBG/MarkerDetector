'''
Author: Hou Gaoxin 710872687@qq.com
Date: 2023-05-13 14:30:13
LastEditors: Hou Gaoxin 710872687@qq.com
LastEditTime: 2023-11-05 16:36:01
FilePath: /Program_detection_program/main_waveletsegmentation.py
Description: 
'''


from fundmental import *
from wavelet import *
from watershed_py import *
import datetime
import baseFun as bs
import mrcfile as mf
import os
from mrc2jpg import *
import argparse


def step1(img, J=3, f=None):
    """
    Wavelet Segmentation Method
    :param img:
    :param J:
    :return:
    """
    global result_dir_root
    wave, __ = waveletprocess2(img, J)
    wavecor = np.float32(wave)
    if save_img:
        cv2.imwrite(f"{result_dir_root}/wave_{filename}.jpg", wavecor)
    para_var = np.var(img)
    # threshold
    ret = para_var * 0.5
    thresh_wave = fun.hardval2(wavecor, ret)
    cv2.imwrite(f'./main8_real_paper/wave_{filename}.jpg', thresh_wave.astype(np.uint8))
    if show_img:
        fun.draw(thresh_wave, '2 value')
    removeSmall(thresh_wave, 4)
    result = watershed_py(img, thresh_wave, f)
    return result


if __name__ == '__main__':
    global show_img, save_img
    show_img = 0
    save_img = 1
    parser = argparse.ArgumentParser(description="WAVELET SEGMENTATION")
    parser.add_argument("mrc_dir", type=str, metavar="MRC file path")
    parser.add_argument("fixed_projection", type=int, metavar="Index of projection to detect" )
    args = parser.parse_args()

    result_dir_root = "./SegmentationResult"
    if not os.path.exists(result_dir_root):
        os.mkdir(result_dir_root)
    f = open(f'{result_dir_root}/time.txt', 'a')
    f.write(f'The wavelet segmentation will be executed. \n')

    file_dir = args.mrc_dir
    agle = args.index_projection
    filename_temp = os.path.basename(file_dir)
    filename = filename_temp.split(".")[0]
    print("==================================")
    print(f'The input image is {filename}.')
    f.write("================================\n")
    f.write(f"The input image is {filename}.\n")
    j = 4
    threshold_para_wave = 2
    step1_remove_small_para = 50 
    projection = get_mrc_file_in_fixed_angle_and_save(file_dir,agle)
    start = datetime.datetime.now()
    img = ToOne(projection).astype(np.float32)
    res = step1(img, j, f)
    end = datetime.datetime.now()
    total_time = end - start
    print('the total time is ', total_time)
    if save_img:
        cv2.imwrite(f'{result_dir_root}/{filename}.jpg', res)
    f.write(f'the time of {filename} is {total_time}\n')


    f.write(f'wavelet process is over\n')
    f.close()