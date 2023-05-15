"""
该算法将复现论文中的算法
1.main3的两条分支的硬阈值，聚类等合成了一条分支，减少了一次聚类，希望可以提高时间.
2.小波使用atrous小波
"""

from fundmental import *
from wavelet import *
from watershed_py import *
import datetime
import baseFun as bs


def step1(img, J=3, f=None):
    """
    小波算法与分水岭算法（论文算法）
    :param img:
    :param J:
    :return:
    """
    # 小波处理——使用自己写的B3样条插值
    wave, __ = waveletprocess2(img, J)
    wavecor = np.float32(wave)
    if save_img:
        cv2.imwrite(f'./experiment1/wave_{filename}', wavecor)
    para_var = np.var(img)
    # threshold
    ret = para_var * 0.5
    thresh_wave = fun.hardval2(wavecor, ret)
    cv2.imwrite(f'./main8_real_paper/wave_{filename}', thresh_wave.astype(np.uint8))
    if show_img:
        fun.draw(thresh_wave, '2 value')
    removeSmall(thresh_wave, 4)
    result = watershed_py(img, thresh_wave, f)
    return result


if __name__ == '__main__':
    global show_img, save_img
    show_img = 0
    save_img = 1
    file_set = ['Centriole.jpg',
                'Motor.jpg',
                'Belt.jpg',
                'Vibrio1.jpg',
                'Vibrio2.jpg',
                'Nitrosop.jpg',
                'Hemocyanin1.jpg',
                'Hemocyanin2.jpg',
                'VEEV.jpg']
    bs.mkdir('./experiment1')
    f = open('./experiment1/time.txt', 'a')
    f.write(f'The wavelet segmentation will be executed. \n')
    global filename
    for filename in file_set:
        print("==================================")
        print(f'The input image is {filename}.')
        f.write("================================\n")
        f.write(f"The input image is {filename}.\n")
        start = datetime.datetime.now()
        j = 4
        threshold_para_wave = 2
        step1_remove_small_para = 50  # 可以用来去掉小波图中小区域点
        img = cv2.imread(f'./mrc/{filename}', 0)
        if img is None:
            print(f'There is no figure {filename}!')
            break
        img = ToOne(img).astype(np.float32)
        res = step1(img, j, f)
        end = datetime.datetime.now()
        total_time = end - start
        print('the total time is ', total_time)
        if save_img:
            cv2.imwrite(f'./experiment1/{filename}', res)
        f.write(f'the time of {filename} is {total_time}\n')


    f.write(f'wavelet process is over\n')
    f.close()