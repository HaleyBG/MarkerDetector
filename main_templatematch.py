'''
该算法是用来与主算法进行对比而写，该算法是模板匹配方法。
'''
import numpy as np

from fundmental import *
from templateMatch import *
from hicluster import *
import datetime
import os
import baseFun as bs


def catch_template(img, inf):
    if show_img:
        draw(img, 'the original picture')
    x = inf[0]
    y = inf[1]
    r = inf[2]
    template = img[x - r:x + r + 1, y - r:y + r + 1]
    if show_img:
        draw(template, 'the picked template')
    return template


def TemplateMatch(img, template, method, threshold, f, draw_img):
    m = template.shape[0]
    r = int((m - 1) / 2)
    result = cv2.matchTemplate(img, template, method)
    # mean = np.mean(result)
    # std = np.std(result)
    # threshold = mean+2.5*std
    result2 = hardval2(result, threshold)
    hi_result = hicluster(result2, r)
    # filename = input('Please give a name used to save the locate information')
    # f = open('./templateMatchResult/locate%s.txt'%(filename), 'w')
    f.write(f'The total number of particle is {len(hi_result.keys())}\n')
    print('the total num of particle is', len(hi_result.keys()))
    img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    for pt in hi_result:
        cv2.circle(img, (hi_result[pt][0][1] + r, hi_result[pt][0][0] + r), r, (0, 0, 255))
    if show_img:
        draw(img, 'the result of match template')
    if save_img:
        cv2.imwrite(f'./experiment2/temp_{filename}', img)
    # f.close()
    return img


if __name__ == '__main__':
    global show_img
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
    info = [[384, 189, 6],
            [552, 505, 15],
            [508, 725, 6],
            [350, 724, 13],
            [510, 769, 13],
            [491, 600, 10],
            [389, 475, 4],
            [385, 615, 5],
            [1470,1180,10]]  # [x,y1,r]

    # threshold_ncc_list = [0.9,
    #                  0.7,
    #                  0.8,
    #                  0.7,
    #                  0.7,
    #                  0.7,
    #                  0.9,
    #                  0.9,
    #                  0.8]

    threshold_ncc_list = np.ones((len(info)))*0.8

    bs.mkdir('./experiment2')
    f = open('./experiment2/time.txt', 'w')
    f.write('=========================================\n')
    python_file_name = os.path.basename(__file__)
    f.write(f"The program file is {python_file_name}.\n")
    f.write('start the process of template matching\n')
    for i in range(len(file_set)):
        threshold_ncc = threshold_ncc_list[i]
        start = datetime.datetime.now()
        filename = file_set[i]
        inf = info[i]
        print(f'This is {filename} turn')
        f.write("=================================")
        f.write(f"The input image is {filename}.\n")
        img_ori = cv2.imread(f'./mrc/{filename}', 0)
        if img_ori is None:
            print(f'There is no figure {filename}!')
            break
        img = cv2.GaussianBlur(img_ori, (5, 5), 0)
        template = catch_template(img, inf)
        res_img = TemplateMatch(img, template, 5, threshold_ncc, f, img_ori)
        end = datetime.datetime.now()
        print('the total time is ', end - start)
        # cv2.imwrite(f'./experiment2/res{filename}', res_img)
        f.write(f'the time of {filename} is {end - start}\n')
    f.write('end the process of template matching \n')
    f.close()
