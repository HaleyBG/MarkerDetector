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
from mrc2jpg import *
import argparse



def catch_template(img, inf):
    if show_img:
        img255 = (img*255).astype(np.uint8)
        draw(img255, 'the original picture')
    x = inf[0]
    y = inf[1]
    r = inf[2]
    template = img[x - r:x + r + 1, y - r:y + r + 1]
    if show_img:
        template255 = (template*255).astype(np.uint8)
        draw(template255, 'the picked template')
    return template


def TemplateMatch(img, template, method, threshold, f, draw_img):
    global result_dir_root
    m = template.shape[0]
    r = int((m - 1) / 2)
    img255 = (img*255).astype(np.uint8)
    template255 = (template*255).astype(np.uint8)
    result = cv2.matchTemplate(img255, template255, method)
    # mean = np.mean(result)
    # std = np.std(result)
    # threshold = mean+2.5*std
    result2 = hardval2(result, threshold)
    if not np.any(result2):
        return [0]
    hi_result = hicluster(result2, r)
    # filename = input('Please give a name used to save the locate information')
    # f = open('./templateMatchResult/locate%s.txt'%(filename), 'w')
    f.write(f'The total number of particle is {len(hi_result.keys())}\n')
    print('the total num of particle is', len(hi_result.keys()))
    draw_img255 = (draw_img*255).astype(np.uint8)
    img = cv2.cvtColor(draw_img255, cv2.COLOR_GRAY2BGR)
    for pt in hi_result:
        cv2.circle(img, (hi_result[pt][0][1] + r, hi_result[pt][0][0] + r), r, (0, 255, 0), 2)
    if show_img:
        draw(img, 'the result of match template')
    if save_img:
        cv2.imwrite(f"{result_dir_root}/temp_{filename.split('.')[0]}.jpg", img)
    # f.close()
    return img
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TEMPLATE MATCHING")
    parser.add_argument("mrc_dir", type=str, metavar="MRC file path")
    parser.add_argument("fixed_projection", type=int, metavar="Index of projection to detect" )
    parser.add_argument("template_info", type=list, metavar="List information of Template in mrc file (x, y, r)")
    parser.add_argument("--threshold", type=float, default=0.75, metavar="Threshold of template matching")
    args = parser.parse_args()

    show_img = 0
    save_img = 1

    result_dir_root = "./TemplateResult"
    if not os.path.exists(result_dir_root):
        os.mkdir(result_dir_root)

    f = open(f'{result_dir_root}/time.txt', 'w')
    f.write('=========================================\n')
    python_file_name = os.path.basename(__file__)
    f.write(f"The program file is {python_file_name}.\n")
    f.write('start the process of template matching\n')

    threshold_ncc = args.threshold
    file_path = args.mrc_dir
    agle = args.index_projection
    inf = args.template_info

    img_ori = get_mrc_file_in_fixed_angle_and_save(file_path, angle=agle)
    start = datetime.datetime.now()
    img_ori = img_ori/255
    filename = os.path.basename(file_path)

    print(f'This is {filename} turn')
    f.write("=================================")
    f.write(f"The input image is {filename}.\n")

    img = cv2.GaussianBlur(img_ori, (5, 5), 0)
    template = catch_template(img, inf)
    template = template_average(template)
    template255 = (template*255).astype(np.uint8)
    template_dir = './TemplateMatching_template'
    if not os.path.exists(template_dir):
        os.mkdir(template_dir)
    cv2.imwrite(f"{template_dir}/{filename.split('.')[0]}.jpg", template255)
    res_img = TemplateMatch(img, template, 5, threshold_ncc, f, img_ori)
    if not np.any(res_img):
        print("No point is detected")
    end = datetime.datetime.now()
    print('the total time is ', end - start)
    f.write(f'the time of {filename} is {end - start}\n')


    f.write('end the process of template matching \n')
    f.close()