'''
Author: Hou Gaoxin 710872687@qq.com
Date: 2023-09-30 13:32:50
LastEditors: Hou Gaoxin 710872687@qq.com
LastEditTime: 2023-10-16 16:57:43
FilePath: /Program_detection_program/tool/mv_jpg2file.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os

# dir_root = '/home/haley/dataset/MarkerDetectorResult7/data_09dec31a_024_result'
# dir_root = '/home/haley/dataset/MarkerDetectorResult7/data_09dec31a_026_result'
# dir_root = '/home/haley/dataset/MarkerDetectorOnlyOneProjection/11_0007'
dir_root = '/home/haley/dataset/MarkerDetector2Pixel/cryo_hypencc_070'
dir_detect_result = f"{dir_root}/detect_result"
if not os.path.exists(dir_detect_result):
    os.mkdir(dir_detect_result)
file_list = os.listdir(dir_root)
for file in file_list:
    if file == 'README.md':
        continue
    dir_result = f"{dir_root}/{file}"
    jpg_list = os.listdir(dir_result)
    for jpg in jpg_list:
        if jpg[:3]=='end':
            dir_jpg = f"{dir_result}/{jpg}"
            dir_2_jpg = f"{dir_detect_result}/{file}.jpg"
            os.system(f"cp {dir_jpg} {dir_2_jpg}")
