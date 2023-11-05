'''
Author: Hou Gaoxin 710872687@qq.com
Date: 2023-10-02 21:18:37
LastEditors: Hou Gaoxin 710872687@qq.com
LastEditTime: 2023-11-05 15:40:59
FilePath: /Program_detection_program/mv_jpg2file_interval_2.py
Description:
'''
import os

root_dir = '/home/haley/dataset/MarkerDetector2Pixel/10016'
num_projection = 121

save_dir = f"{root_dir}/detect_result_interval2"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


interval = 3
for i in range(num_projection):
    if interval > 2:
        interval = 0
        file_name = f'result_{i}'
        file_i_dir = f"{root_dir}/{file_name}"
        file_list = os.listdir(file_i_dir)
        for file in file_list:
            if file[:3]=="end":
                file_dir = f"{file_i_dir}/{file}"
                os.system(f"cp {file_dir} {save_dir}/result_{i}.jpg")
                break
    interval += 1
