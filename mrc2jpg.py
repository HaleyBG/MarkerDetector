'''
@Project ：Program_detection 
@File    ：read_mrc_file.py
@IDE     ：PyCharm 
@Author  ：Haley
@Date    ：2023/4/20 下午10:20 
@Usage   : Used to read a projection in mrcfile
'''
import glob
import os
import mrcfile as mf
import cv2
import numpy as np
import fundmental as fun


def get_filename_with_mrc_end(file_dir: str):
    """
    Get the filename in file_dir with particular end name
    :param file_dir:e.g.:/data/ETdata
    :return:
    """
    return glob.glob(f"{file_dir}/*_e2.mrc")


def read_all_file_in_filepath(file_path: str):
    """
    get the mrc in this mrc_file and get in the sub_file in this mrc_file
    :param file_path: e.g.:/data/ETdata
    :return:
    """
    mrc_list = get_filename_with_mrc_end(file_path)
    file_path_list = os.listdir(file_path)
    for sub_file in file_path_list:
        if os.path.isdir(f"{file_path}/{sub_file}"):
            sub_mrc_list = read_all_file_in_filepath(f"{file_path}/{sub_file}")
            mrc_list.extend(sub_mrc_list)
    return mrc_list


def get_mrc_file_in_some_angle(file_path_name: str, number: int):
    """
    get picture in special angle
    :param file_path_name: e.g.:/data/ETdata/*_e2.mrc
    :param number:
    :return:
    """
    file = mf.open(file_path_name)
    file_data = file.data
    return file_data[number]


def get_mrc_file_in_fixed_angle_and_save(file_path_name: str, angle: int = -1,
                                         equalizeHist: bool = False):
    """

    :param file_path_name:
    :param interval: interval of angles
    :param angle: if fixed angles?
    :param equalizeHist: if equalized?
    :return:
    """
    file = mf.open(file_path_name)
    file_data = file.data
    file_max = np.max(file_data)
    file_min = np.min(file_data)
    file_data = (((file_data - file_min)/file_max)*255).astype(np.uint8)
    temp = file_path_name.split('/')[-1]
    file_name = temp.split('.')[0]
    # set the fixed angle
    if angle == -1:
        total_number = file_data.shape[0]
        if total_number % 2:
            high = int((total_number - 1) / 2)
            mid = int(high / 2)
        else:
            high = int((total_number) / 2 - 1)
            mid = int(high / 2)
        # 保存比mid大的
        i = mid
        while i < total_number:
            projection = file_data[i]
            # projection = cv2.normalize(projection, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if equalizeHist:
                projection = cv2.equalizeHist(projection)
            # cv2.imwrite(f"./projection/{file_name}_{i}.jpg", projection)
            print(f"The {i}th projection is succeed!")
            i += 5
        i = mid
        while i >=0:
            projection = file_data[i]
            projection = cv2.normalize(projection, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if equalizeHist:
                projection = cv2.equalizeHist(projection)
            # cv2.imwrite(f"./projection/{file_name}_{i}.jpg", projection)
            print(f"The {i}th projection is succeed!")
            i -= 5

    elif angle == -2:
        total_number = file_data.shape[0]
        if total_number % 2:
            mid = int(total_number / 2)
        else:
            mid = int((total_number - 1) / 2)
        projection = file_data[mid]
        projection = cv2.normalize(projection, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if equalizeHist:
            projection = cv2.equalizeHist(projection)
        # cv2.imwrite(f"./projection/{file_name}_high2.jpg", projection)

    else:
        projection = file_data[angle]
        pro_mean = np.mean(projection)
        pro_std = np.std(projection)
        projection = projection.copy()
        projection[projection>pro_mean+4*pro_std] = pro_mean + 4*pro_std
        projection[projection<pro_mean-4*pro_std] = pro_mean - 4*pro_std
        projection = cv2.normalize(projection, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if equalizeHist:
            projection = cv2.equalizeHist(projection)
        # fun.draw(projection)
        cv2.imwrite(f"./projection/{file_name}_{angle}.jpg", projection)
        return projection


if __name__ == "__main__":

    root_dir_list = [
        '/media/haley/Expansion/ETdata/70001/70001.st',
    ]
    agle_list = [
    55,
        ]

    if not os.path.exists('./projection'):
        os.mkdir('./projection')
    for i in range(len(agle_list)):
        root_dir = root_dir_list[i]
        agle = agle_list[i]
        print(f"=============================")
        print(f"{root_dir}")
        get_mrc_file_in_fixed_angle_and_save(root_dir, angle=agle)
