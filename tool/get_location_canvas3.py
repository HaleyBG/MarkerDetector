'''
Author: Hou Gaoxin 710872687@qq.com
Date: 2023-09-26 16:16:07
LastEditors: Hou Gaoxin 710872687@qq.com
LastEditTime: 2023-09-27 16:57:38
FilePath: /Program_detection_program/tool/get_location_canvas3.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tkinter import *
from PIL import Image, ImageTk
import os

def main():
    global resize_num
    # 创建 Tkinter 窗口
    root = Tk()
    root.title("Scrollable Image Viewer with Coordinate Saving")

    # 创建 Canvas 以显示图像
    canvas = Canvas(root)
    canvas.pack(side="top", fill="both", expand=True)

    # 创建滚动条
    scrollbar = Scrollbar(root, command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.config(yscrollcommand=scrollbar.set)

    # 打开图像文件并缩小尺寸
    image = Image.open(file_dir)
    width = image.width
    height = image.height
    new_width = int(width/resize_num)
    new_height = int(height/resize_num)
    image = image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)

    # 在 Canvas 中显示图像
    canvas.create_image(0, 0, anchor="nw", image=photo)
    canvas.config(scrollregion=canvas.bbox("all"))

    # 记录上一次鼠标位置的变量
    prev_x = None
    prev_y = None
    # 拖动图片的函数
    def drag(event):
        global prev_x, prev_y
        x, y = event.x, event.y
        if prev_x and prev_y:
            canvas.move("all", x - prev_x, y - prev_y)
        prev_x, prev_y = x, y

    # 松开鼠标按钮时重置变量，并保存点击坐标
    def release(event):
        global prev_x, prev_y
        prev_x = None
        prev_y = None

    # 保存点击坐标的函数
    def save_coordinates(event):
        global i
        x, y = event.x, event.y
        canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill="white")
        with open(f"del_coor_{i}.txt", "a") as file:
            file.write(f"{resize_num*x} {resize_num*y}\n")
        
    # 将拖动事件和鼠标释放事件绑定到 Canvas 上
    canvas.bind("<Button-1>", save_coordinates)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", release)

    # 运行 Tkinter 主循环
    root.mainloop()


if __name__=="__main__":
    # root_dir = "/home/haley/dataset/ETdataFiducialMarkerLocation/data_10004_result"
    # root_dir = "/home/haley/dataset/ETdataFiducialMarkerLocation/data_12jan05_AT1BT1_result"
    root_dir = "/home/haley/dataset/ETdataFiducialMarkerLocation/data_2017_ywc259_10019_result"
    # root_dir = ""
    num_projection = 121
    resize_num = 3
    for i in range(num_projection):
        res_dir = f"{root_dir}/result_{i}"
        file_list = os.listdir(res_dir)
        for file_name in file_list:
            if file_name[:3]=="end":
                file_dir = f"{res_dir}/{file_name}"
                file_name_end = file_name.split('.')[0]
                main()


