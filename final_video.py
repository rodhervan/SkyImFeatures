import cv2
import numpy as np
import glob
import os




img_array = []

folder_path = 'D:\\RODRIGO\\Tesis IMEC\\Python\\20230807_seg_avg'
if os.path.exists(folder_path):
    direction = glob.glob(f'{folder_path}/*.png')
else:
    print(f"Folder '{folder_path}' does not exist.")

for filename in direction:
    img = cv2.imread(filename)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)  # White color in BGR
    font_thickness = 2
    # cv2.putText(img, os.path.basename(filename), bottom_left_corner, font, font_scale, font_color, font_thickness)

    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('20230807_avg.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()