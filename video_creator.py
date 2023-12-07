import cv2
import numpy as np
import glob
import os

# img_array = []

# folder_path = 'D:\\RODRIGO\\Tesis IMEC\\Python\\20230807_normal'
# if os.path.exists(folder_path):
#     direction = glob.glob(f'{folder_path}/*.png')
# else:
#     print(f"Folder '{folder_path}' does not exist.")

# for filename in direction:
#     img = cv2.imread(filename)
    
#     color_mapped = cv2.applyColorMap(img, cv2.COLORMAP_JET)
#     flipVertical = cv2.flip(color_mapped, 0)
#     height, width, layers = flipVertical.shape
#     size = (width,height)
#     img_array.append(flipVertical)

# out = cv2.VideoWriter('20230807_color.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()



img_array = []

folder_path = 'D:\\RODRIGO\\Tesis IMEC\\Python\\20230807_sun_pos'
if os.path.exists(folder_path):
    direction = glob.glob(f'{folder_path}/*.png')
else:
    print(f"Folder '{folder_path}' does not exist.")

for filename in direction:
    img = cv2.imread(filename)
    
    # color_mapped = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    # flipVertical = cv2.flip(color_mapped, 0)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('20230807_general.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()