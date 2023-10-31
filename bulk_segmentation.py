import os, os.path
import cv2 as cv
import cv2 
import numpy as np
# import the necessary packages
import argparse
import imutils
from matplotlib import pyplot as plt

def watershed_seg (img):
    # img = cv.imread('20230807_color - Copy/20230807143000.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),200,100)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    
    
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    
    markers = cv.watershed(img,markers)
    img[markers == -1] = [0,0,0]
    return img

    
# Create the 'modified' directory if it doesn't exist
output_directory = '20230807_segmented_color'
os.makedirs(output_directory, exist_ok=True)
 
# iterate over files in
# that directory
directory = '20230807_color'
# directory = 'nf'
for filename in os.scandir(directory):
    if filename.is_file():
        # print(filename.path)
        # name = str(filename).replace('<DirEntry ','').replace('>','').replace("'","").replace('jpf','jpg')
        name = str(filename).replace('<DirEntry ','').replace('>','').replace("'","")
        # new_path = 'modified/' + name
        # print(new_path)
        # image = cv2.imread(filename.path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.imread(filename.path)
        
        img = watershed_seg (image)
        # stretched_image = cv2.convertScaleAbs(image, alpha=255/(max_val-min_val), beta=-255*min_val/(max_val-min_val))
        
        # write modified image
        
        # Construct the new path in the 'modified' directory
        new_path = os.path.join(output_directory, name)
        
        cv2.imwrite(new_path, img)


