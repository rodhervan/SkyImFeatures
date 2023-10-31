import os, os.path
import cv2
import numpy as np
# import the necessary packages
import argparse
import imutils

        
# Create the 'modified' directory if it doesn't exist
output_directory = 'Angulos_proc'
os.makedirs(output_directory, exist_ok=True)
 
# iterate over files in
# that directory
directory = 'Angulos'
# directory = 'nf'
for filename in os.scandir(directory):
    if filename.is_file():
        # print(filename.path)
        # name = str(filename).replace('<DirEntry ','').replace('>','').replace("'","").replace('jpf','jpg')
        name = str(filename).replace('<DirEntry ','').replace('>','').replace("'","")
        # new_path = 'modified/' + name
        # print(new_path)
        # image = cv2.imread(filename.path, cv2.IMREAD_GRAYSCALE)
        
        # image = cv2.imread(filename.path)
        
        # color_mapped = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    
        
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", type=str, default=name,
        	help="path to the input image")
        args = vars(ap.parse_args())


        # load the image and show it
        # image = cv2.imread(args["image"])
        image = cv2.imread(filename.path)
        cv2.imshow("Original", image)
        # grab the dimensions of the image and calculate the center of the
        # image
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), -3.5, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        cv2.imshow("Rotated by 45 Degrees", rotated)
        # rotate our image by -90 degrees around the image
        # M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
        # rotated = cv2.warpAffine(image, M, (w, h))
        # cv2.imshow("Rotated by -90 Degrees", rotated)
        
        # flipVertical = cv2.flip(rotated, 0)
        new_path = os.path.join(output_directory, name)
        cv2.imwrite(new_path, rotated) 
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        # # Find the minimum and maximum pixel values
        # min_val = np.min(image)
        # max_val = np.max(image)
        
        # # Stretch the contrast
        # stretched_image = cv2.convertScaleAbs(image, alpha=255/(max_val-min_val), beta=-255*min_val/(max_val-min_val))
        
        # write modified image
        
        # Construct the new path in the 'modified' directory
        # new_path = os.path.join(output_directory, name)
        
        # cv2.imwrite(new_path, flipVertical)