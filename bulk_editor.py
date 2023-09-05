import os, os.path
import cv2
import numpy as np

        
# Create the 'modified' directory if it doesn't exist
output_directory = 'color_map'
os.makedirs(output_directory, exist_ok=True)
 
# iterate over files in
# that directory
directory = 'pshop_edited'
# directory = 'nf'
for filename in os.scandir(directory):
    if filename.is_file():
        # print(filename.path)
        name = str(filename).replace('<DirEntry ','').replace('>','').replace("'","").replace('jpf','jpg')
        # new_path = 'modified/' + name
        # print(new_path)
        
        image = cv2.imread(filename.path, cv2.IMREAD_GRAYSCALE)
        
        src = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        
        # # Find the minimum and maximum pixel values
        # min_val = np.min(image)
        # max_val = np.max(image)
        
        # # Stretch the contrast
        # stretched_image = cv2.convertScaleAbs(image, alpha=255/(max_val-min_val), beta=-255*min_val/(max_val-min_val))
        
        # write modified image
        
        # Construct the new path in the 'modified' directory
        new_path = os.path.join(output_directory, name)
        
        cv2.imwrite(new_path, src)