# import the cv2 library
# import cv2 as cv
# import numpy as np
 
# The function cv2.imread() is used to read an image.
# img_grayscale = cv.imread('20230131171730.jp2')

# im_color = cv.applyColorMap(img_grayscale, cv.COLORMAP_HOT) 

# # The function cv2.imshow() is used to display an image in a window.
# cv.imshow('graycsale image',img_grayscale)
 
# # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
# cv.waitKey(0)

# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('adjusted.jp2', cv2.IMREAD_GRAYSCALE)

# # Find the minimum and maximum pixel values
# min_val = np.min(image)
# max_val = np.max(image)

# # Stretch the contrast
# stretched_image = cv2.convertScaleAbs(image, alpha=255/(max_val-min_val), beta=-255*min_val/(max_val-min_val))

# im_color = cv2.applyColorMap(stretched_image, cv2.COLORMAP_JET) 

# # Display or save the result
# cv2.imshow('Stretched Image', stretched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2 as cv
# image_path = '20230131171730.jp2'
# src = cv.imread(image_path, cv.IMREAD_GRAYSCALE) 
# # src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# dst = cv.equalizeHist(src)

# # create a CLAHE object (Arguments are optional).
# clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(20,8))
# cl1 = clahe.apply(src)


# # cv.imshow('Source image', src)

# cv.imshow('Equalized Image', cl1)
# cv.waitKey(0)
# cv.destroyAllWindows()



# import cv2 as cv
# import numpy as np
# # Load the image
# image_path = '20230131171730.jp2'
# image = cv.imread(image_path, cv.IMREAD_UNCHANGED)  # Load image without altering color space

# # Check if image has multiple channels/bands
# if len(image.shape) == 3:  # Multi-channel image
#     # Split the image into its color channels
#     channels = cv.split(image)

#     # Display each channel separately
#     for i, channel in enumerate(channels):
#         cv.imshow(f'Channel {i}', channel)
    
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# elif len(image.shape) == 2:  # Single-channel (grayscale) image
#     # Display the single band as a grayscale image
#     maxmin = cv.minMaxLoc(image)
    
#     cv.imshow('Grayscale Band', image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()



import cv2
import numpy as np

# Load the image
image = cv2.imread('20230803000000.jp2', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('adjusted.jp2', cv2.IMREAD_GRAYSCALE)

# Find the minimum and maximum pixel values
min_val = np.min(image)
max_val = np.max(image)

# Stretch the contrast
stretched_image = cv2.convertScaleAbs(image, alpha=255/(max_val-min_val), beta=-255*min_val/(max_val-min_val))

im_color = cv2.applyColorMap(stretched_image, cv2.COLORMAP_JET) 

# Display or save the result
cv2.imshow('Stretched Image', stretched_image)
# cv2.imshow('Stretched Image', im_color)
cv2.waitKey(0)
cv2.destroyAllWindows()