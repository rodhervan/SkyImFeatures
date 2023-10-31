import cv2
import numpy as np



# # Load the image
# image = cv2.imread('20230803000000.jp2', cv2.IMREAD_GRAYSCALE)
# # image = cv2.imread('adjusted.jp2', cv2.IMREAD_GRAYSCALE)

# # Find the minimum and maximum pixel values
# min_val = np.min(image)
# max_val = np.max(image)

# # Stretch the contrast
# stretched_image = cv2.convertScaleAbs(image, alpha=255/(max_val-min_val), beta=-255*min_val/(max_val-min_val))

# im_color = cv2.applyColorMap(stretched_image, cv2.COLORMAP_JET) 

# # Display or save the result
# cv2.imshow('Stretched Image', stretched_image)
# # cv2.imshow('Stretched Image', im_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Load the image
# image = cv2.imread('20230803000000.jp2', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('pshop_edited/20230822221100.jpf', cv2.IMREAD_GRAYSCALE)

color_on_def = cv2.applyColorMap(image, cv2.COLORMAP_JET)

# Find the minimum and maximum pixel values
min_val = np.min(image)
max_val = np.max(image)

# Stretch the contrast
stretched_image = cv2.convertScaleAbs(image, alpha=255/(max_val-min_val), beta=-255*min_val/(max_val-min_val))

# cmap

# Load the 360-degree image
photoshop = cv2.imread('adjusted.jp2', cv2.IMREAD_GRAYSCALE)
src = cv2.applyColorMap(photoshop, cv2.COLORMAP_JET)
color = src 
scale_percent = 100 # percent of original size
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)
dim = (width, height)
center = (width/2, height/2)

# cv2.warpPolar(src, dsize, center, maxRadius,flags[,dst])

radius = 180
rescale = cv2.warpPolar(src, (400,600), (316,237), radius, cv2.WARP_FILL_OUTLIERS )
rotate = cv2.rotate(rescale, cv2.ROTATE_90_COUNTERCLOCKWISE)
next_resc = cv2.warpPolar(rescale, (radius*2,radius*2), (radius,radius), radius, cv2.WARP_INVERSE_MAP )

cv2.imshow("entrada", src)

cv2.imshow("estirada polar", rotate)

cv2.imshow("recovery", next_resc)





# cv2.imwrite('color_on_default.jpg', color_on_def)
# cv2.imwrite('photoshop.jpg', photoshop)
# cv2.imwrite('balance.jpg', stretched_image)
# cv2.imwrite('azimuth.jpg', rotate)
# cv2.imwrite('color.jpg', color)
cv2.waitKey(0)
cv2.destroyAllWindows()

