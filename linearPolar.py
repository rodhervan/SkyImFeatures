import cv2
import numpy as np
 


image_path = 'adjusted.jp2'

source = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 



#--- ensure image is of the type float ---
img = source.astype(np.float32)
 
#--- the following holds the square root of the sum of squares of the image dimensions ---
#--- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
 
polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
 
polar_image = polar_image.astype(np.uint8)
cv2.imshow("Polar Image", polar_image)

cv2.waitKey(0)
cv2.destroyAllWindows()