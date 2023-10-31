import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# img = cv.imread('20230807_normal/20230807114600.png')
# img = cv.imread('20230807_color/20230807112500.png')
img = cv.imread('20230807_color/20230807152030.png')


radius = 180
rescale = cv.warpPolar(img, (400,600), (316,237), radius, cv.WARP_FILL_OUTLIERS )
rotate = cv.rotate(rescale, cv.ROTATE_90_COUNTERCLOCKWISE)

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
    return thresh

seg_default = watershed_seg (img)
seg_warped = watershed_seg (rotate)


back_to_90 = cv.rotate(seg_warped, cv.ROTATE_90_CLOCKWISE)
next_resc = cv.warpPolar(back_to_90, (radius*2,radius*2), (radius,radius), radius, cv.WARP_INVERSE_MAP )
mask = np.zeros(next_resc.shape[:2], dtype="uint8")
cv.circle(mask, (180, 180), 180, 255, -1)
masked = cv.bitwise_and(next_resc, next_resc, mask=mask)


cv.imshow("Default", seg_default)
cv.imshow("Warped", masked)



cv.waitKey(0)
cv.destroyAllWindows()