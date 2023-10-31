import numpy as np
import cv2 as cv
cap = cv.VideoCapture(cv.samples.findFile('color_vid.avi'))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    # flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    prev = prvs
    flow=None
    pyr_scale = 0.5
    levels=3
    winsize=15
    iterations=3
    poly_n=5
    poly_sigma=1.2
    flags=0
    flow = cv.calcOpticalFlowFarneback(	prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next

cv.destroyAllWindows()