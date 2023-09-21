import numpy as np
import cv2
import time
import pandas as pd



def draw_flow(img, flow, step=10):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):

    # h, w = flow.shape[:2]
    # fx, fy = flow[:,:,0], flow[:,:,1]

    # ang = np.arctan2(fy, fx) + np.pi
    # v = np.sqrt(fx*fx+fy*fy)

    # hsv = np.zeros((h, w, 3), np.uint8)
    # hsv[...,0] = ang*(180/np.pi/2)
    # hsv[...,1] = 255
    # hsv[...,2] = np.minimum(v*4, 255)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    hsv = np.zeros_like(img)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr, mag, ang



cap = cv2.VideoCapture('color_vid.avi')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Initialize the DataFrame with an empty MultiIndex
df = pd.DataFrame(columns=['fx', 'fy'])
frame = 0
f_list = []
dataframe_list = []
abc_dataframe = []
while True:
# while frame < 30:

    suc, img = cap.read()
    if not suc:
        print('No frames grabbed!')
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray


    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)
    print(f"{fps:.2f} FPS")
    
    draw_flow_img = draw_flow(gray, flow)
    draw_hsv_img, mag, ang = draw_hsv(flow)
    deg_ang = np.rad2deg(ang)
    nested_list = [mag,deg_ang]
    abc = np.array(nested_list)
    abc = np.transpose(abc, (1, 2, 0))
    all4 = np.dstack((flow, abc))
    
    cv2.imshow('flow', draw_flow_img)
    cv2.imshow('flow HSV', draw_hsv_img)
    
    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('flow.png', draw_flow_img)
        cv2.imwrite('hsv.png', draw_hsv_img)
    # # Create a MultiIndex representing the grid coordinates
    index = pd.MultiIndex.from_product([range(480), range(640)], names=['row', 'column'])
    # Reshape the flow array to a 2D shape (480*640, 2)
    flow_2d = all4.reshape(-1, 4)
    # abc_2d = abc.reshape(-1, 2)
    
    # Create a DataFrame with the MultiIndex
    df = pd.DataFrame(flow_2d, index=index, columns=['fx', 'fy', 'mag', 'ang'])
    # df_abc = pd.DataFrame(abc_2d, index=index, columns=['mag', 'ang'])
    f_list.append(frame)
    dataframe_list.append(df)
    # abc_dataframe.append(df_abc)
    frame += 1    
df = pd.concat( dataframe_list, keys=f_list )
# df_abc = pd.concat( abc_dataframe, keys=f_list )

# nested_list = [flow,abc]
# all4 = np.dstack((flow, abc))


frame_1 =  df.loc[0,:,:]


cap.release()
cv2.destroyAllWindows()