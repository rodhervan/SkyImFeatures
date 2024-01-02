import imageio.v2 as imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import blend_modes
from skimage.filters import threshold_yen, gaussian
from skimage.measure import label, regionprops
def TwoDToRGBA (img):
    background_img = np.array(img)  
    # Rescale the pixel values to the range [0, 255]
    scaled_img = ((background_img - np.min(background_img)) / (np.max(background_img) - np.min(background_img)) * 255).astype(np.uint8)
    # Creates an RGBA zeros image with the rescaled content
    rgba_image = np.zeros((background_img.shape[0], background_img.shape[1], 4), dtype=np.uint8)
    # The rescaled values get copied to every channel
    rgba_image[:, :, 0:3] = scaled_img[:, :, np.newaxis]
    # The alpha channel is set to a constant value of 255, for a fully opaque channel
    alpha_value = 255
    rgba_image[:, :, 3] = alpha_value
    # Now 'rgba_image' is an RGBA image with dimensions (~, ~, 4)
    return rgba_image
def TwoDToRGB(img):
    background_img = np.array(img)
    # Rescale the pixel values to the range [0, 255]
    scaled_img = ((background_img - np.min(background_img)) / (np.max(background_img) - np.min(background_img)) * 255).astype(np.uint8)
    # Create an RGB image with the rescaled content
    rgb_image = np.zeros((background_img.shape[0], background_img.shape[1], 3), dtype=np.uint8)
    # Copy the rescaled values to all three channels (RGB)
    rgb_image[:, :, :] = scaled_img[:, :, np.newaxis]
    # Now 'rgb_image' is an RGB image with dimensions (height, width, 3)
    return rgb_image
def remove_small(slc, c=0.0001):
    new_slc = slc.copy()
    labels = label(slc,connectivity=1,background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    idxs = np.where(areas/(640*480) < c)[0]
    for i in idxs:
        new_slc[tuple(rps[i].coords.T)] = 0
    return new_slc
def create_radial_gradient(size, center, radius):
    y, x = np.ogrid[:size[0], :size[1]]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    gradient = 1 - np.clip(distance / radius, 0, 1)
    return gradient
def multiply_with_gradient(image, gradient, opacity):
    background_img_float = image.astype(float)
    foreground_img_float = gradient.astype(float)
    blended_img_float = blend_modes.multiply(background_img_float, foreground_img_float, opacity)
    # Convert blended image back into PIL image
    blended_img = np.uint8(blended_img_float)
    return blended_img
def segmentation(filepath): 
    # Read the JP2 image
    image = imageio.imread(filepath)
    # Image gets converted to int32
    int_img = image.astype(int)
    # # camera mask is multiplied to the image to make it the darkest part of it by -1
    camara = Image.open('camera.png')
    camara = np.asarray(camara)
    # Only the alpha channel is needed, and is divided by 255 to get the number in the range [0,1]
    slicee = camara[:,:,3]/255
    img = int_img * slicee
    img[img<1000] = np.max(np.min(int_img) - 1, 0)
    rgba_image = TwoDToRGBA (img)
    image_size = img.shape
    gradient_center = (235, 314)  # Center of the gradient correponding to the center of the camera
    gradient_radius = 210  # Radius of the gradient
    gradient_1 = create_radial_gradient(image_size, gradient_center, gradient_radius)
    rgba_grad_1 = TwoDToRGBA (gradient_1)
    gradient_radius = 250  # Radius of the gradient
    gradient_2 = create_radial_gradient(image_size, gradient_center, gradient_radius)
    rgba_grad_2 = TwoDToRGBA (gradient_2)
    first_grad = multiply_with_gradient(rgba_image, rgba_grad_1, 1)
    second_grad = multiply_with_gradient(first_grad, rgba_grad_2, 1)
    gauss = gaussian(first_grad[:,:,0], sigma=0.5)
    scaled_img = ((gauss - np.min(gauss)) / (np.max(gauss) - np.min(gauss)) * 255)
    thresh  = threshold_yen(scaled_img)
    binary = scaled_img > thresh
    big_mask = remove_small(binary)
    label_big_mask = label(big_mask)
    big_clouds = rgba_image[:,:,0]*big_mask

    output_img = TwoDToRGB(big_clouds)


    return label_big_mask, output_img

# path to jp2 image
filepath = '20230807/20230807152030.jp2'
label_big_mask, big_clouds = segmentation(filepath)


import cv2
import time

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

    hsv = np.zeros_like(img)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr, mag, ang, hsv



# # start time to calculate FPS
# start = time.time()

# flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# # prevgray = gray

# # End time
# end = time.time()
# # calculate the FPS for current frame detection
# fps = 1 / (end-start)
# # print(f"{fps:.2f} FPS")

# draw_flow_img = draw_flow(gray, flow)
# draw_flow_prev = draw_flow(prevgray, flow)
# draw_hsv_img, mag, ang, hsv = draw_hsv(flow)

# plt.figure(figsize=(14,8))
# plt.subplot(1,2,1)
# plt.imshow(draw_flow_img)


# plt.subplot(1,2,2)
# plt.imshow(draw_hsv_img)




filepath = '20230807/20230807171730.jp2'
next_filepath = '20230807/20230807171800.jp2'

prev_mask, prev_img = segmentation(filepath)
mask, img = segmentation(next_filepath)

prevgray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


prev_gray = prevgray
frame_gray = gray


lk_params = dict(winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 20,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )


trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0


# Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
if len(trajectories) > 0:
    img0, img1 = prev_gray, frame_gray
    p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1

    new_trajectories = []

    # Get all the trajectories
    for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
        if not good_flag:
            continue
        trajectory.append((x, y))
        if len(trajectory) > trajectory_len:
            del trajectory[0]
        new_trajectories.append(trajectory)
        # Newest detected point
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    trajectories = new_trajectories

    # Draw all the trajectories
    cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
    cv2.putText(img, 'Puntos: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


# Update interval - When to update and detect new features
if frame_idx % detect_interval == 0:
    mask = np.zeros_like(frame_gray)
    mask[:] = 255

    # Lastest point in latest trajectory
    for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
        cv2.circle(mask, (x, y), 5, 0, -1)

    # Detect the good features to track
    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
    if p is not None:
        # If good features can be tracked - add that to the trajectories
        for x, y in np.float32(p).reshape(-1, 2):
            trajectories.append([(x, y)])


frame_idx += 1
prev_gray = frame_gray

# End time
end = time.time()
# calculate the FPS for current frame detection
# fps = 1 / (end-start)

# Show Results
# cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Optical Flow', img)
cv2.imshow('Mask', mask)


# if cv2.waitKey(10) & 0xFF == ord('q'):
#     break
    


# cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()


