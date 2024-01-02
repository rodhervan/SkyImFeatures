import imageio.v2 as imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import blend_modes
from skimage.filters import threshold_yen, threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage import measure, color
import cv2
import time
import os



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


    # For ease of use the single channel image gets converted to RGB (similar to the RGBA process)
    output_img = TwoDToRGB(big_clouds)


    return label_big_mask, output_img





# Parameter definitos for LK optical flow and border detection
lk_params = dict(winSize=(15, 15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=7)

trajectory_len = 40
detect_interval = 5
trajectories = []
frame_idx = 0

# Specify the path to the folder containing images
image_folder = '20230807'

# Get the list of image files in the folder
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.jp2'))])
# Read and preprocess the first image
first_im = os.path.join(image_folder, image_files[0])
mask_1, prev_gray = segmentation(first_im)
prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)



for image_file in image_files[0:200]:
    # Construct the full path to the image
    image_path = os.path.join(image_folder, image_file)

    # Read and preprocess the image
    mask_1, frame = segmentation(image_path)
    # # If the images are already segmented use the command below
    # frame = cv2.imread(image_path)
    
    
    # Image gets transformed to single channel gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lk_img = frame.copy()

    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, gray
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
            cv2.circle(lk_img, (int(x), int(y)), 4, (0, 0, 255), -1)

        trajectories = new_trajectories

        # Draw all the trajectories
        cv2.polylines(lk_img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        
        cv2.putText(lk_img, 'Points: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)



        # prevgray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
        # framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        def draw_hsv(flow):

            hsv = np.zeros_like(frame)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return bgr, mag, ang, hsv

        draw_hsv_img, mag, ang, hsv = draw_hsv(flow)


        #### Label for velocity 
        # Read the image
        image = draw_hsv_img
        # Convert the image to the Lab color space
        image_lab = color.rgb2lab(image)
        # Separate the L, a, and b channels
        L, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]
        # Use Otsu's thresholding on the L channel to create a binary mask
        thresh = threshold_otsu(L)
        binary_mask = L > thresh
        small_binary = remove_small(binary_mask)
        # Label connected components in the binary mask
        labeled_image, num_labels = label(small_binary, connectivity=2, return_num=True)

        all_centroids = []

        for num in range(1,num_labels+1):
            num_mask = labeled_image == num
            x_flow = flow[:,:,0] * num_mask
            y_flow = flow[:,:,1] * num_mask
            non_zero_elements_x = x_flow[x_flow != 0]
            non_zero_elements_y = y_flow[y_flow != 0]
            average_non_zero_x = np.average(non_zero_elements_x)
            average_non_zero_y = np.average(non_zero_elements_y)
            
            over_img = frame[:,:,0] * num_mask
            mask_over_img = over_img !=0
            small_ovr = remove_small(mask_over_img, c=0.0001)
            label_over_small, small_id = label(small_ovr, return_num=True)

            for labl in range(1,small_id+1): 
                centroide = measure.centroid(label_over_small == labl)
                all_centroids.append(centroide)
                line_img = cv2.arrowedLine(frame,
                                    (int(centroide[1]), int(centroide[0])),
                                    (int(centroide[1] + average_non_zero_x * 5), int(centroide[0] + average_non_zero_y * 5)),
                                    (0,255,0), 2)

        centroids_array = np.float32(all_centroids).reshape(-1, 1, 2)
        # Use the points from centroids_array
        p = centroids_array
        
        # If points are available, add them to the trajectories
        if p is not None:
            for y, x in p.reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1
    prev_gray = gray

    # Show Results
    cv2.imshow('Optical Flow', lk_img)
    # cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

plt.figure(figsize=(10,6))
plt.title('Last image in set')
plt.imshow(lk_img)



