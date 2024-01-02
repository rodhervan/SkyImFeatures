import imageio.v2 as imageio
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import *
from skimage.metrics import *
from skimage.segmentation import *
from skimage.feature import *
from skimage.measure import label, regionprops, centroid
from skimage import measure, color, io
from scipy import ndimage as ndi
from scipy.interpolate import *
import pvlib
import copy
import pandas as pd
import blend_modes
import warnings
import os
import json
warnings.simplefilter(action='ignore', category=FutureWarning)



def normalize_img(img):
    gauss = gaussian(img, sigma=20)*2**16-1
    img[img>np.max(gauss)]=np.max(gauss)
    return img
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
def remove_small(slc, c=0.0001, remove_big = False):
    new_slc = slc.copy()
    max_area = slc.shape[0]*slc.shape[1]
    labels = label(slc,connectivity=1,background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    if remove_big:
        idxs = np.where(areas/(max_area) > c)[0]
    else:
        idxs = np.where(areas/(max_area) < c)[0]
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
# White circle to mask camera
def white_circle(slicee, cutoff_radius = 200):
    shape = (480,640); center = (235, 314); radius = cutoff_radius
    y, x = np.ogrid[:shape[0], :shape[1]]
    circle = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    circle_image = circle * slicee
    return circle_image
def solar_pos(filepath):
    
    tz = 'America/Bogota'
    lat, lon = 9.789103, -73.722451 # 9.789103, -73.722451 Esta es las coordenas
    altitude = 50

    #Ubicación Geográfica
    location = pvlib.location.Location(lat, lon, tz, altitude)
    times = pd.date_range('2023-01-01 00:00:00', '2024-12-31', inclusive='left',
                          freq='H', tz=tz)
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    # remove nighttime
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
    # draw hour labels
    for hour in np.unique(solpos.index.hour):
        # choose label position by the smallest radius for each hour
        subset = solpos.loc[solpos.index.hour == hour, :]
        r = subset.apparent_zenith
        pos = solpos.loc[r.idxmin(), :]
        # ax.text(np.radians(pos['azimuth']), pos['apparent_zenith'], str(hour))
    YY = filepath[-18:-14]
    MM = filepath[-14:-12]
    DD = filepath[-12:-10]
    day = YY+'-'+MM+'-'+DD
    # draw individual days
    for date in pd.to_datetime([day]):
        times = pd.date_range(date, date+pd.Timedelta('24h'), freq='30s', tz=tz)
        solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
        solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
        label = date.strftime('%Y-%m-%d')
        azimuth_radians = np.radians(solpos.azimuth)

    # Convert polar coordinates to Cartesian coordinates
    x_direct = solpos.apparent_zenith * np.sin(azimuth_radians)
    y_direct = solpos.apparent_zenith * np.cos(azimuth_radians)

    x = -(x_direct)*3.18+312
    y =  y_direct*2 + 238
    
    # Rotate about center 3.5 degrees
    x_c = 314; y_c = 235; j_rot = np.deg2rad(3.5)
    x_rot = (x - x_c)* np.cos(j_rot)- (y-y_c)*np.sin(j_rot) + x_c
    y_rot = (x - x_c)* np.sin(j_rot)+ (y-y_c)*np.cos(j_rot) + y_c

    # Reflect about x axis
    x_final = x_rot
    y_final = -y_rot + 478

    return x_final, y_final, day

# gets the time from the filepath following the convention '~/YYYYMMDDhhmmss.png'
def get_time (filepath):
    st = filepath[-10:-4]
    hh = str(int(st[0:2])-5)
    if len(hh) == 1:
        hh = '0'+hh
    mm = st[2:4]
    ss = st[4:6]
    timer = hh + ':'+ mm + ':' + ss  
    return timer

def get_solar_coords (x_mapped, y_mapped, day, timer):
    x = x_mapped[day + ' ' + timer + '-05:00']
    y = y_mapped[day + ' ' + timer + '-05:00']
    return x, y

def solar_xy (timer, x_mapped, y_mapped, day):
    if int(timer[:-6]) <5:
        timer = '05:47:00'
        solar_x, solar_y =  get_solar_coords (x_mapped, y_mapped, day, timer)
        solar_y =solar_y
        covered = 'No sun'
    if (int(timer[:-6]) ==5)&(int(timer[-5:-3])<47):
        timer = '05:47:00'
        solar_x, solar_y =  get_solar_coords (x_mapped, y_mapped, day, timer)
        solar_y = solar_y
        covered = 'No sun'
    else:    
        solar_x, solar_y =  get_solar_coords (x_mapped, y_mapped, day, timer)
        solar_y = solar_y
        covered = 'say if yes or no'
    return solar_x, solar_y, covered

# Solar pos calibration 
def solar_calibration():
    # Load image
    solar_image = Image.open('solar_pos.png')
    solar_image = np.asarray(solar_image)
    
    calibration_images = [20230808112930, 20230808131100, 20230808140700, 20230808153730, 20230808162300,
                          20230808175830, 20230808190630, 20230808200130, 20230808213400]
    
    x_m = np.array([])
    y_m = np.array([])
    for im in calibration_images:
        im = str(im)
        img_s = im + '.jp2'
        # get the solar coords for image comparison
        x_mapped, y_mapped, day = solar_pos(img_s)
        timer = get_time (img_s)
        solar_x, solar_y, covered = solar_xy (timer, x_mapped, y_mapped, day)
        x_m = np.append(x_m, solar_x)
        y_m = np.append(y_m, solar_y)
        
    
    # Real data
    x_real = np.array([132, 178, 207, 262, 290, 354, 395.5, 427, 471.5])
    y_real = np.array([270, 255, 255.245, 252, 253, 257.978, 262.268, 272, 282.172])
    
    # Polynomial regression
    degree = 3  # degree of the polynomial
    coefficients_x = np.polyfit(x_m, x_real, degree)
    coefficients_y = np.polyfit(y_m, y_real, degree)
    poly_x = np.poly1d(coefficients_x)
    poly_y = np.poly1d(coefficients_y)
    return poly_x, poly_y

def rings (img, slicee,cutoff_radius=200, center=(235,314)):
    only_circle = img
    # Create a meshgrid of coordinates
    y, x = np.ogrid[:only_circle.shape[0], :only_circle.shape[1]]
    # Calculate the distance of each pixel from the center
    distance_map = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    # Create an array to store the average values for each ring
    average_values = np.zeros(cutoff_radius)
    std_dev_values = np.zeros(cutoff_radius)
    var_values = np.zeros(cutoff_radius)
    # Create a list to store pixel values and their coordinates
    pixel_values_and_coords = []
    
    # Iterate over each ring and store pixel values and coordinates, excluding masked values
    for r in range(1, cutoff_radius+1):
        ring_pixels = np.logical_and(distance_map >= r - 1, distance_map < r)
        ring_pixels = np.logical_and(ring_pixels, slicee)  # Apply the mask
        # Get the coordinates of pixels in the ring
        ring_coords = np.column_stack(np.where(ring_pixels))
        # Get the pixel values in the ring
        ring_values = only_circle[ring_pixels]
        # Calculate the average value for the current ring
        if ring_values.shape[0] != 0:
            average_values[r - 1] = np.mean(ring_values)
            std_dev_values[r - 1] = np.std(ring_values)
            var_values[r - 1] = np.var(ring_values)
        # Append pixel values and coordinates to the list
        pixel_values_and_coords.append((ring_values, ring_coords))
    return pixel_values_and_coords, average_values, std_dev_values, var_values

def segmentation (filepath):
    # # camera mask is multiplied to the image to make it the darkest part of it by -1
    camara = Image.open('camera.png')
    camara = np.asarray(camara)
    # Only the alpha channel is needed, and is divided by 255 to get the number in the range [0,1]
    slicee = camara[:,:,3]/255
    cutoff_radius = 200
    circle = white_circle(slicee).astype(int)
    slicee = slicee.astype(bool)
    mask_circ = circle > 0


    # Load image
    image = imageio.imread(filepath)
    gauss = gaussian(image, sigma=20)*2**16-1
    image[image>np.max(gauss)]=np.max(gauss)


    ## bird removal
    inv_mask = slicee == 0
    region_values = image[inv_mask]
    mean_value = np.mean(region_values)
    # print(mean_value)
    # Parameters for the rectangle
    image_size = image.shape
    width = 75; height = 250
    top_left = (314-width/2, 200)
    bottom_right = (top_left[0] + width, top_left[1] + height)
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    rectangle = (x >= top_left[0]) & (x <= bottom_right[0]) & (y >= top_left[1]) & (y <= bottom_right[1])
    rectangle_image = np.zeros(image_size)
    rectangle_image[rectangle] = 1
    rectangle_data = rectangle_image*image
    strip = rectangle_data[200:450,277:352]

    average_brightness_list = []
    average_std_list = []
    for y in range(height):
        row_pixels = strip[y, :]
        average_brightness = np.mean(row_pixels)
        average_brightness_list.append(average_brightness)
        average_std = np.std(row_pixels)
        average_std_list.append(average_std)
        
    for y in range(height):
        strip[y, :] -= average_brightness_list[y] + average_std_list[y]*0.05
        strip[y, :] = np.clip(strip[y, :], 0, None)
        
    thresh  = threshold_yen(strip)
    binary = strip > thresh
    big_mask = remove_small(binary, c=0.04)
    bird_mask = rectangle_data*0
    bird_mask[200:450,277:352] = big_mask
    bird_mask = (bird_mask > 0).astype(bool)
    final_mask = ~bird_mask & mask_circ

    # Image gets converted to int32
    int_img = image.astype(int)
    img = int_img * slicee
    img[img<1000] = np.max(np.min(int_img) - 1, 0)
    only_circle = ((img- np.min(img)) /
                    (np.max(img) - np.min(img)) * 2**16-1)*final_mask
    center = (235, 314)
    pixel_values_and_coords, average_values, std_dev_values, var_values = rings (only_circle, slicee,cutoff_radius, center)
    # Find the average value of ring 70
    reference_average_value = average_values[15] 
    # # Apply corrections on every ring basis
    for i in range(len(pixel_values_and_coords)):
        ring_data = pixel_values_and_coords[i]
        correction_value = reference_average_value - average_values[i]
        # Subtract the correction value and set negative values to zero
        corrected_ring_values = np.maximum(0, ring_data[0] + correction_value)
        pixel_values_and_coords[i] = (corrected_ring_values, ring_data[1])

    # Reconstruct the image
    reconstructed_image = np.zeros_like(only_circle)
    for ring_data in pixel_values_and_coords:
        ring_values, ring_coords = ring_data
        reconstructed_image[ring_coords[:, 0], ring_coords[:, 1]] = ring_values

    # Compute and plot new average values after corrections
    new_average_values = np.zeros(cutoff_radius)
    new_std_values = np.zeros(cutoff_radius)
    for i in range(len(pixel_values_and_coords)):
        ring_values = pixel_values_and_coords[i][0]
        if len(ring_values) > 0:
            new_average_values[i] = np.mean(ring_values)
            new_std_values[i] = np.std(ring_values)
        else:
            new_average_values[i] = 0
            new_std_values[i] = 0
        
    ## Define circle around sun position
    sun_values_and_coords, sun_avg, sun_std, sun_var = rings (reconstructed_image, slicee, 30, (new_solar_y, new_solar_x))

    # Calculate the differences between consecutive averages
    percentage_max = sun_avg/(np.max(sun_avg))*100
    # differences = np.diff(percentage_max)
    # second_d = np.diff(differences)
    # # Find local maxima indices
    # local_maxima_indices = np.where((differences[:-1] > 0) & (differences[1:] < 0))[0] + 1
    def evaluate_sun (percentage_max):
        if np.max(percentage_max) in percentage_max[0:6]:
            flag = 'sun'
            mean_6 = np.mean(percentage_max[6:])
            std_6 = np.std(percentage_max[6:])
            # max_id =  np.argmax(percentage_max[:6])
            if (mean_6 < 10)&(std_6<6):
                flag = 'sun_no_clouds'
            if (mean_6 > 10)&(std_6<9):
                flag = 'sun_minor_clouds'
            if (mean_6> 35)&(std_6<9):
                flag = 'sun_major_clouds'
            if any(i > 80 for i in percentage_max[6:]): 
                flag = 'cloudy_around_sun'   
        else:
            flag = 'sun_covered'
        return flag
         

    flag = evaluate_sun (percentage_max)
    sun_img = np.zeros_like(reconstructed_image) 
     
    if flag == 'sun':
        for ring_data in sun_values_and_coords[:]:
            ring_values, ring_coords = ring_data
            sun_img[ring_coords[:, 0], ring_coords[:, 1]] = ring_values
        thresh = np.max(sun_img)*0.5
        sun_mask = sun_img > thresh
        sun_mask=remove_small(sun_mask, c =0.00025, remove_big = True)
        lab, num = label(sun_mask, return_num=True)
        if num == 0:
           bad_calibration = True
           cut = False
        if num == 1:
            sun_x, sun_y = centroid(lab)
            sun_values_and_coords, sun_avg, sun_std, sun_var = rings (reconstructed_image, slicee, 30, (sun_x, sun_y))
            percentage_max = sun_avg/(np.max(sun_avg))*100
            flag = evaluate_sun (percentage_max)
            if flag == 'sun':
                bad_calibration = True
            else:
                cut = True
                sun_mask = lab == 1
        if (num > 1):
            bad_calibration = True
            cut = False


    elif ((flag=='sun_no_clouds')|(flag=='sun_minor_clouds')):
        cut = True
        
        for ring_data in sun_values_and_coords[:]:
            ring_values, ring_coords = ring_data
            sun_img[ring_coords[:, 0], ring_coords[:, 1]] = ring_values
            
            thresh = np.max(sun_img)*0.5
            sun_mask = sun_img > thresh
            
    elif (flag=='sun_major_clouds'):
        cut = True
        no_sun = copy.deepcopy(sun_values_and_coords)
        for n in range(0, 5):
            if (percentage_max[n]>80):
                no_sun[n][0][:] = -1
        for ring_data in no_sun[:]:
            ring_values, ring_coords = ring_data
            sun_img[ring_coords[:, 0], ring_coords[:, 1]] = ring_values
        sun_mask = sun_img == -1
        for ring_data in sun_values_and_coords[:]:
            ring_values, ring_coords = ring_data
            sun_img[ring_coords[:, 0], ring_coords[:, 1]] = ring_values
    else:
        cut = False
        sun_mask = np.zeros_like(reconstructed_image, dtype=bool)
        
    # print(flag)
    if cut:
        no_sun_image = ~sun_mask*reconstructed_image
    else:
        no_sun_image = reconstructed_image
    plt.imshow(no_sun_image)


    rgba_image = TwoDToRGBA (no_sun_image)
    image_size = rgba_image.shape
    gradient_center = (235, 314)  # Center of the gradient correponding to the center of the camera
    gradient_radius = 210  # Radius of the gradient
    gradient_1 = create_radial_gradient(image_size, gradient_center, gradient_radius)
    rgba_grad_1 = TwoDToRGBA (gradient_1)
    gradient_radius = 250  # Radius of the gradient
    # gradient_2 = create_radial_gradient(image_size, gradient_center, gradient_radius)
    # rgba_grad_2 = TwoDToRGBA (gradient_2)
    first_grad = multiply_with_gradient(rgba_image, rgba_grad_1, 1)
    # second_grad = multiply_with_gradient(first_grad, rgba_grad_2, 1)
    gauss = gaussian(first_grad[:,:,0], sigma=0.5)
    no_sun_image = ((gauss - np.min(gauss)) / (np.max(gauss) - np.min(gauss)) * 2**16-1)*mask_circ


    thresh  = threshold_otsu(no_sun_image)
    binary = no_sun_image > thresh

    big_mask = remove_small(binary)
    big_clouds = rgba_image[:, :, 0] * big_mask * mask_circ * ~sun_mask

    # For ease of use the single channel image gets converted to RGB (similar to the RGBA process)
    output_img = TwoDToRGB(big_clouds)
    return output_img, flag


filepath = '20230807/20230807163300.jp2'

poly_x, poly_y = solar_calibration()
x_mapped, y_mapped, day = solar_pos(filepath)
timer = get_time (filepath)
solar_x, solar_y, covered = solar_xy (timer, x_mapped, y_mapped, day)

new_solar_x = poly_x(solar_x)
new_solar_y = poly_y(solar_y)

output_img, sun = segmentation(filepath)


plt.clf()
plt.imshow(output_img)
plt.title(sun)



# import json

# Load sun data from sun_data.json


# Parameter definitions for LK optical flow and border detection
preseg = False
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
trajectories_vel = []
frame_idx = 0
flow = np.empty((480, 640, 2))

# Specify the path to the folder containing images

# image_folder = '20230807'


# presegemented image folder (for speed)
image_folder = '20230807_seg_corrected'
preseg = True
with open('sun_data.json', 'r', encoding='utf-8') as sun_file:
    sun_data = json.load(sun_file)



# Get the list of image files in the folder
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.jp2'))])
# Read and preprocess the first image
first_im = os.path.join(image_folder, image_files[0])

if not preseg:
    prev_gray, sun  = segmentation(first_im)
# # If the images are already segmented use the command below
if preseg:
    prev_gray = cv2.imread(first_im)

prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)

image_data = {}

# for image_file in image_files[900:965]:
for image_file in image_files[900:965]:
    # Construct the full path to the image
    image_path = os.path.join(image_folder, image_file)

    # Read and preprocess the image
    x_mapped, y_mapped, day = solar_pos(image_path)
    timer = get_time (image_path)
    solar_x, solar_y, covered = solar_xy (timer, x_mapped, y_mapped, day)
    new_solar_x = poly_x(solar_x)
    new_solar_y = poly_y(solar_y)
    if not preseg:
        frame, sun = segmentation(image_path)
    
    # Get sun data for the current image_file
    if preseg:
        frame = cv2.imread(image_path)
        preprocess_name = image_file[:-3]+'jp2'
        sun = sun_data.get(preprocess_name, "no value")
    
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
        good = d < 10
        # Store data for the current image
        # Convert NumPy arrays to Python lists
        trajectories_list = [[[float(coord) for coord in point] for point in trajectory] for trajectory in trajectories]
        trajectories_list_vel = [[[float(coord) for coord in point] for point in trajectory] for trajectory in trajectories_vel]

        image_data[image_file] = {
            'sun': sun,
            'trajectories': trajectories_list,
            'flow_info': trajectories_list_vel
        }
        
        
        new_trajectories = []
        new_trajectories_vel = []
        # Get all the trajectories
        for trajectory, trajectory_vel, (x, y), good_flag in zip(trajectories, trajectories_vel, p1.reshape(-1, 2), good):
            r_y, r_x = round(y), round(x)
            # if (r_y<480)&(r_x<640):
            vel_x, vel_y = flow[r_y, r_x, :]
            if not good_flag:
                continue
            trajectory.append((x, y)); trajectory_vel.append((vel_x, vel_y))
            if len(trajectory) > trajectory_len:
                del trajectory[0], trajectory_vel[0]
            new_trajectories.append(trajectory); new_trajectories_vel.append(trajectory_vel)
            # Draw the newest detected point
            cv2.circle(lk_img, (int(x), int(y)), 4, (0, 0, 255), -1)
            
        trajectories = new_trajectories; trajectories_vel = new_trajectories_vel

        # Draw all the trajectories
        cv2.polylines(lk_img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(lk_img, image_file, (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(lk_img, sun, (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(lk_img, 'Trajectories: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.circle(lk_img, (round(new_solar_x), round(new_solar_y)), 4, (255, 0, 0), -1)
        
        

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(gray)
        mask[:] = 255
        
        # # # Points generated from goodFeaturesToTrack
        # # Lastest point in latest trajectory
        # for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
        #     cv2.circle(mask, (x, y), 5, 0, -1)
        # # Detect the good features to track
        # p = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
        # if p is not None:
        #     # If good features can be tracked - add that to the trajectories
        #     for x, y in np.float32(p).reshape(-1, 2):
        #         trajectories.append([(x, y)])
                
        
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
                vel_x, vel_y = flow[round(y), round(x), :]
                trajectories_vel.append([(vel_x, vel_y)])
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

# Save the data as a JSON file
with open('image_data.json', 'w', encoding='utf-8') as f:
    json.dump(image_data, f, ensure_ascii=False, indent=4)