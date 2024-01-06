import imageio.v2 as imageio
from PIL import Image
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu, threshold_yen
from skimage.measure import label, regionprops, centroid
# from skimage import measure, color
import pvlib
import copy
import pandas as pd
import blend_modes
import warnings
# import os
# import json
# from datetime import datetime
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
        # solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
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

# gets the time from the filepath following the convention '~/YYYYMMDDhhmmss.jp2'
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




def bird_removal (image):
    # inv_mask = slicee == 0
    # region_values = image[inv_mask]
    # mean_value = np.mean(region_values)
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
    return bird_mask



def load_and_cut (filepath):
    # #  Load image
    image = imageio.imread(filepath)
    gauss = gaussian(image, sigma=20)*2**16-1
    image[image>np.max(gauss)]=np.max(gauss)
    
    # # camera mask is multiplied to the image to make it the darkest part of it by -1
    camara = Image.open('camera.png')
    camara = np.asarray(camara)
    # Only the alpha channel is needed, and is divided by 255 to get the number in the range [0,1]
    slicee = camara[:,:,3]/255
    circle = white_circle(slicee).astype(int)
    slicee = slicee.astype(bool)
    mask_circ = circle > 0
    bird_mask = bird_removal(image)
    final_mask = ~bird_mask & mask_circ
    
    # Image gets converted to int32
    int_img = image.astype(int)
    img = int_img * slicee
    img[img<1000] = np.max(np.min(int_img) - 1, 0)
    only_circle = ((img- np.min(img)) /
                    (np.max(img) - np.min(img))*255)*final_mask
    return only_circle, final_mask
  
def rings (img, final_mask, cutoff_radius=200, center=(235,314), glob_list = False):
    only_circle = img
    # Create a meshgrid of coordinates
    y, x = np.ogrid[:only_circle.shape[0], :only_circle.shape[1]]
    # Calculate the distance of each pixel from the center
    distance_map = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    # Create an array to store the average and std values for each ring
    average_values = np.zeros(cutoff_radius)
    std_dev_values = np.zeros(cutoff_radius)
    # Create a list to store pixel values and their coordinates
    pixel_values_and_coords = []
    
    # Iterate over each ring and store pixel values and coordinates, excluding masked values
    for r in range(1, cutoff_radius+1):
        ring_pixels = np.logical_and(distance_map >= r - 1, distance_map < r)
        ring_pixels = np.logical_and(ring_pixels, final_mask)  # Apply the mask
        # Get the coordinates of pixels in the ring
        ring_coords = np.column_stack(np.where(ring_pixels))
        # Get the pixel values in the ring
        ring_values = only_circle[ring_pixels]
        pixel_values_and_coords.append((ring_values, ring_coords))
        
    global_max = 0
    global_min = 0
    global_list = []
    # Find the global maximum and minimum
    for element in pixel_values_and_coords:
        array_values = element[0]
        if array_values.size > 0:
            local_max = np.max(array_values)
            local_min = np.min(array_values)
            for n in array_values:
                global_list.append(n)
            global_max = max(global_max, local_max)
            global_min = min(global_min, local_min)
    # Normalize each value in the arrays based on the global maximum and minimum
    normalized_data = []
    for element in pixel_values_and_coords:
        array_values = element[0]
        if array_values.size == 0:
            array_values = np.array([0.0])
        normalized_array = (array_values - global_min) / (global_max - global_min)*255
        normalized_data.append((normalized_array, *element[1:])) 
        
    average_values = []
    std_dev_values = []
    for normalized_element in normalized_data:
        normalized_array = normalized_element[0]
        # Calculate average and standard deviation
        average = np.mean(normalized_array)
        std_dev = np.std(normalized_array)
        # Append to the lists
        average_values.append(average)
        std_dev_values.append(std_dev)
    if glob_list:
        return normalized_data, average_values, std_dev_values, global_list
    else:
        return normalized_data, average_values, std_dev_values

# filepath = '20230808/20230808131100.jp2'
# filepath = '20230807/20230807220400.jp2'
def average_curve():
    filepath_list = ['20230808/20230808131100.jp2', '20230808/20230808140700.jp2',
                     '20230808/20230808153730.jp2', '20230808/20230808131700.jp2',
                     '20230808/20230808175830.jp2', '20230807/20230807220400.jp2',
                     '20230807/20230807221330.jp2' ,'20230807/20230807214200.jp2']
    # filepath_list = ['20230808/20230808131100.jp2', '20230808/20230808140700.jp2',
    #                  '20230808/20230808153730.jp2', '20230808/20230808131700.jp2',
    #                  '20230808/20230808175830.jp2']
    norm_average_values_list = []
    for filepath in filepath_list:
        only_circle, final_mask = load_and_cut (filepath) 
        center = (235, 314); cutoff_radius = 200
        pixel_values_and_coords, average_values, std_dev_values = rings (only_circle, final_mask,cutoff_radius, center)

        # norm_average_values = ((average_values - np.min(average_values)) /
        #                         (np.max(average_values) - np.min(average_values))*255)
        norm_average_values = average_values
        norm_average_values_list.append(norm_average_values)
    norm_average_values_array = np.array(norm_average_values_list)
    resulting_avg = np.mean(norm_average_values_array, axis=0)
    # plt.plot(resulting_avg, marker='o', linestyle='-', color='b', label='Average')
    return resulting_avg

def build_from_rings(im, pixel_values_and_coords):
    reconstructed_image = np.zeros_like(im)
    for ring_data in pixel_values_and_coords:
        ring_values, ring_coords = ring_data
        reconstructed_image[ring_coords[:, 0], ring_coords[:, 1]] = ring_values
    return reconstructed_image

def ring_cleaning(only_circle, final_mask, cutoff_radius = 200, center = (235, 314), return_stats = False):
    # def ring slicing and cleaning
    pixel_values_and_coords, average_values, std_dev_values, global_list = rings (
            only_circle, final_mask,cutoff_radius, center, glob_list=True)
    reference_average_value = 0
    flag = 'tbd'
    # mean = np.mean(average_values); max_of_avg = np.max(average_values); std = np.std(average_values)
    # percent_avg = mean/max_of_avg
    # percent_std = std/mean
    glob_mean = np.mean(global_list); glob_max = np.max(global_list); glob_std = np.std(global_list)
    all_mean = round(glob_mean/glob_max,4); all_std = round(glob_std/glob_mean,4)
    per_min =  round((np.sum(global_list < 0.1*glob_max) / len(global_list))*100,4)
    # print(all_mean,  all_std, per_min)
    # plt.figure(figsize=(9,6))
    # plt.subplot(2,2,3)  
    # plt.plot(global_list)
    # plt.text(20, 240, [all_mean,all_std,per_min], fontsize=12, color='red')
    # if ((percent_avg>0.4)&(percent_std<0.7)):
    if ((all_mean>0.4)&(all_std<0.5)&(per_min<1)):   
        reference_average_value = np.mean(average_values[15:30])/2
        flag = 'cloudy'
    # # Apply corrections on every ring basis
    for i in range(len(pixel_values_and_coords)):
        ring_data = pixel_values_and_coords[i]
        # coeff_var =std_dev_values[i]/average_values[i]*100
        # correction_value = reference_average_value - average_values[i]
        scale_factor = 0; degree = 2; exp_factor = 35
        exp_scale = scale_factor*np.exp(1/exp_factor*(-200+i))
        border_scale = -scale_factor*((1/200)*i)**degree+1+exp_scale
        correction_value = reference_average_value - average_values[i]*border_scale
        # Subtract the correction value and set negative values to zero
        corrected_ring_values = np.maximum(0, ring_data[0] + correction_value)
        pixel_values_and_coords[i] = (corrected_ring_values, ring_data[1])
    # Reconstruct the image
    reconstructed_image = build_from_rings(only_circle, pixel_values_and_coords)
    
    if return_stats:
        return reconstructed_image, flag, average_values, std_dev_values
    else: 
        return reconstructed_image, flag
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
    
def sun_mask_and_pos_predicted (reconstructed_image, final_mask, flag, coords, radius = 30):

    new_solar_y, new_solar_x = coords
    ## Define circle around sun position
    sun_values_and_coords, sun_avg, sun_std = rings (reconstructed_image, final_mask, 30, (new_solar_y, new_solar_x))

    # Calculate the differences between consecutive averages
    percentage_max = sun_avg/(np.max(sun_avg))*100
    # differences = np.diff(percentage_max)
    # second_d = np.diff(differences)
    # # Find local maxima indices
    # local_maxima_indices = np.where((differences[:-1] > 0) & (differences[1:] < 0))[0] + 1    
    cut = False
    bad_calibration = False
    if flag == 'tbd':
        # print(flag)
        flag = evaluate_sun (percentage_max)
        # print(flag)
    else:
        cut = False
        
    sun_img = np.zeros_like(reconstructed_image) 
    # Tries to fin the sun if the input coords where off
    if flag == 'sun':
        sun_img = build_from_rings(sun_img, sun_values_and_coords)
        thresh = np.max(sun_img)*0.5
        sun_mask = sun_img > thresh
        sun_mask=remove_small(sun_mask, c =0.00025, remove_big = True)
        lab, num = label(sun_mask, return_num=True)
        if num == 0:
            bad_calibration = True
            cut = False
        if num == 1:
            sun_x, sun_y = centroid(lab)
            sun_values_and_coords, sun_avg, sun_std = rings (reconstructed_image, final_mask, 30, (sun_x, sun_y))
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
        
        # for ring_data in sun_values_and_coords[:]:
        #     ring_values, ring_coords = ring_data
        #     sun_img[ring_coords[:, 0], ring_coords[:, 1]] = ring_values
        sun_img = build_from_rings(sun_img, sun_values_and_coords)
            
        thresh = np.max(sun_img)*0.5
        sun_mask = sun_img > thresh
        sun_x, sun_y = centroid(sun_mask)
            
    elif (flag=='sun_major_clouds'):
        cut = True
        no_sun = copy.deepcopy(sun_values_and_coords)
        for n in range(0, 5):
            if (percentage_max[n]>80):
                no_sun[n][0][:] = -1

        sun_img = build_from_rings(sun_img, no_sun[:]) 
        sun_mask = sun_img == -1
        sun_x, sun_y = centroid(sun_mask)
        sun_img = build_from_rings(sun_img, sun_values_and_coords)
    else:
        cut = False
        sun_mask = np.zeros_like(reconstructed_image, dtype=bool)
        
    # print(flag)
    if cut:
        no_sun_image = ~sun_mask*reconstructed_image
        out_x, out_y = sun_x, sun_y
    else:
        no_sun_image = reconstructed_image
        out_x, out_y = new_solar_y, new_solar_x
    # plt.imshow(no_sun_image)

    return no_sun_image, flag, out_x, out_y, bad_calibration, sun_mask


def segmentation (filepath, ret_coords = False):
    only_circle, final_mask = load_and_cut (filepath) 
    reconstructed_image, flag = ring_cleaning(only_circle, final_mask)
    no_sun_image, flag, out_x, out_y, bad_calibration, sun_mask = sun_mask_and_pos_predicted (
        reconstructed_image,final_mask, flag, (new_solar_y, new_solar_x))
    
    rgba_image = TwoDToRGBA (no_sun_image)
    grad_result = rgba_image
    
    image_size = rgba_image.shape
    gradient_center = (235, 314)  # Center of the gradient correponding to the center of the camera
    gradient_radius = 210  # Radius of the gradient
    gradient_1 = create_radial_gradient(image_size, gradient_center, gradient_radius)
    rgba_grad_1 = TwoDToRGBA (gradient_1)
    gradient_radius = 250  # Radius of the gradient
    # gradient_2 = create_radial_gradient(image_size, gradient_center, gradient_radius)
    # rgba_grad_2 = TwoDToRGBA (gradient_2)
    # # multiply with the first gradient
    grad_result = multiply_with_gradient(rgba_image, rgba_grad_1, 0.5)
    # # multiply with the second gradient
    # grad_result = multiply_with_gradient(first_grad, rgba_grad_2, 1)
    
    gauss = gaussian(grad_result[:,:,0], sigma=0.5)
    gradient_image = ((gauss - np.min(gauss)) / (np.max(gauss) - np.min(gauss)) * 255)*final_mask


    thresh  = threshold_otsu(gradient_image)
    binary = gradient_image > thresh
    
    total_mask = final_mask*~sun_mask
    big_mask = remove_small(binary)
    big_clouds = rgba_image[:, :, 0] * big_mask * total_mask
    cloud_factor = np.sum(big_mask)/np.sum(total_mask)
    # For ease of use the single channel image gets converted to RGB (similar to the RGBA process)
    output_img = TwoDToRGB(big_clouds)

    if ret_coords:
        return output_img, flag, cloud_factor, out_x, out_y, bad_calibration
    else:
        return output_img, flag, cloud_factor
preseg = False   
json_file = ''

# resulting_avg = average_curve()
# filepath = 'JP2_files/20230808/20230808213400.jp2'
# only_circle, final_mask = load_and_cut (filepath) 
# reconstructed_image, flag = ring_cleaning(only_circle, final_mask)
# plt.figure(figsize=(12,6))
# plt.subplot(1,3,1)    
# plt.imshow(only_circle)
# plt.axis('off')

# plt.subplot(1,3,2)   
# plt.imshow(reconstructed_image)
# plt.axis('off')



# path to jp2 image
filepath = 'JP2_files/20230807/20230807110400.jp2' ##### 0.5109 0.4782
# filepath = 'JP2_files/20230807/20230807131700.jp2' # 0.3884 0.6096
# filepath = 'JP2_files/20230807/20230807121300.jp2' # 0.4566 0.5645
# filepath = 'JP2_files/20230807/20230807180600.jp2' # 0.3307 0.5066
# filepath = 'JP2_files/20230807/20230807100230.jp2' # # # 0.5138 0.5315
# filepath = 'JP2_files/20230807/20230807163300.jp2' # 
# filepath = 'JP2_files/20230807/20230807152030.jp2' # 
# filepath = 'JP2_files/20230807/20230807192300.jp2' # 
# filepath = 'JP2_files/20230808/20230808192300.jp2' # 

# filepath = 'JP2_files/20230807/20230807143530.jp2' ###### 0.3951 0.4528

# filepath = 'JP2_files/20230808/20230808131700.jp2' # 
# filepath = 'JP2_files/20230808/20230808112930.jp2' # 
# filepath = 'JP2_files/20230808/20230808131100.jp2' # 
# filepath = 'JP2_files/20230808/20230808140700.jp2' # 
# filepath = 'JP2_files/20230808/20230808153730.jp2' # 
# filepath = 'JP2_files/20230808/20230808162300.jp2' # 
# filepath = 'JP2_files/20230807/20230807121300.jp2' # 
# filepath = 'JP2_files/20230808/20230808175830.jp2' # 

# filepath = 'JP2_files/20230808/20230808190630.jp2' # 

# filepath = 'JP2_files/20230808/20230808200130.jp2' # 

# filepath = 'JP2_files/20230808/20230808201030.jp2' ##

# filepath = 'JP2_files/20230808/20230808213400.jp2' ##### 0.4952 0.2914
# filepath = 'JP2_files/20230807/20230807121000.jp2' # 
# filepath = 'JP2_files/20230807/20230807175500.jp2' # 
# filepath = 'JP2_files/20230807/20230807174730.jp2' # 
# filepath = 'JP2_files/20230807/20230807143530.jp2' #### 0.3951 0.4528
# filepath = 'JP2_files/20230807/20230807102200.jp2' ##### 0.617 0.2958


# resulting_avg = average_curve()
poly_x, poly_y = solar_calibration()
x_mapped, y_mapped, day = solar_pos(filepath)
timer = get_time (filepath)
solar_x, solar_y, covered = solar_xy (timer, x_mapped, y_mapped, day)
new_solar_x = poly_x(solar_x); new_solar_y = poly_y(solar_y)
output_img, flag, cloud_factor = segmentation(filepath)

# plt.figure(figsize=(9,6))
plt.subplot(2,2,1)  
cut, mask = load_and_cut(filepath)
plt.imshow(cut)
plt.axis('off')


plt.subplot(2,2,2)   
plt.imshow(output_img)
plt.axis('off')
plt.tight_layout()











