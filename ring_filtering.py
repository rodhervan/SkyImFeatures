import imageio.v2 as imageio
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import *
from skimage.metrics import *
from skimage.segmentation import *
from skimage.feature import *
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from scipy.interpolate import CubicSpline
import pvlib
import pandas as pd
import blend_modes
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# path to jp2 image
# filepath = '20230807/20230807110400.jp2'
# filepath = '20230807/20230807131700.jp2'
# filepath = '20230807/20230807121300.jp2'
# filepath = '20230807/20230807180600.jp2'
# filepath = '20230807/20230807100230.jp2'
filepath = '20230807/20230807163300.jp2'


# filepath = '20230808/20230808112930.jp2'
# filepath = '20230808/20230808131100.jp2'
# filepath = '20230808/20230808140700.jp2'
# filepath = '20230808/20230808153730.jp2'
# filepath = '20230808/20230808162300.jp2'
# filepath = '20230808/20230808175830.jp2'
# filepath = '20230808/20230808190630.jp2'
# filepath = '20230808/20230808200130.jp2'
# filepath = '20230808/20230808201030.jp2'
# filepath = '20230808/20230808213400.jp2'



cutoff_radius = 200


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
# White circle to mask camera
def white_circle(slicee):
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

    # Adjust Cartesian coordinates for a (480,640) image
    x = -(x_direct)*3.18+312
    k=0.0073; x0=304.4; a=67.6; L=477
    x_mapped = L / (1 + np.exp(-k * (x - x0))) + a
    y_mapped =  y_direct*2 + 238

    # Rotate about center 3.5 degrees
    x_c = 314; y_c = 235; j_rot = np.deg2rad(3.5)
    x_rot = (x_mapped - x_c)* np.cos(j_rot)- (y_mapped-y_c)*np.sin(j_rot) + x_c
    y_rot = (x_mapped - x_c)* np.sin(j_rot)+ (y_mapped-y_c)*np.cos(j_rot) + y_c

    # Reflect about x axis
    x_final = x_rot
    y_final = -y_rot + 477

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



# # camera mask is multiplied to the image to make it the darkest part of it by -1
camara = Image.open('camera.png')
camara = np.asarray(camara)
# Only the alpha channel is needed, and is divided by 255 to get the number in the range [0,1]
slicee = camara[:,:,3]/255
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
big_mask = remove_small(binary, c=0.005)
bird_mask = rectangle_data*0
bird_mask[200:450,277:352] = big_mask

plt.subplot(1,4,1)
plt.plot(range(1, 250 + 1), average_brightness_list, marker='o', linestyle='-', color='b', label='Average')

plt.subplot(1,4,2)
plt.imshow(big_mask)

plt.subplot(1,4,3)
plt.imshow(strip)

im = rectangle_image*image
plt.subplot(1,4,4)
plt.imshow(im[200:450,277:352])
# plt.colorbar()

bird_mask = (bird_mask > 0).astype(bool)

plt.clf()
final_mask = ~bird_mask & mask_circ
plt.imshow(final_mask)



# Given data
x_m = np.array([136.03, 175.37, 203.44, 257.17, 286.81, 349.70, 391.02, 420.32, 459.47])
x_real = np.array([128, 172, 202, 257, 286.5, 349, 390.5, 422, 466.5])
# Perform cubic spline interpolation
spline = CubicSpline(x_m, x_real)
# get the solar coords for image comparison
x_mapped, y_mapped, day = solar_pos(filepath)
timer = get_time (filepath)
if int(timer[:-6]) <5:
    timer = '05:47:00'
    solar_x, solar_y =  get_solar_coords (x_mapped, y_mapped, day, timer)
    solar_y = 480 - solar_y
    covered = 'No sun'
if (int(timer[:-6]) ==5)&(int(timer[-5:-3])<47):
    timer = '05:47:00'
    solar_x, solar_y =  get_solar_coords (x_mapped, y_mapped, day, timer)
    solar_y = 480 - solar_y
    covered = 'No sun'
else:    
    solar_x, solar_y =  get_solar_coords (x_mapped, y_mapped, day, timer)
    solar_y = 480 - solar_y
    covered = 'say if yes or no'

new_solar_x = spline(solar_x)+5


plt.clf()
fig, ax = plt.subplots()
ax.imshow(image)
ax.plot(new_solar_x, solar_y, 'r.', markersize=3, label=label)








# # Image gets converted to int32
# int_img = image.astype(int)
# img = int_img * slicee
# img[img<1000] = np.max(np.min(int_img) - 1, 0)
# only_circle = ((img- np.min(img)) /
#                 (np.max(img) - np.min(img)) * 2**16-1)*final_mask

# # Create a meshgrid of coordinates
# y, x = np.ogrid[:480, :640]
# # Calculate the distance of each pixel from the center
# distance_map = np.sqrt((x - 314)**2 + (y - 235)**2)
# # Create an array to store the average values for each ring
# average_values = np.zeros(cutoff_radius)
# std_dev_values = np.zeros(cutoff_radius)
# # Create a list to store pixel values and their coordinates
# pixel_values_and_coords = []

# # Iterate over each ring and store pixel values and coordinates, excluding masked values
# for r in range(1, cutoff_radius+1):
#     ring_pixels = np.logical_and(distance_map >= r - 1, distance_map < r)
#     ring_pixels = np.logical_and(ring_pixels, slicee)  # Apply the mask
#     # Get the coordinates of pixels in the ring
#     ring_coords = np.column_stack(np.where(ring_pixels))
#     # Get the pixel values in the ring
#     ring_values = only_circle[ring_pixels]
#     # Calculate the average value for the current ring
#     average_values[r - 1] = np.mean(ring_values)
#     std_dev_values[r - 1] = np.std(ring_values)
#     # Append pixel values and coordinates to the list
#     pixel_values_and_coords.append((ring_values, ring_coords))

# # Find the average value of ring 70
# reference_average_value = average_values[15] 
# # # Plot the average values before corrections
# plt.clf()
# plt.subplot(2, 2, 1)
# plt.plot(range(1, cutoff_radius + 1), average_values, marker='o', linestyle='-', color='b', label='Average')
# plt.plot(range(1, cutoff_radius + 1), std_dev_values, marker='o', linestyle='-', color='r', label='Standard Deviation')
# plt.title('Before Corrections')
# plt.xlabel('Ring Number')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# # # Apply corrections on a per ring basis
# for i in range(len(pixel_values_and_coords)):
#     ring_data = pixel_values_and_coords[i]
#     correction_value = reference_average_value - average_values[i]
#     # Subtract the correction value and set negative values to zero
#     corrected_ring_values = np.maximum(0, ring_data[0] + correction_value)
#     pixel_values_and_coords[i] = (corrected_ring_values, ring_data[1])

# # Reconstruct the image
# reconstructed_image = np.zeros_like(only_circle)
# for ring_data in pixel_values_and_coords:
#     ring_values, ring_coords = ring_data
#     reconstructed_image[ring_coords[:, 0], ring_coords[:, 1]] = ring_values

# # Compute and plot new average values after corrections
# new_average_values = np.zeros(cutoff_radius)
# new_std_values = np.zeros(cutoff_radius)
# for i in range(len(pixel_values_and_coords)):
#     new_average_values[i] = np.mean(pixel_values_and_coords[i][0])
#     new_std_values[i] = np.std(pixel_values_and_coords[i][0])
    

# plt.subplot(2, 2, 2)
# plt.plot(range(1, cutoff_radius+1), new_average_values, marker='o', linestyle='-', color='b', label='Average')
# plt.plot(range(1, cutoff_radius+1), new_std_values, marker='o', linestyle='-', color='r', label='Standard Deviation')
# plt.title('After Corrections')
# plt.xlabel('Ring Number')
# plt.ylabel('Value')
# plt.grid(True)

# # Display the original input image
# plt.subplot(2, 2, 3)
# plt.imshow(only_circle)
# plt.title('Input Image')
# plt.colorbar()


# # Display the reconstructed image
# plt.subplot(2, 2, 4)
# plt.imshow(reconstructed_image)
# plt.title('Reconstructed Image')
# plt.colorbar()

# # Adjust layout for better visualization
# plt.tight_layout()
# plt.show()



# rgba_image = TwoDToRGBA (reconstructed_image)
# image_size = rgba_image.shape
# gradient_center = (235, 314)  # Center of the gradient correponding to the center of the camera
# gradient_radius = 210  # Radius of the gradient
# gradient_1 = create_radial_gradient(image_size, gradient_center, gradient_radius)
# rgba_grad_1 = TwoDToRGBA (gradient_1)
# gradient_radius = 250  # Radius of the gradient
# gradient_2 = create_radial_gradient(image_size, gradient_center, gradient_radius)
# rgba_grad_2 = TwoDToRGBA (gradient_2)
# first_grad = multiply_with_gradient(rgba_image, rgba_grad_1, 1)
# # second_grad = multiply_with_gradient(first_grad, rgba_grad_2, 1)
# gauss = gaussian(first_grad[:,:,0], sigma=0.5)
# reconstructed_image = ((gauss - np.min(gauss)) / (np.max(gauss) - np.min(gauss)) * 2**16-1)*mask_circ


# # fig, ax = try_all_threshold(reconstructed_image, figsize=(10, 6), verbose=False)

# thresh  = threshold_otsu(reconstructed_image)
# binary = reconstructed_image > thresh

# big_mask = remove_small(binary)
# label_big_mask = label(big_mask)
# big_clouds = rgba_image[:, :, 0] * big_mask * mask_circ

# # For ease of use the single channel image gets converted to RGB (similar to the RGBA process)
# output_img = TwoDToRGB(big_clouds)
# plt.clf()
# plt.imshow(output_img)


# # window_size=25
# # thresh  = threshold_niblack(scaled_img, window_size=window_size, k=0.4)
# # binary = scaled_img > thresh

# # big_mask = remove_small(binary)
# # label_big_mask = label(big_mask)
# # big_clouds = rgba_image[:, :, 0] * big_mask * mask_circ

# # # For ease of use the single channel image gets converted to RGB (similar to the RGBA process)
# # output_img = TwoDToRGB(big_clouds)

# # plt.imshow(output_img)





