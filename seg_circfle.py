import imageio.v2 as imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import *
from skimage.metrics import *
from skimage.segmentation import *
from skimage.feature import *
from scipy import ndimage as ndi
import blend_modes
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# path to jp2 image
# filepath = '20230807/20230807110400.jp2'
# filepath = '20230807/20230807110330.jp2'
# filepath = '20230807/20230807121300.jp2'
# filepath = '20230807/20230807180600.jp2'
filepath = '20230807/20230807100230.jp2'
# filepath = '20230807/20230807163300.jp2'




from skimage.measure import label, regionprops
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
    shape = (480,640); center = (235, 314); radius = 200
    y, x = np.ogrid[:shape[0], :shape[1]]
    circle = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    circle_image = circle * slicee
    return circle_image

def white_donut(slicee, internal_radius, external_radius):
    shape = (480,640); center = (235, 314)
    y, x = np.ogrid[:shape[0], :shape[1]]
    circle1 = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= external_radius ** 2
    circle2 = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= internal_radius ** 2
    donut = circle2^circle1
    donut_image = donut * slicee
    return donut_image


# # camera mask is multiplied to the image to make it the darkest part of it by -1
camara = Image.open('camera.png')
camara = np.asarray(camara)
# Only the alpha channel is needed, and is divided by 255 to get the number in the range [0,1]
slicee = camara[:,:,3]/255
circle = white_circle(slicee).astype(int)
mask_circ = circle > 0


image = imageio.imread(filepath)
gauss = gaussian(image, sigma=20)*2**16-1
image[image>np.max(gauss)]=np.max(gauss)
# Image gets converted to int32
int_img = image.astype(int)
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
second_grad = multiply_with_gradient(first_grad, rgba_grad_2, 0)
gauss = gaussian(second_grad[:,:,0], sigma=0.5)
scaled_img = ((gauss - np.min(gauss)) / (np.max(gauss) - np.min(gauss)) * 2**16-1)*mask_circ


# Apply the mask to the image
region_values = scaled_img[mask_circ]

# Calculate descriptive statistics
mean_value = np.mean(region_values)
std_dev = np.std(region_values)

# # Plot histogram
# plt.hist(region_values, bins=30, color='blue', alpha=0.7)
# plt.title('Histogram of Values in the Selected Region')
# plt.xlabel('Pixel Values')
# plt.ylabel('Frequency')
# plt.show()

# # Print the statistics
# print(f'Mean value in the selected region: {mean_value}')
# print(f'Standard deviation in the selected region: {std_dev}')

if std_dev > 3000:
    window_size=25
    thresh  = threshold_niblack(scaled_img, window_size=window_size, k=0.4)
    binary = scaled_img > thresh
    
    big_mask = remove_small(binary)
    label_big_mask = label(big_mask)
    big_clouds = rgba_image[:, :, 0] * big_mask * mask_circ
    
    # For ease of use the single channel image gets converted to RGB (similar to the RGBA process)
    output_img = TwoDToRGB(big_clouds)
else:
    thresh  = threshold_yen(scaled_img)
    binary = scaled_img > thresh

    big_mask = remove_small(binary)
    label_big_mask = label(big_mask)
    big_clouds = rgba_image[:, :, 0] * big_mask * mask_circ

    # For ease of use the single channel image gets converted to RGB (similar to the RGBA process)
    output_img = TwoDToRGB(big_clouds)




plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
plt.imshow(output_img, cmap='gray')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(scaled_img)
plt.colorbar()


donut = white_donut(slicee, 100, 200).astype(int)
truth_donut = donut >0

# gauss = farid(image, mask_circ)
gauss = gaussian(scaled_img, sigma=15)
# gauss = meijering(scaled_img, sigmas=range(1, 2, 1))
# gauss = prewitt(scaled_img, mask=mask_circ)
# gauss = roberts(scaled_img, mask=mask_circ)
# gauss = sato(scaled_img)
# gauss = sobel(scaled_img, mask=mask_circ)
# gauss = corner_fast(scaled_img)
# gauss = canny(scaled_img,  sigma=3, low_threshold=15, high_threshold=100, mask=None, use_quantiles=False, mode='constant', cval=0.0)
# gauss = corner_shi_tomasi(scaled_img, sigma=0.5)

## preprocess if brightess is still high
grad_high = gradient_1**3

plt.subplot(2,2,3)
plt.imshow(gauss)
plt.colorbar()

fill_coins = ndi.binary_fill_holes(gauss)

inverse = gradient_1*1
inverse[inverse>0] = 1-inverse[inverse>0] 


img_donut = scaled_img*truth_donut
values = img_donut[truth_donut]

avg_donut = np.mean(values)
std_donut = np.std(values)
    

plt.clf()
# Plot histogram
# plt.hist(region_values, bins=30, color='blue', alpha=0.7)
# plt.title('Histogram of Values in the Selected Region')
# plt.xlabel('Pixel Values')
# plt.ylabel('Frequency')
# plt.show()

# # Print the statistics
# print(f'Mean value in the selected region: {mean_value}')
# print(f'Standard deviation in the selected region: {std_dev}') 


# print(avg_donut)
# print(std_donut)
# plt.subplot(2,2,4)
# plt.imshow(truth_donut)
# plt.colorbar()

# Create a meshgrid of coordinates
y, x = np.ogrid[:480, :640]

# Calculate the distance of each pixel from the center
distance_map = np.sqrt((x - 235)**2 + (y - 314)**2)

# Create an array to store the average values for each ring
average_values = np.zeros(200)

# Iterate over each ring and calculate the average value, excluding masked values
for r in range(1, 201):
    ring_pixels = np.logical_and(distance_map >= r - 1, distance_map < r)
    ring_pixels = np.logical_and(ring_pixels, slicee)  # Apply the mask
    average_values[r - 1] = np.mean(scaled_img[ring_pixels])


# Plot the average values
plt.plot(range(1, 201), average_values, marker='o', linestyle='-', color='b')
plt.title('Average Values for Each Ring')
plt.xlabel('Ring Number')
plt.ylabel('Average Value')
plt.grid(True)
plt.show()




