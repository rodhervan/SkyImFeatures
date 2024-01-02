import imageio.v2 as imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import blend_modes
from skimage.filters import threshold_yen, gaussian
from skimage.measure import label, regionprops
import os
import cv2
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
    gauss = gaussian(image, sigma=20)*2**16-1
    image[image>np.max(gauss)]=np.max(gauss)
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



# Create the 'modified' directory if it doesn't exist
output_directory = '20230807_seg_yen8'
os.makedirs(output_directory, exist_ok=True)


# iterate over files in
# that directory
directory = '20230807'
first_image = '20230807000000.jp2'

filepath = directory + '/' + first_image

# directory = 'test_folder'
# directory = 'nf'
for filename in os.scandir(directory):
    if filename.is_file():
        name = str(filename).replace('<DirEntry ','').replace('>','').replace("'","")
        next_filepath = directory + '/' + name 
        
        
        prev_mask, prev_img = segmentation(filepath)
        mask, img = segmentation(next_filepath)
        
        
        

        # # ax.plot(x_mapped, y_mapped,'r' , label=label)
        new_path = os.path.join(output_directory, name[:-4]+'.png')
        cv2.imwrite(new_path, img)
        
        
        # filepath = next_filepath
        # plt.show()nam
        
        



