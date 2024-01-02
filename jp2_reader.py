# import imageio.v2 as imageio
# from PIL import Image
# import numpy as np

# filepath = '20230807/20230807152030.jp2'

# # Read the JP2 image
# image = imageio.imread(filepath)    

# # Now 'image' contains the image data, and you can use it as needed
# # For example, you can display the image using a library like matplotlib
# import matplotlib.pyplot as plt

# int_array = image.astype(int)

# plt.imshow(int_array)
# plt.colorbar()



import imageio.v2 as imageio
from PIL import Image
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import cv2

filepath = '20230807/20230807180530.jp2'
# filepath = '20230807/20230807171730.jp2'


# Read the JP2 image
image = imageio.imread(filepath)    
int_img = image.astype(int)

# Perform histogram equalization
equalized_image = exposure.equalize_hist(image)

# Convert to integer for display
equalized_int_array = (equalized_image * 255).astype(int)


camara = Image.open('camera.png')
camara = np.asarray(camara)
slicee = camara[:,:,3]/255
img = int_img * slicee
img[img<1000] = np.max(np.min(int_img) - 1, 0)

# print(img.shape)
plt.figure(figsize=(8,6))
# ÷plt.pcolormesh(img, cmap='Greys_r')
# plt.pcolormesh(img)
# plt.colorbar()



#   # Inputs to blend_modes need to be numpy arrays.
# # Create an RGBA image with the rescaled content
# rgba_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint16)
# # Copy the rescaled values to the first three channels (RGB)
# rgba_image[:, :, 0:3] = img[:, :, np.newaxis]
# # Set the alpha channel to a constant value (255 for fully opaque)
# alpha_value = 255
# rgba_image[:, :, 3] = alpha_value
# # Now 'rgba_image' is an RGBA image with dimensions (480, 640, 4)
#     # return rgba_image
# # rgba_image = TwoDToRGBA (img)

# # plt.imshow(rgba_image)
# # plt.colorbar()

# import cv2
# cv2.imshow('tiff', rgba_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def create_radial_gradient(size, center, radius):

    y, x = np.ogrid[:size[0], :size[1]]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    gradient = 1 - np.clip(distance / radius, 0, 1)
    return gradient

def TwoDToRGBA (img):

    background_img = np.array(img)  # Inputs to blend_modes need to be numpy arrays.
    # Rescale the pixel values to the range [0, 255]
    scaled_img = ((background_img - np.min(background_img)) / (np.max(background_img) - np.min(background_img)) * 255).astype(np.uint8)
    # Create an RGBA image with the rescaled content
    rgba_image = np.zeros((background_img.shape[0], background_img.shape[1], 4), dtype=np.uint8)
    # Copy the rescaled values to the first three channels (RGB)
    rgba_image[:, :, 0:3] = scaled_img[:, :, np.newaxis]
    # Set the alpha channel to a constant value (255 for fully opaque)
    alpha_value = 255
    rgba_image[:, :, 3] = alpha_value
    # Now 'rgba_image' is an RGBA image with dimensions (480, 640, 4)
    return rgba_image

import blend_modes
def multiply_with_gradient(image, gradient, opacity):
    # Inputs to blend_modes need to be floats.
    background_img_float = image.astype(float)
    foreground_img_float = gradient.astype(float)
    blended_img_float = blend_modes.multiply(background_img_float, foreground_img_float, opacity)

    # Convert blended image back into PIL image
    blended_img = np.uint8(blended_img_float)  # Image needs to be converted back to uint8 type for PIL handling.
    # blended_img_raw = Image.fromarray(blended_img)  # Note that alpha channels are displayed in black by PIL by default.
    return blended_img


rgba_image = TwoDToRGBA (img)

image_size = img.shape
gradient_center = (235, 314)  # Center of the gradient
gradient_radius = 210  # Radius of the gradient
gradient_1 = create_radial_gradient(image_size, gradient_center, gradient_radius)
rgba_grad_1 = TwoDToRGBA (gradient_1)

gradient_radius = 250  # Radius of the gradient
gradient_2 = create_radial_gradient(image_size, gradient_center, gradient_radius)
rgba_grad_2 = TwoDToRGBA (gradient_2)


first_grad = multiply_with_gradient(rgba_image, rgba_grad_1, 1)
second_grad = multiply_with_gradient(first_grad, rgba_grad_2, 1)

# cv2.imshow('tiff', second_grad)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(first_grad[:,:,0], cmap="jet")
# plt.colorbar()



from skimage.measure import label, regionprops
import matplotlib.patches as mpatches


from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb



labels = first_grad[:,:,0]



from skimage.filters import try_all_threshold, threshold_yen, gaussian

gauss = gaussian(labels, sigma=0.8)
scaled_img = ((gauss - np.min(gauss)) / (np.max(gauss) - np.min(gauss)) * 255)

thresh  = threshold_yen(scaled_img)
binary = scaled_img > thresh
fig, ax = try_all_threshold(scaled_img, figsize=(10, 6), verbose=False)
# plt.subplot(2,1,1)
# plt.imshow(binary)

# plt.subplot(2,1,2)
# # #mask = labels > 70

# plt.imshow(scaled_img, cmap='jet')
# plt.colorbar()









