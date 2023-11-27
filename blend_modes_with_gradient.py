import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import blend_modes


# # Import background image
# background_img_raw = Image.open('20230807_normal/20230807163300.png')  # RGBA image
# background_img = np.array(background_img_raw)  # Inputs to blend_modes need to be numpy arrays.
# # Rescale the pixel values to the range [0, 255]
# scaled_img = ((background_img - np.min(background_img)) / (np.max(background_img) - np.min(background_img)) * 255).astype(np.uint8)
# # Create an RGBA image with the rescaled content
# rgba_image = np.zeros((background_img.shape[0], background_img.shape[1], 4), dtype=np.uint8)
# # Copy the rescaled values to the first three channels (RGB)
# rgba_image[:, :, 0:3] = scaled_img[:, :, np.newaxis]
# # Set the alpha channel to a constant value (255 for fully opaque)
# alpha_value = 255
# rgba_image[:, :, 3] = alpha_value
# # Now 'rgba_image' is an RGBA image with dimensions (480, 640, 4)
# # plt.imshow(rgba_image)
# # Inputs to blend_modes need to be floats.
# background_img_float = rgba_image.astype(float)


def TwoDToRBGA (filepath):
    background_img_raw = Image.open(filepath)  # RGBA image
    background_img = np.array(background_img_raw)  # Inputs to blend_modes need to be numpy arrays.
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
    # plt.imshow(rgba_image)
    return rgba_image
    
rgba_image = TwoDToRBGA ('20230807_normal/20230807163300.png')
# Inputs to blend_modes need to be floats.
background_img_float = rgba_image.astype(float)


# Import foreground image
foreground_img_raw = Image.open('gradiente.png')  # RGBA image
foreground_img = np.array(foreground_img_raw)  # Inputs to blend_modes need to be numpy arrays.
foreground_img_float = foreground_img.astype(float)  # Inputs to blend_modes need to be floats.







# Blend images
opacity = 1  # The opacity of the foreground that is blended onto the background is 70 %.
blended_img_float = blend_modes.multiply(background_img_float, foreground_img_float, opacity)



# Convert blended image back into PIL image
blended_img = np.uint8(blended_img_float)  # Image needs to be converted back to uint8 type for PIL handling.
blended_img_raw = Image.fromarray(blended_img)  # Note that alpha channels are displayed in black by PIL by default.
                                                # This behavior is difficult to change (although possible).
                                                # If you have alpha channels in your images, then you should give
                                                # OpenCV a try.
# # Display blended image
# blended_img_raw.show()
plt.imshow(blended_img, cmap='gray')