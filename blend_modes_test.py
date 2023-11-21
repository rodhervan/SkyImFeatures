from PIL import Image
import numpy
import blend_modes


# Import background image
background_img_raw = Image.open('20230807_normal/20230807163300.png')  # RGBA image
background_img = numpy.array(background_img_raw)  # Inputs to blend_modes need to be numpy arrays.
background_img_float = background_img.astype(float)  # Inputs to blend_modes need to be floats.
background_img_float = numpy.stack((background_img_float, background_img_float, background_img_float, numpy.ones_like(background_img_float)), axis=-1)

# Import foreground image
foreground_img_raw = Image.open('20230807_normal/20230807163300.png')  # RGBA image
foreground_img = numpy.array(foreground_img_raw)  # Inputs to blend_modes need to be numpy arrays.
foreground_img_float = foreground_img.astype(float)  # Inputs to blend_modes need to be floats.
foreground_img_float = numpy.stack((foreground_img_float, foreground_img_float, foreground_img_float, 0.5*numpy.ones_like(foreground_img_float)), axis=-1)


# Blend images
opacity = 0.7  # The opacity of the foreground that is blended onto the background is 70 %.
blended_img_float = blend_modes.multiply(background_img_float, foreground_img_float, opacity)



# Convert blended image back into PIL image
blended_img = numpy.uint8(blended_img_float)  # Image needs to be converted back to uint8 type for PIL handling.
blended_img_raw = Image.fromarray(blended_img)  # Note that alpha channels are displayed in black by PIL by default.
                                                # This behavior is difficult to change (although possible).
                                                # If you have alpha channels in your images, then you should give
                                                # OpenCV a try.

# Display blended image
blended_img_raw.show()