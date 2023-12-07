import os, os.path
import numpy as np
# import the necessary packages
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label,regionprops
import blend_modes



def TwoDToRBGA (img):
    background_img_raw = img
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
  
def create_radial_gradient(size, center, radius):
    y, x = np.ogrid[:size[0], :size[1]]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    gradient = 1 - np.clip(distance / radius, 0, 1)
    return gradient

def multiply_with_gradient(image, gradient, opacity):
    # Inputs to blend_modes need to be floats.
    background_img_float = image.astype(float)
    foreground_img_float = gradient.astype(float)
    blended_img_float = blend_modes.multiply(background_img_float, foreground_img_float, opacity)

    # Convert blended image back into PIL image
    blended_img = np.uint8(blended_img_float)  # Image needs to be converted back to uint8 type for PIL handling.
    # blended_img_raw = Image.fromarray(blended_img)  # Note that alpha channels are displayed in black by PIL by default.
    return blended_img

def remove_small(slc, c=0.0001):
    new_slc = slc.copy()
    labels = label(slc,connectivity=1,background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    idxs = np.where(areas/(640*480) < c)[0]
    for i in idxs:
        new_slc[tuple(rps[i].coords.T)] = 0
    return new_slc



# Create the 'modified' directory if it doesn't exist
output_directory = '20230822_seg_ag5'
os.makedirs(output_directory, exist_ok=True)
 
# iterate over files in
# that directory
directory = '20230822_normal'
# directory = 'test_folder'
# directory = 'nf'
for filename in os.scandir(directory):
    if filename.is_file():
        # print(filename.path)
        # name = str(filename).replace('<DirEntry ','').replace('>','').replace("'","").replace('jpf','jpg')
        name = str(filename).replace('<DirEntry ','').replace('>','').replace("'","")
        input_name = directory + '/' + name 
        
        img = Image.open(input_name)
        img = asarray(img)
        raw = img
        
        camara = Image.open('camara.png')
        camara = asarray(camara)
        slicee = camara[:,:,3]
        img = img* slicee
        
        image_size = img.shape
        gradient_center = (235, 314)  # Center of the gradient
        gradient_radius = 200  # Radius of the gradient
        gradient_1 = create_radial_gradient(image_size, gradient_center, gradient_radius)
        rgba_grad_1 = TwoDToRBGA (gradient_1)
        
        gradient_radius = 250  # Radius of the gradient
        gradient_2 = create_radial_gradient(image_size, gradient_center, gradient_radius)
        rgba_grad_2 = TwoDToRBGA (gradient_2)
        
        rgba_image = TwoDToRBGA (img)
        
        first_grad = multiply_with_gradient(rgba_image, rgba_grad_1, 1)
        second_grad = multiply_with_gradient(first_grad, rgba_grad_2, 1)
        
        twoDArray = second_grad[:,:,0]
        non_zero_mask = twoDArray != 0
        non_zero_values = twoDArray[non_zero_mask]
        average_non_zero = np.mean(non_zero_values)
        std_non_zero = np.std(non_zero_values)
        # mask =  (second_grad[:,:,0] > average_non_zero+std_non_zero)
        mask =  (second_grad[:,:,0] > average_non_zero+5)
        
        mask.shape
        mask = clear_border(mask)
        mask = remove_small(mask)
        area_image = second_grad[:,:,0] * mask
        big_mask = remove_small(mask)
        small_area_img = second_grad[:,:,0] * big_mask
        output_img = TwoDToRBGA(small_area_img)
        
        new_path = os.path.join(output_directory, name)
        plt.imsave(new_path, output_img)
        
        # im = Image.fromarray(small_area_img)
        # image = im.save('new_file.jpeg')
        
        # cv2.imwrite(new_path, rotated) 
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        

