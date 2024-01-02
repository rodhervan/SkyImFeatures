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
from scipy.interpolate import *
import pvlib
import copy
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
# filepath = '20230807/20230807163300.jp2'
# filepath = '20230807/20230807152030.jp2'
# filepath = '20230807/20230807192300.jp2'
# filepath = '20230808/20230808192300.jp2'




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

def solar_pos(filepath):
    
    tz = 'America/Bogota'
    lat, lon = 9.789103, -73.722451 # 9.789103, -73.722451 Esta es las coordenas
    altitude = 50

    #Ubicación Geográfica
    location = pvlib.location.Location(lat, lon, tz, altitude)
    times = pd.date_range('2023-01-01 00:00:00', '2024-12-31', closed='left',
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

    x = -(x_direct)*2.6+312
    y =  y_direct*2.23 + 236
    
    # Rotate about center 3.5 degrees
    x_c = 314; y_c = 235; j_rot = np.deg2rad(3.3)
    x_rot = (x - x_c)* np.cos(j_rot)- (y-y_c)*np.sin(j_rot) + x_c
    y_rot = (x - x_c)* np.sin(j_rot)+ (y-y_c)*np.cos(j_rot) + y_c

    # Reflect about x axis
    x_final = x_rot + 5
    y_final = y_rot + 5

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

# Load image
image = Image.open('solar_pos.png')
image = np.asarray(image)


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
y_real = np.array([270, 258, 255.245, 252, 253, 257.978, 262.268, 272, 282.172])

# Polynomial regression
degree = 3  # degree of the polynomial
coefficients_x = np.polyfit(x_m, x_real, degree)
coefficients_y = np.polyfit(y_m, y_real, degree)
poly_x = np.poly1d(coefficients_x)
poly_y = np.poly1d(coefficients_y)


new_solar_x = poly_x(x_m)
new_solar_y = poly_y(y_m)

plt.clf()
plt.subplot(1,2,1)
plt.imshow(image)
plt.plot(new_solar_x, new_solar_y, 'r.', markersize=1, label=label)



filepath = '20230808/20230808131700.jp2'
filepath = '20230807/20230807131700.jp2'

x_mapped, y_mapped, day = solar_pos(filepath)
timer = get_time (filepath)
solar_x, solar_y, covered = solar_xy (timer, x_mapped, y_mapped, day)

new_solar_x = poly_x(solar_x)
new_solar_y = poly_y(solar_y)


image = imageio.imread(filepath)
int_img = image.astype(int)

# plt.clf()
plt.subplot(1,2,2)
plt.imshow(int_img)
plt.plot(new_solar_x, new_solar_y, 'r.', markersize=1, label=label)
plt.title('sun_no_clouds')


