import pvlib
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error

tz = 'America/Bogota'
lat, lon = 9.789103, -73.722451 # 9.789103, -73.722451 Esta es las coordenas
altitude = 50

#Ubicación Geográfica
location = pvlib.location.Location(lat, lon, tz, altitude)

times = pd.date_range('2023-01-01 00:00:00', '2023-12-31', closed='left',
                      freq='H', tz=tz)
solpos = pvlib.solarposition.get_solarposition(times, lat, lon)

# remove nighttime
solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

ax = plt.subplot(1, 1, 1, projection='polar')
# draw the analemma loops
points = ax.scatter(np.radians(solpos.azimuth), solpos.apparent_zenith,
                    s=2, label=None, c=solpos.index.dayofyear)
ax.figure.colorbar(points)

# draw hour labels
for hour in np.unique(solpos.index.hour):
    # choose label position by the smallest radius for each hour
    subset = solpos.loc[solpos.index.hour == hour, :]
    r = subset.apparent_zenith
    pos = solpos.loc[r.idxmin(), :]
    ax.text(np.radians(pos['azimuth']), pos['apparent_zenith'], str(hour))

# draw individual days
for date in pd.to_datetime(['2023-03-20', '2023-06-21', '2023-09-23', '2023-12-22', '2023-08-07']):
    times = pd.date_range(date, date+pd.Timedelta('24h'), freq='5min', tz=tz)
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
    label = date.strftime('%Y-%m-%d')
    azimuth_radians = np.radians(solpos.azimuth)
    ax.plot(azimuth_radians, solpos.apparent_zenith, label=label)

ax.figure.legend(loc='upper left')

# change coordinates to be like a compass
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rmax(90)

plt.show()


# for i in range(1,5):
#     fig = plt.gcf()
    
#     axes_coords = [0, 0, 1, 1] # plotting full width and height
    
#     pic = plt.imread('rotated'+ str(i)+'.png')
#     ax_image = fig.add_axes(axes_coords)
#     ax_image.imshow(pic, alpha=0.9)
#     ax_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image
    
#     ax_polar = fig.add_axes(axes_coords, projection='polar')
#     ax_polar.patch.set_alpha(0)
#     ax_polar.plot(azimuth_radians, solpos.apparent_zenith, label=label)
#     ax_polar.set_theta_zero_location('N')
#     ax_polar.set_theta_direction(-1)
#     ax_polar.set_rmax(90)
    
    
#     plt.show()


fig = plt.gcf()

axes_coords = [0, 0, 1, 1] # plotting full width and height


folder_path = 'D:\\RODRIGO\\Tesis IMEC\\Python\\Angulos_proc'
if os.path.exists(folder_path):
    direction = glob.glob(f'{folder_path}/*.png')
else:
    print(f"Folder '{folder_path}' does not exist.")

counter = 0  # Initialize a counter to keep track of the number of images processed
for filename in direction:
    if counter > 0 and counter % 1 == 0:  # Process an image every 100 images
        pic = plt.imread(filename)
        
        # Add your image processing code here
        ax_image = fig.add_axes(axes_coords)
        ax_image.imshow(pic, alpha=0.9)
        ax_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image

        ax_polar = fig.add_axes(axes_coords, projection='polar')
        ax_polar.patch.set_alpha(0)
        ax_polar.plot(azimuth_radians, solpos.apparent_zenith, label=label)
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.set_rmax(90)

        plt.savefig(filename)

    counter += 1  # Increment the counter after processing each image

















