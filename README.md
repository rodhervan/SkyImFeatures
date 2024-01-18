# Feature extraction from infrared sky images for solar energy estimation

This repository contains all the necessary code to extract features from a set of infrared `.JP2` images taken with a sky cam.


## Getting started

These instructions show how to set up the code for it to run on your local machine.

### Prerequisites

- You must have installed python 3.11.5 or higher
- pip package manager

### Installation

1. It is suggested to create a virtual environment (although optional), for this make sure venv is installed or install it from terminal
```bash
   pip install virtualenv
```
Create the virtual environment and activate it
```bash
   python -m venv skyImEnv
   skyImEnv\Scripts\activate
```

2. Clone the repository
```bash
   git clone https://github.com/rodhervan/SkyImFeatures/tree/main
```

3. Navigate to the project directory and install the prerequisite libraries
```bash
   cd  SkyImFeatures
   pip install -r requirements.txt
```
## Usage

### Naming convention and files
To use this code there are several ways. Although cloning this repository will add all the folders, only one is necessary for the code to work. The mentioned folder is named `JP2_files` and should be located inside the same directory as the scripts, here will be the location for the `.jp2` files that will be organized by date, each also with their own directory. The naming convention is of the type `YYYYMMDD` for any folder containing the files of a single day. The image files should also be named `YYYYMMDDhhmmss.jp2` as so, an example file should look like `JP2_files/20230807/20230807000000.jp2` (make sure that the name for the days in the images and the folder match). The `camera.png` file is also a requirement as it is used to mask the camera stick for all images. Example files for day 2023/08/07 are [here](https://drive.google.com/file/d/1ncm2ZZ2fJwmPjt4Bf-qrUbEVnFU3xBFt/view?usp=sharing).

### Jupyter notebook

The [Final_document.ipynb](https://github.com/rodhervan/SkyImFeatures/blob/main/Final_document.ipynb) file contains a detailed explanation on how the functions work. To run it the example files mentioned need to be downloaded. 

### Code usage

To run the code there are different approaches. The most simple way is to run the  [save_to_json.py](https://github.com/rodhervan/SkyImFeatures/blob/main/save_to_json.py) file. This code has two opperating modes although by default it will perform all the required processes. To select the desired folder go to line `561` and change it accordingly (following convention). That being said it can be useful to presegment the images beforehand to only get partial data. To do this run the [png_saver.py](https://github.com/rodhervan/SkyImFeatures/blob/main/png_saver.py) file. This will create the folders `Generated_data` and `Segmented_images`, containing a JSON file with all the sun positions, cloud fraction and sky conditions; the other folder will only contain the segmented images. With this data the [save_to_json.py](https://github.com/rodhervan/SkyImFeatures/blob/main/save_to_json.py) code can run in the other opperating mode. To make this possible comment line `561` and uncomment lines `570` and `571`. The results from this code are saved in the folders `Generated_data` as `image_data_YYYYMMDD.json` containing all the detected trajectories and their respective velocities as well as the previous data. The images showing the trajectories and speeds are also saved in the `Fully_processed` folder. 

Finally, once all the files are generated, a video file can be created using [final_video.py](https://github.com/rodhervan/SkyImFeatures/blob/main/final_video.py).
