# Feature extraction from infrared sky images for solar energy estimation

This repository contains all the necessary code to extract features from a set of infrared `.JP2` images taken with a sky cam.


## Getting started

These instructions show how to set up the code for it to run on your local machine.

### Prerequisites

- You must have installed python 3.11.5 or higher
- pip package manager

### Installation

It is suggested to create a virtual environment, for this make sure venv in installed or install it from terminal
```bash
   pip install virtualenv
```
```bash
   python -m venv skyImEnv
   skyImEnv\Scripts\activate
```

Clone the repository
```bash
   git clone https://github.com/rodhervan/SkyImFeatures/tree/main
```

Navigate to the project directory and install the prerequisite libraries
```bash
   cd  SkyImFeatures
   pip install -r requirements.txt
```
## Usage

To use this code there are several ways. Although cloning this repository will add all the folders, only one folder is necessary for the code to work. This said folder should be located in the same folder as the scripts and named `JP2_files`, inside this folder will be the location for the `.jp2` files. The naming convention is of the type `YYYYMMDD` for the folder containing the files of a single day. The day files should also be named `YYYYMMDDhhmmss.jp2` as so, an example file should look like `JP2_files/20230807/20230807000000.jp2`. Example files for day 2023/08/07 are here. 

To run the code there are different approaches. The most simple way is to run the  [save_to_json.py](https://github.com/rodhervan/SkyImFeatures/blob/main/save_to_json.py) file. 
