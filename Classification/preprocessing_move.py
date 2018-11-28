# -*- coding: utf-8 -*-
"""
This script resizes (optionally) the training data to a resolution of choice
and copies the training data to folders containing training and validation sets. 
The splitting ratio can be defined in the definitions below. 

Considering the flow_from_directory function in keras, the training and 
validation data continuously flow from the directories specified as inputs. 
An example:
        train_data         
            1            -- a subfolder containing positive class images
            0            -- a subfolder containing positive class images
        validation_data
            1            -- a subfolder containing positive class images
            0            -- a subfolder containing positive class images
        
Therefore, the script moves all of the data to folders specified above in a 
random fashion. 

This script splits the input data for a one-vs-all classifier only. 
Change the 'disease' variable in order to choose either MM-vs-all or SK-vs-all.

The flag 'ONLY_2017_DATA' set to True only moves the training data from ISIC
2017 challenge. If it's set to False, it will also move the training data from
ISIC 2018 challenge dataset (only 'MM' and 'SK' classes though)

Same script can be applied to organise the test dataset images. It must be ensured
that the test folder looks as follows
    test
        dir             -- a subfolder containing ALL of the images (can be of any name)
        

"""
import os, glob, math
from shutil import copy2, move


#%% PARAMETERS AND DEFINITIONS
# resizing
RESIZE = False  # should the images be resized before moving?
original_path = 'C:\\ISIC_data\\ISIC-2017_Training_Data' # specify where the data is located
target_path = 'C:\\ISIC_data\\ISIC-2017_Training_Data\\224x224' # specify where the resized data should be placed
target_resolution = 224 # specify the target resolution, same for both axes

# datasets
ONLY_2017_DATA = False
# specify where the ground truth .csv file is located (ISIC 2017)
ground_truth_path_2017 = 'C:\\ISIC_data\\ISIC-2017_Training_Data\\Train_GroundTruth.csv' 
img_path_2017 = 'C:\\ISIC_data\\ISIC-2017_Training_Data\\224x224' # path to images to be copied
# specify where the ground truth .csv file is located (ISIC 2018). If ONLY_2017_DATA
# is set to True, you can comment the next two definitions out
ground_truth_path_2018 = 'C:\\ISIC_data\\ISIC-2018_Training_Data\\ISIC2018_Task3_Training_GroundTruth.csv'
img_path_2018 = 'C:\\ISIC_data\\ISIC-2018_Training_Data\\224x224' # path to images to be copied

# other options
disease = 'SK'  # specify which classifier is of interest. Either 'SK' or 'MM'
ratio = 0.7 # training set ratio w.r.t. total number of images. ratio = (no of images in the training set) / (total no of images)
train_dir = 'C:\\ISIC_data\\train__' + disease # target training set folder
valid_dir = 'C:\\ISIC_data\\valid__' + disease # target validation set folder 



def import_csv(csvPath):
    """
    Imports data from a .csv file
    Inputs
        csvPath     -- path to the .csv file
    Outputs
        data        -- data as a vector of lists (rows in the .csv file)
    """
    import csv
    
    data = [];
    with open(csvPath) as csvFile:
        csvReader = csv.reader(csvFile)    
        for row in csvReader:
            data.append(row)
    return data


def resize_images(original_path, target_path, target_resolution):
    """
    Resizes high resolution .jpg images to smaller
    Inputs
        original_path     -- path to the original data
        target_path       -- target path for the resized images
        target_resolution -- target resolution 
    """
    import cv2
    import os
    from glob import glob
    
    newImgHeight = target_resolution
    newImgWidth = target_resolution
    
    imgList = glob(os.path.join(original_path, '*.jpg'))    
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    os.chdir(target_path)
    for i in range(len(imgList)):
        imgPath = imgList[i]
        original = cv2.imread(imgPath)
        resized = cv2.resize(original, (newImgHeight, newImgWidth), interpolation = cv2.INTER_CUBIC);
        filename = os.path.basename(os.path.normpath(imgPath))
        cv2.imwrite(filename, resized)
        print("Processing %s/%s" % (i, len(imgList)))

##############################################################################
#%% Resize images 
if RESIZE:
    resize_images(original_path, target_path, target_resolution)


#%% Copy the 2017 ISIC challenge data to the target training and validation folders
# import ground truth data
gtData = import_csv(ground_truth_path_2017)
gtData = gtData[1:]

# copy
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

if disease == 'MM':
    idx = 1
elif disease == 'SK':
    idx = 2

move_dir_pos = os.path.join(train_dir, '1')
if not os.path.exists(move_dir_pos):
    os.makedirs(move_dir_pos)

move_dir_neg = os.path.join(train_dir, '0')
if not os.path.exists(move_dir_neg):
    os.makedirs(move_dir_neg)

for image in gtData:
    imgPath = os.path.join(img_path_2017, image[0] + '.jpg')
    print('Copying ' + image[0])
    if image[idx] == '1': # positive
        copy2(imgPath, move_dir_pos)
    else: 
        copy2(imgPath, move_dir_neg)
       
        
#%% Copy the 2018 ISIC challenge data to the target training and validation folders
if not ONLY_2017_DATA:
    gtData = import_csv(ground_truth_path_2018)
    gtData = gtData[1:]
    gtData = [[item[0], item[1], item[5]] for item in gtData]
    
    for image in gtData:
        imgPath = os.path.join(img_path_2018, image[0] + '.jpg')
        print('Copying ' + image[0])
        if image[idx] == '1.0': # positive
            copy2(imgPath, move_dir_pos)
        else: 
            pass
        
        
#%% Move random images to the validation set folder
import random

valid_move_dir_pos = os.path.join(valid_dir, '1')
if not os.path.exists(valid_move_dir_pos):
    os.makedirs(valid_move_dir_pos)

valid_move_dir_neg = os.path.join(valid_dir, '0')
if not os.path.exists(valid_move_dir_neg):
    os.makedirs(valid_move_dir_neg)

files_list = glob.glob(move_dir_pos+'\*.jpg')
mv_pos = math.floor(len(files_list) *(1-ratio))

for i in range(3):
    random.shuffle(files_list)

for i in range(1, mv_pos):
    print('Moving %s to validation set' % files_list[i])
    move(files_list[i], valid_move_dir_pos)
    
files_list = glob.glob(move_dir_neg+'\*.jpg')
mv_neg = math.floor(len(files_list) * (1-ratio))

for i in range(3):
    random.shuffle(files_list)

for i in range(1, mv_neg):
    print('Moving %s to validation set' % files_list[i])
    move(files_list[i], valid_move_dir_neg)





















