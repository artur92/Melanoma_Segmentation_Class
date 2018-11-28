# -*- coding: utf-8 -*-
from __future__ import print_function #This should allways be at the beginig of the files
"""
Created on Thu Apr 26 19:03:09 2018

@author: Arturo May
"""
'''
This file is used to organise all the image in on folder call train, and standardize the name of the files,
the mask will end with _mask jpg, also standardize the images into 420, 580 pixel to guarantee that all the images
will fit in the network and convert to gray scale. Requirements: 

SubFolder Data_To_Organise: 
    SubFolder call images: all the images in RGB 
    SubFolder call masks: all the images of groundtruth should have the same name that the images.


The next SubFolder should be inside a folder call raw.
    

SubFolder call train: Where all the images will be saved 
'''

import os
import sys
import numpy as np
import tensorflow as tf
import random
import warnings
import cv2
from tqdm import tqdm
import re

# Set some parameters

PATH = os.getcwd() # Get the path of the file
organise = PATH + '/Data_To_Organise' #Folder Path


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

tf.reset_default_graph() # This for avoid the kernel to stop when running the conv2
data_dir_list = os.listdir(organise + '/images/') #Images folder
mask_dir_list =  os.listdir(organise + '/masks/') #Mask folder
new_folder = PATH + '/raw/'+ '/train/' #output folder train
image_rows = 420  
image_cols = 580


sys.stdout.flush() # To work with tqdm

print("Loading images and masks")
i= 0

old_name = ''
i = 0
for n in tqdm(range(0,len(data_dir_list))):
    
        #Images
        img=cv2.imread(organise + '/images/' + data_dir_list[n] ) #img)              
        img_name =os.path.splitext(data_dir_list[n])[0] #Get only the name of the files without extension
        #Masks
        mask =cv2.imread(organise + '/masks/' + mask_dir_list[i] ) #img) 
        mask_name =os.path.splitext(mask_dir_list[i])[0] #Get only the name of the files without extension

       #The data from 2017 challenge had some images call superpixels that where messing with the results 
       #in the so need a method to aoid this files where s used this is why is created a nez folder call 
       #train to save all the images that we need to implement
 
        if(re.search('superpixels', data_dir_list[n])):            
           
           old_name = '' # Temporal fix should not  be  like this resize, convert grayscale and save 
          #For compare that the names match between the images and the masks
        elif(n<= len(mask_dir_list) and (img_name + '_segmentation') == mask_name):
             
            img = cv2.resize(img, (image_cols, image_rows))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(new_folder+img_name+'_segmentation.jpg',img)
              #Masks 
            mask = cv2.resize(mask, (image_cols, image_rows))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(new_folder+mask_name+'_mask.jpg',mask)
            i = i +1
         
            
           


