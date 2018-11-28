#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 02:53:12 2018

@author: apple
"""
#This file was created in case of the data augmentation was implemented in the finals test is not used
import Augmentor
p = Augmentor.Pipeline("/Users/apple/Downloads/ultrasound-nerve-segmentation-master-2/Data_To_Organise/images_new")
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
p.ground_truth("/Users/apple/Downloads/ultrasound-nerve-segmentation-master-2/Data_To_Organise/mask_new")
# Add operations to the pipeline as normal:
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.sample(10000)