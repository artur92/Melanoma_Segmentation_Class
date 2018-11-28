# -*- coding: utf-8 -*-
"""
ResNet-18 classifier

"""
from __future__ import print_function
import os, math
import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing import image

# move to the directory where this file is located, so that 'resnet18' model could
# be loaded successfully
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import resnet18

#%% PARAMETERS
DATA_DIR = 'C:\\ISIC_data' # main working folder 
TRAIN_DIR = os.path.join(DATA_DIR, 'train_MM') # folder containing training data 
VALID_DIR = os.path.join(DATA_DIR, 'valid_MM') # folder containing validation data
TB_LOG_DIR = 'C:\\ISIC_data\\logs' # folder where tensorboard logs should be saved
BATCH_SIZE = 16
SIZE = (224, 224)
EPOCHS = 3
CLASSES = 2
MODEL_NAME = 'resnet18_MM_3.h5' # the name under which the network weights should be 
                                # saved as a .h5 file after training is finished

# input image dimensions
img_rows, img_cols = SIZE[0], SIZE[1]
img_channels = 3

#%%
early_stopper = EarlyStopping(patience=10)
tensorboard = TensorBoard(log_dir=TB_LOG_DIR, \
                          histogram_freq=0, batch_size=BATCH_SIZE, \
                          write_graph=True, write_grads=False, write_images=False, \
                          embeddings_freq=0, embeddings_layer_names=None, \
                          embeddings_metadata=None)


num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

gen = keras.preprocessing.image.ImageDataGenerator()
val_gen = keras.preprocessing.image.ImageDataGenerator()

batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

model = resnet18.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), CLASSES)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(batches,
                    steps_per_epoch=num_train_steps, 
                    epochs=EPOCHS, 
                    callbacks=[early_stopper, tensorboard],
                    validation_data=val_batches, 
                    validation_steps=num_valid_steps)
model.save(MODEL_NAME)
