import math, os
import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from IPython.display import display 


#%% PARAMETERS
DATA_DIR = 'C:\\ISIC_data' # main working folder 
TRAIN_DIR = os.path.join(DATA_DIR, 'train_MM') # folder containing training data 
VALID_DIR = os.path.join(DATA_DIR, 'valid_MM') # folder containing validation data
TB_LOG_DIR = 'C:\\ISIC_data\\logs' # folder where tensorboard logs should be saved
BATCH_SIZE = 16
SIZE = (224, 224)
EPOCHS = 30

MODEL_NAME = 'resnet50_MM_30.h5' # the name under which the network weights should be 
                                 # saved as a .h5 file after training is finished

#%%
early_stopping = EarlyStopping(patience=10)
tensorboard = TensorBoard(log_dir=TB_LOG_DIR, \
                          histogram_freq=0, batch_size=16, \
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

model = keras.applications.resnet50.ResNet50(weights='imagenet')

classes = list(iter(batches.class_indices))
model.layers.pop()
for layer in model.layers:
    layer.trainable=False
last = model.layers[-1].output
x = Dense(len(classes), activation="softmax")(last)
finetuned_model = Model(model.input, x)
finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
for c in batches.class_indices:
    classes[batches.class_indices[c]] = c
finetuned_model.classes = classes

finetuned_model.fit_generator(batches, 
                              steps_per_epoch=num_train_steps, 
                              epochs=EPOCHS, 
                              callbacks=[early_stopping, tensorboard], 
                              validation_data=val_batches, 
                              validation_steps=num_valid_steps)
finetuned_model.save(MODEL_NAME)
    
