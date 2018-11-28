# -*- coding: utf-8 -*-
"""
Evaluation script. Defines the model of interest (either resnet18 or resnet50),
loads trained weights and evaluates shows the performance in terms of ROC curves. 
"""
import os
import keras
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt

# move to the directory where this file is located, so that 'resnet18' model could
# be loaded successfully
os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%% PARAMETERS
model_to_evaluate = 'resnet18'  # model to evaluate (either 'resnet18' or 'resnet50')
classifier = 'SK' # classifier of interest. Either 'SK' or 'MM'
weights_to_evaluate = 'C:\\ISIC_data\\models\\resnet18_SK_3.h5'# location of the model weights to be loaded
TEST_DIR = 'C:\\ISIC_data\\test\\'  # directory of the test data
# location of the test dataset ground truth 
ground_truth_path = 'C:\\ISIC_data\\ISIC-2017_Test_Data\\Test_GroundTruth.csv' 
SIZE = (224, 224)    
BATCH_SIZE = 16
classes = ['0', '1']


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

def define_model(model_name):
    """
    Defines the model of interest for evaluation
    Input
        model_name -- either 'resnet18' or 'resnet50'
    """
    import resnet18
    from keras.layers import Dense
    from keras.models import Model
    from keras.optimizers import Adam
        
    if model_name == 'resnet18':
        nb_classes = 2
        img_rows, img_cols = 224, 224
        img_channels = 3
        model = resnet18.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        print('Resnet18 built')
        return model
        
    elif model_name == 'resnet50':
        classes = ['0', '1']
        model = keras.applications.resnet50.ResNet50(weights='imagenet')
        model.layers.pop()
        for layer in model.layers:
            layer.trainable=False
        last = model.layers[-1].output
        x = Dense(len(classes), activation="softmax")(last)
        finetuned_model = Model(model.input, x)
        finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        finetuned_model.classes = classes
        print('Resnet50 built')
        return finetuned_model
    else:
        print('The model_name input must be either \'resnet18\' or \'resnet50\'')
        return 
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    

#%% Build the model for evaluation
print('The model is being defined...')
model = define_model(model_to_evaluate)

#%% Print available model weights
#for item in glob.glob("C:\\ISIC_data\\models\\*.h5"):
#    print(item)

#%% load the trained model weights
model.load_weights(weights_to_evaluate)

# feed the test set and obtain the predictions
print('The neural network is obtaining predictions...')
test_gen = image.ImageDataGenerator()
test_batches = test_gen.flow_from_directory(TEST_DIR, target_size=SIZE, shuffle=False, class_mode=None, batch_size=BATCH_SIZE)
y_pred = model.predict_generator(test_batches)


#%% obtain ground truth data
if classifier == 'MM':
    index = 1
    confindex = 0
else:
    index = 2
    confindex = 1
    
gtData = import_csv(ground_truth_path)
gtData = gtData[1:]
gtData = [np.float32(item[index]) for item in gtData]

y_true = np.zeros((len(gtData), 2))
y_true[:,1] = gtData
for index, value in enumerate(gtData):
    if value == 1.0:
        y_true[index, 0] = 0
    else:
        y_true[index, 0] = 1

# get labels and classes
y_pred_labels = np.argmax(y_pred, 1)
y_true_labels = np.argmax(y_true, 1)

y_pred_classes = []
for item in y_pred_labels:
    if item == confindex:
        y_pred_classes.append('Others')
    else:
        y_pred_classes.append(classifier)

y_true_classes = []
for item in y_true_labels:
    if item == confindex:
        y_true_classes.append('Others')
    else:
        y_true_classes.append(classifier)     


#%% Compute the metrics
# Confusion matrix
conf_mat = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure()
plot_confusion_matrix(conf_mat, classes=[classifier, 'Others'], 
                      title='Confusion matrix')

# Average precision score
avg_prec = average_precision_score(y_true, y_pred)

# Compute micro-average ROC curve and ROC area
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plot 
plt.figure()
lw = 2
# change '1' to '0' in the next line to swap between classes
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


#%% Plot ROC curves of all of the classes, as well as micro- and macro-averaged ones
skplt.metrics.plot_roc_curve(y_true_labels, y_pred)
plt.show()






























    
    