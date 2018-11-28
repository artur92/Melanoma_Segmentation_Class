Instructions:

1. Install the following packages (Python 3.6 recommended): Tensorflow, Keras, OpenCV, numpy, 
matplotlib, sklearn, scikit-plot
2. Download the classification training data from the ISIC 2017 and/or ISIC 2018 challenge websites
3. Unzip the data to your working directory
4. The images should be resized to the same resolution and assigned to training and validation sets. Open 
'preprocessing_move.py' file, define the parameters and run it.
5. Run either 'resnet18_classifier.py' or 'resnet50_classifier.py' to train the neural network. Do not forget to 
set the parameters right within the files. 
6. Open 'evaluation.py', define the parameters and run to see the resulting ROC curve and confusion matrix.
