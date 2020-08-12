import logging
import sys
import os
from datetime import datetime

FilePath = os.getcwd()
Datetime = str(datetime.now())
Log = "output.log"
BackUpName = str(FilePath)+"/"+Datetime+"-"+Log
NewName = str(FilePath)+"/"+Log

os.rename(NewName,BackUpName)

a_logger = logging.getLogger()
a_logger.setLevel(logging.DEBUG)

output_file_handler = logging.FileHandler(str(Log))
stdout_handler = logging.StreamHandler(sys.stdout)

a_logger.addHandler(output_file_handler)
a_logger.addHandler(stdout_handler)

a_logger.debug("FilePath : "+str(FilePath))


"""
/*****************************************************************************************

1. Get data
Obtaining images and resizing to 70 x 70 px. We use a 70 x 70 px size for a 
quicker training of model.Also here we get image labels from folder name

********************************************************************************************/
"""
import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])
	
install('opencv-python')
import cv2
from glob import glob
install('numpy')
import numpy as np
install('matplotlib')
from matplotlib import pyplot as plt
import math
install('pandas')
import pandas as pd

ScaleTo = 70  # px to scale
seed = 7  # fixing random

path = '/home/ec2-user/Keras_Ex/NN/Assignment_CNN_Group/Data/Train/train/*/*.png' 
files = glob(path)

trainImg = []
trainLabel = []
j = 1
num = len(files)

# Obtain images and resizing, obtain labels
for img in files:
    print(str(j) + "/" + str(num), end="\r")
    trainImg.append(cv2.resize(cv2.imread(img), (ScaleTo, ScaleTo)))  # Get image (with resizing)
    trainLabel.append(img.split('/')[-2])  # Get image label (folder name)
    j += 1

trainImg = np.asarray(trainImg)  # Train images set
trainLabel = pd.DataFrame(trainLabel)  # Train labels set


"""
/*****************************************************************************************

 look at some examples of plant photos

********************************************************************************************/
"""


# Show some example images
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(trainImg[i])
	

"""	
/*************************************************************************************************************

 
2. Cleaning data
For removing the background, we'll use the fact, that all plants on our photos are green. 
We can create a mask, which will leave some range of green color and remove other part of image.

2.1. Masking green plant
For creating mask, which will remove background, we need to convert RGB image to HSV. 
HSV is alternative of the RGB color model. In HSV, it is easier to represent a color range than in RGB color space


Besides of this, we'll blur image firstly for removing noise. After creating HSV image, we'll create mask based on empirically selected range of green color, convert it to boolean mask and apply it to the origin image.

Use gaussian blur for remove noise
Convert color to HSV
Create mask
Create boolean mask
Apply boolean mask and getting image whithout background
****************************************************************************************************************/
"""


clearTrainImg = []
examples = []; getEx = True
for img in trainImg:
    # Use gaussian blur
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   
    
    # Convert to HSV image
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    
    # Create mask (parameters - green color range)
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Create bool mask
    bMask = mask > 0  
    
    # Apply the mask
    clear = np.zeros_like(img, np.uint8)  # Create empty image
    clear[bMask] = img[bMask]  # Apply boolean mask to the origin image
    
    clearTrainImg.append(clear)  # Append image without backgroung
    
    # Show examples
    if getEx:
        plt.subplot(2, 3, 1); plt.imshow(img)  # Show the original image
        plt.subplot(2, 3, 2); plt.imshow(blurImg)  # Blur image
        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  # HSV image
        plt.subplot(2, 3, 4); plt.imshow(mask)  # Mask
        plt.subplot(2, 3, 5); plt.imshow(bMask)  # Boolean mask
        plt.subplot(2, 3, 6); plt.imshow(clear)  # Image without background
        plt.savefig('Train_sample.png')  # Save image
        getEx = False

clearTrainImg = np.asarray(clearTrainImg)


"""	
/*************************************************************************************************************
 
Good result! Let's look at other examples of masked images

****************************************************************************************************************/
"""


# Show sample result
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(clearTrainImg[i])	


"""	
/*************************************************************************************************************	
2.2. Normalize input
Now set the values of input from [0...255] to [0...1] (RGB color-space encode colors with numbers [0...255]). CNN will be faster train if we use [0...1] input
****************************************************************************************************************/
"""


clearTrainImg = clearTrainImg / 255


"""
/***********************************************************************************************************************************		
2.3. Categories labels
Now we encode image labels. Labels are 12 string names, so we could create classes array with this names,
for example ['Black-grass' 'Charlock' 'Cleavers' 'Common Chickweed' 'Common wheat' 'Fat Hen' 'Loose Silky-bent' 'Maize' 
'Scentless Mayweed' 'Shepherds Purse' 'Small-flowered Cranesbill' 'Sugar beet'] and encode every label by position in this array. 
For example 'Charlock' -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].

We need to encode all labels in this way.

**************************************************************************************************************************************/
"""
install('tensorflow')
install('keras')
from keras.utils import np_utils
install('scikit-learn')
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Encode labels and create classes
le = preprocessing.LabelEncoder()
le.fit(trainLabel[0])
a_logger.debug("Classes: " + str(le.classes_))
encodeTrainLabels = le.transform(trainLabel[0])

# Make labels categorical
clearTrainLabel = np_utils.to_categorical(encodeTrainLabels)
num_clases = clearTrainLabel.shape[1]
a_logger.debug("Number of classes: " + str(num_clases))

# Plot of label types numbers
trainLabel[0].value_counts().plot(kind='bar')


"""
/***********************************************************************************************************************************	
3. Model
3.1. Split dataset
Split data on training and validation set. 10% of data became the validation set

Our data is unbalanced, so for avoide inaccurate evaluation of model set stratify=clearTrainLabel


**************************************************************************************************************************************/
"""


from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(clearTrainImg, clearTrainLabel, 
                                                test_size=0.1, random_state=seed, 
                                                stratify = clearTrainLabel)


"""
/***********************************************************************************************************************************
3.2. Data generator¶
To avoide overfitting we need to create image generator which will randomly rotate, zoom, shift and flip image during the fitting of the model.

Set random rotation from 0 to 180 degrees
Set random zoom at 0.1
Set random shifting at 0.1
Set horisontal and vertical flips

**************************************************************************************************************************************/
"""


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=180,  # randomly rotate images in the range
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically 
        horizontal_flip=True,  # randomly flip images horizontally
        vertical_flip=True  # randomly flip images vertically
    )  
datagen.fit(trainX)


"""
/***********************************************************************************************************************************
3.3. Create model
For creation model i used Keras Sequential.

I created model with six convolutional layers and three fully-connected layers in the end. First two convolutional layers have 64 filters, next 128 filters and the last two layers have 256 filters. After each pair of convolution layers model have max pooling layer. Also, to reduce overfitting after each pair of convolution layers we use dropout layer (10% between convolutional layers and 50% between fully connect layers) and between each layer we use batch normalization layer.

In the end i used three fully-connected layers for classifing. In the last layer the neural net outputs destribution of probability for each of 12 classes.
**************************************************************************************************************************************/
"""


import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD,Adam
from keras.applications import VGG19, VGG16, ResNet50

numpy.random.seed(seed)  # Fix seed

base_model_resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(ScaleTo, ScaleTo, 3), classes=clearTrainLabel.shape[1])


#Adding the final layers to the above base models where the actual classification is done in the dense layers
model_resnet50 = Sequential()
model_resnet50.add(base_model_resnet) 
model_resnet50.add(Flatten()) 
model_resnet50.add(Dense(1024,activation=('relu'),input_dim=512))
model_resnet50.add(Dense(512,activation=('relu'))) 
model_resnet50.add(Dropout(.4))
model_resnet50.add(Dense(256,activation=('relu'))) 
model_resnet50.add(Dropout(.3))
model_resnet50.add(Dense(128,activation=('relu')))
model_resnet50.add(Dropout(.2))
model_resnet50.add(Dense(num_clases,activation=('softmax')))


model_resnet50.summary()

#Defining the hyperparameters
#batch_size= 10
#epochs=35
learn_rate=.001
sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)

#Compiling the VGG19 model
model_resnet50.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])



"""
/***********************************************************************************************************************************
3.4. Fit model
Here we'll train our model. Firstly, we set several callbacks. First callback reduce model learning rate. With high learning rate model quiker is the convergance, however with high learning rate the model could fall into a local minimum. So, we will decreace the learning rate at the process of fitting. We will reduce learning rate if the accuracy is not improved after for three epoch. Other two callbacks save best and last weights of model.

We won't train model on kaggle kernel, becauce it is too long process, so i comment the lines of code with fitting.

**************************************************************************************************************************************/
"""


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

# learning rate reduction --val_accuracy val_acc , save_freq='epoch' F"
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.4, 
                                            min_lr=0.00001)

a_logger.debug("===== ModelCheckpoint Begins ====="+"")
# checkpoints
filepath="/home/ec2-user/Keras_Ex/NN/Assignment_CNN_Group/Saved_Models/Saved/ResNet50/weights.best_{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')

filepath="/home/ec2-user/Keras_Ex/NN/Assignment_CNN_Group/Saved_Models/Saved/ResNet50/weights.last_auto4.hdf5"
checkpoint_all = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1, save_best_only=False, mode='max')

# all callbacks
callbacks_list = [checkpoint, learning_rate_reduction, checkpoint_all]

a_logger.debug("===== trainX.shape[0] ====="+str(trainX.shape[0]))

# fit model
hist = model_resnet50.fit_generator(datagen.flow(trainX, trainY, batch_size=10), 
                            epochs=35, validation_data=(testX, testY), 
                            steps_per_epoch=trainX.shape[0]/10, callbacks=callbacks_list)
							
a_logger.debug("===== ModelCheckpoint Ends ====="+"")


"""
/***********************************************************************************************************************************

4. Evaluate model
4.1. Load model from file
Here we load the weights of best-fitting model from file (i used kaggle dataset with weights of model, which i trained earlier). 
Also i load from Data.npz training and validation data sets, on which model was fitting for evaluating of model accuracy.

**************************************************************************************************************************************/
"""


a_logger.debug("===== load_weights Begins ====="+"")

#Plotting the training and validation loss
f,ax=plt.subplots(2,1) #Creates 2 subplots under 1 column
#Training loss and validation loss
ax[0].plot(model_resnet50.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(model_resnet50.history.history['val_loss'],color='r',label='Validation Loss')
#Training accuracy and validation accuracy
ax[1].plot(model_resnet50.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(model_resnet50.history.history['val_accuracy'],color='r',label='Validation Accuracy')
plt.savefig('Training_and_Validation_Loss_plt.png')  # Save image

model_resnet50.load_weights("/home/ec2-user/Keras_Ex/NN/Assignment_CNN_Group/Saved_Models/Saved/ResNet50/weights.best_34-0.93.hdf5")

#data = np.load("/home/ec2-user/Keras_Ex/NN/Assignment_CNN_Group/Saved Models/Data.npz")
#d = dict(zip(("trainX","testX","trainY", "testY"), (data[k] for k in data)))

#rainX = d['trainX']
#testX = d['testX']
#trainY = d['trainY']
#testY = d['testY']

a_logger.debug(model_resnet50.evaluate(trainX, trainY))  # Evaluate on train set
a_logger.debug(model_resnet50.evaluate(testX, testY))  # Evaluate on test set

a_logger.debug("===== load_weights Ends ====="+"")


"""
/***********************************************************************************************************************************
4.2. Confusion matrix
A good way to look at model errors.
**************************************************************************************************************************************/
"""

a_logger.debug("===== plot_confusion_matrix Begins ====="+"")

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_matrix.png')

# Predict the values from the validation dataset
predY = model_resnet50.predict(testX)
predYClasses = np.argmax(predY, axis = 1) 
trueY = np.argmax(testY, axis = 1) 

# confusion matrix
confusionMTX = confusion_matrix(trueY, predYClasses) 

# plot the confusion matrix
plot_confusion_matrix(confusionMTX, classes = le.classes_)

a_logger.debug("===== plot_confusion_matrix Ends ====="+"")


"""
/***********************************************************************************************************************************
4.3. Get results
And finnaly we get the result of prediction on test data.
**************************************************************************************************************************************/
"""


a_logger.debug("===== Test Data Prediction Begins ====="+"")

path = '/home/ec2-user/Keras_Ex/NN/Assignment_CNN_Group/Data/Train/test/*.png'
files = glob(path)

testImg = []
testId = []
j = 1
num = len(files)

# Obtain images and resizing, obtain labels
for img in files:
    print("Obtain images: " + str(j) + "/" + str(num), end='\r')
    testId.append(img.split('/')[-1])  # Images id's
    testImg.append(cv2.resize(cv2.imread(img), (ScaleTo, ScaleTo)))
    j += 1

testImg = np.asarray(testImg)  # Test images set

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(testImg[i])


clearTestImg = []
examples = []; getEx = True
for img in testImg:
    # Use gaussian blur
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   
    
    # Convert to HSV image
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    
    # Create mask (parameters - green color range)
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Create bool mask
    bMask = mask > 0  
    
    # Apply the mask
    clear = np.zeros_like(img, np.uint8)  # Create empty image
    clear[bMask] = img[bMask]  # Apply boolean mask to the origin image
    
    clearTestImg.append(clear)  # Append image without backgroung
    
    # Show examples
    if getEx:
        plt.subplot(2, 3, 1); plt.imshow(img)  # Show the original image
        plt.subplot(2, 3, 2); plt.imshow(blurImg)  # Blur image
        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  # HSV image
        plt.subplot(2, 3, 4); plt.imshow(mask)  # Mask
        plt.subplot(2, 3, 5); plt.imshow(bMask)  # Boolean mask
        plt.subplot(2, 3, 6); plt.imshow(clear)  # Image without background
        plt.savefig('Test_sample.png')  # Save image
        getEx = False

clearTestImg = np.asarray(clearTestImg)


# Show sample result
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(clearTestImg[i])	


clearTestImg = clearTestImg / 255


a_logger.debug("===== model.predict Begins ====="+"")


def save_data(saving_path, file, df):
      if (df.to_csv(saving_path +'/'+ file + '.csv', index = False)):
          a_logger.debug("File "+str(file)+" Successfully Saved To : "+str(saving_path))#return True
      else:
          a_logger.debug("File "+str(file)+" Failed to Saved To : "+str(saving_path))#return False


pred = model_resnet50.predict(clearTestImg)

# Write result to file
predNum = np.argmax(pred, axis=1)
predStr = le.classes_[predNum]

res = {'file': testId, 'species': predStr}
res = pd.DataFrame(res)
a_logger.debug(res)
save_data(str(FilePath), "res", res)
#res.to_csv("/home/ec2-user/Keras_Ex/NN/Assignment_CNN_Group/Saved_Models/Saved/ResNet50/res.csv", index=False)

a_logger.debug("===== Test Data Prediction Ends ====="+"")