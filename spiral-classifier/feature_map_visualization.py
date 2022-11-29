#%%
# *********************************
#            IMPORT LIBRARIES
# *********************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

import cv2
import PIL

# import ML/DL libraries
from sklearn.model_selection import train_test_split
from sklearn import utils # used to shuffle data

from keras.preprocessing.image import ImageDataGenerator # used for image augmentation
from tensorflow.keras.applications.inception_v3 import preprocess_input


import tensorflow as tf
# used for building and training a new model
from keras import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications import VGG16, ResNet50

# import functions from other python files
from code_files.imagePreprocessing import * 

# %%

# ********************************
#           IMPORT IMAGES
# ********************************
# import images (and labels) and store in dataframe
# FLAG: set the path to the desired dataset
data_path = 'datasets/handPD_orig_HT/'  

trainImgs = pd.DataFrame()
testImgs  = pd.DataFrame()
trainArray = []
testArray  = []


for dataType in os.listdir(data_path):
    img_path = []
    lbl = []
    lblName = []
    for group in os.listdir(os.path.join(data_path, dataType)):
        for img in os.listdir(os.path.join(data_path, dataType, group)):
            path = os.path.join(data_path, dataType, group, img)
            img_path.append(path) 

            # convert the image and store as a matrix
            drawing = cv2.imread(path)
            drawing = cv2.resize(drawing, (256,256))

            if dataType == 'test':
                testArray.append(drawing)
            else:
                trainArray.append(drawing)

            # store the labels
            if group == 'healthy':
                lbl.append(0)
                lblName.append('healthy')
            else:
                lbl.append(1)
                lblName.append('parkinsons')

    if dataType == 'train':
        trainLbls = lbl
        trainImgs['image'] = img_path
        trainImgs['label'] = lblName
    else:
        testLbls = lbl
        testImgs['image'] = img_path
        testImgs['label'] = lblName

# shuffle the data
trainImgs, trainArray, trainLbls = utils.shuffle(trainImgs, trainArray, trainLbls)
testImgs, testArray, testLbls = utils.shuffle(testImgs, testArray, testLbls)

# convert labels to categorical for training model
trainLbls_categorical = tf.keras.utils.to_categorical(trainLbls)
print("Labels of first 5 images: \n", trainLbls_categorical[0:5])

# display first five images
print("Lables of first train 5 images: ", trainLbls[0:5])
display(trainImgs.head())

print("Test labels: ", testLbls[0:5])
display(testImgs.head())

# %%

# ******************************
#     VISUALIZE FEATURE MAPS
# ******************************
# import VGG16 or ResNet-50 pretrained model 
model = VGG16(weights='imagenet', include_top=False,input_shape=(256,256,3)) # setting include_top=False removes the fully connected layers of the model
# model.summary()

# summarize feature map shapes
for i in range(len(model.layers)):
    layer=model.layers[i]
    # check for conv layer
    if 'conv' in layer.name:
        print(i, layer.name, layer.output.shape)

# %%

# choose second conv block from each layer to display
blocks = [2,5,9,13,17]
output_layers = [model.layers[i].output for i in blocks]
# redefine model to output right after each conv layer
vis_model = Model(inputs=model.inputs, outputs=output_layers)

# select images to save visualizations for and put them in an array
# manually choose two healthy and two parkinsons
img2vis = np.array([trainArray[0], trainArray[1], trainArray[4], trainArray[6]])

# iterate through each image and save the feature maps
for i in range(4):
    img = np.expand_dims(img2vis[i], axis=0)
    img = preprocess_input(img)
    feature_maps = vis_model.predict(img)
    blocknum = 0

    for fmap in feature_maps:
        fmap_size = np.shape(fmap)
        # determine the number of images to plot
        if fmap_size[3]==64:
            rows=8;cols=8
        elif fmap_size[3]==128:
            rows=16; cols=8
        elif fmap_size[3]==256:
            rows=16;cols=16
        else:
            rows=32;cols=16

        itr = 1
        fig, ax = plt.subplots(rows,cols, figsize=(50,50))
        title = 'Image_' + str(i) + 'fmap: ' + str(np.shape(fmap))
        fig.suptitle(title)

        for r in range(rows):
            for c in range(cols):
                ax[r][c].imshow(fmap[0,:,:,itr-1], cmap='gray')
                itr += 1

        savename = 'Image_' + str(i) + '__block_' +  str(blocks[blocknum]) + '.png'
        print(savename)
        fig.savefig(savename)
        plt.close()
        blocknum += 1



# %%
