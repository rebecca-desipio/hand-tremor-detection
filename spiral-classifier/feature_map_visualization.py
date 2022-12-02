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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import utils, svm, metrics # used to shuffle data

from keras.preprocessing.image import ImageDataGenerator # used for image augmentation
from tensorflow.keras.applications.inception_v3 import preprocess_input


import tensorflow as tf
# used for building and training a new model
from keras import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications import VGG16, ResNet50

# import functions from other python files
from code_files.imagePreprocessing import * 

print('done importing libraries')

# %%
print('importing images...')
# ********************************
#           IMPORT IMAGES
# ********************************
# import images (and labels) and store in dataframe
# FLAG: set the path to the desired dataset
data_path = 'datasets/handPD_orig_HT/'  

trainImgs = pd.DataFrame()
trainArray = []

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
            drawing = cv2.resize(drawing, (224,224))

            if dataType == 'train':
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


# shuffle the data
# trainImgs, trainArray, trainLbls = utils.shuffle(trainImgs, trainArray, trainLbls)

# convert labels to categorical for training model
trainLbls_categorical = tf.keras.utils.to_categorical(trainLbls)
# print("Labels of first 5 images: \n", trainLbls_categorical[0:5])

# used in the feature extraction section
numImgs = len(trainLbls)
print("Total number of images: ", numImgs)

sns.countplot(trainLbls)

# %%

# ******************************
#         IMPORT MODEL
# ******************************
# import VGG16 or ResNet-50 pretrained model 
model = ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3)) # setting include_top=False removes the fully connected layers of the model
model.summary()

# summarize feature map shapes
for i in range(len(model.layers)):
    layer=model.layers[i]
    # check for conv layer
    if 'conv' in layer.name:
        print(i, layer.name, layer.output.shape)

# %%

# ******************************
#     VISUALIZE FEATURE MAPS
# ******************************
# choose second conv block from each layer to display
blocks = [10,28,50,80, 112, 132, 174]
output_layers = [model.layers[i].output for i in blocks]
# redefine model to output right after each conv layer
vis_model = Model(inputs=model.inputs, outputs=output_layers)

# select images to save visualizations for and put them in an array
# manually choose two healthy and two parkinsons
img2vis = np.array([trainArray[0], trainArray[8], trainArray[72], trainArray[88]])

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
        # elif fmap_size[3]==128:
        #     rows=16; cols=8
        # elif fmap_size[3]==256:
        #     rows=16;cols=16
        else:
            rows=8;cols=8

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

# ***************************
#       OBTAIN FILTERS
# ***************************
# retrieve weights from hidden layer
filters, biases = model.layers[10].get_weights() # set the layer to visualize
# normalize filter values to 0-1 to visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first 8 filters
n = 16
fig, ax = plt.subplots(3,n, figsize=(50,10))
title = 'ResNet50_filters_block_10'
fig.suptitle(title)
for i in range(n):
    # get the filter
    f = filters[:,:,:,i]
    # plot each channel seperately
    for j in range(3):
        ax[j][i].imshow(f[:,:,j],cmap='gray')

savename = title + '.png'
print(savename)
fig.savefig(savename)
plt.close()

# %%

# ******************************************************************************************************
# ------------------------------------------------------------------------------------------------------
#                                           CLASSIFICATION  
# ------------------------------------------------------------------------------------------------------
# ******************************************************************************************************

# 1. VGG16 or RestNet50
# 2. SVM
# 3. Naive Bayes
# 4. Random Forest

# .........................
#    FEATURE EXTRACTION
# .........................
# define a function that will extract the features from conv network
def extract_features(imgs, num_imgs):
    datagen = ImageDataGenerator(rescale=1./255) # define to rescale pixels in image
    batch_size = 32
    
    features = np.zeros(shape=(num_imgs, 7,7,2048)) # shape equal to output of convolutional base
    lbls = np.zeros(shape=(num_imgs,2))

    # preprocess data
    generator = datagen.flow_from_dataframe(imgs, x_col = 'image', y_col='label', target_size=(224,224), class_mode='categorical', batch_size=batch_size)

    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        lbls[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= num_imgs:
            break
    return features, lbls

# extract features for both the trainImgs and testImgs
train_feat, train_lbls = extract_features(trainImgs, numImgs)

# %%

# =============================
#       VGG16 or ResNet50
# =============================
# split into training and testing data
train_feat, test_feat, train_lbls, test_lbls = train_test_split(train_feat, train_lbls, test_size=0.2, random_state=42)
trainArray, testArray, _,_ = train_test_split(trainArray, trainLbls, test_size=0.2, random_state=42)

fig, ax = plt.subplots(1,2, figsize=(8,3))
sns.countplot(train_lbls[:,1], ax=ax[0])
sns.countplot(test_lbls[:,1], ax=ax[1])

# evaluate on VGG16 classifier (using cross validation)
# define a function that will fit the model
def defineModel(size): # size is the dimension of the last layer in the pretrained model
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=(size,size,2048)))
    # global average pooling is used instead of fully connected layers on top of the feature maps
    # it takes the average of each feature map and the resulting layer is fed directly into the softmax layer
    model.add(Dense(2, activation='softmax'))
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)  # use the Adam optimizer and set an effective learning rate 
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# train the model using cross validation
# will start with k-fold cross validation, taking 80% as training each fold

# define model checkpoint callback
model_chkpt = tf.keras.callbacks.ModelCheckpoint('20221130_rn50_kfold_handPD_bal_HT_bs32_rs42.h5', verbose=0, save_best_only=True)

def fit_and_evaluate(train_feat, train_lbls, val_feat, val_lbls, epochs):
    model = None
    model = defineModel(8) # FLAG: need to set the size based on the last layer
    trained_model = model.fit(train_feat, train_lbls, batch_size=32, epochs=epochs, validation_data=(val_feat, val_lbls), callbacks=model_chkpt, verbose=0)

    # testScore = model.evaluate(test_feat, test_lbls)
    return trained_model

# train with k-fold validation
model_history = []
epochs = 250

num_val_samples = int(np.ceil(len(trainArray) * 0.20))
k = int(np.floor(len(trainArray) / num_val_samples))

for i in range(k):
    print("Training on fold K = ", i+1)
    startPt = i * num_val_samples
    endPt   = (i+1) * num_val_samples

    if endPt > len(train_feat):
        endPt = len(train_feat)

    val_x = train_feat[startPt:endPt]
    val_y = train_lbls[startPt:endPt]
    train_x = np.delete(train_feat, np.linspace(startPt, endPt-1, num_val_samples).astype(np.int), axis=0)
    train_y = np.delete(train_lbls, np.linspace(startPt, endPt-1, num_val_samples).astype(np.int), axis=0)

    model_history.append(fit_and_evaluate(train_x, train_y, val_x, val_y, epochs=epochs))
    # print(model_history)
    
    print("======="*12, end="\n")

# %%

# ...............
# VIEW RESULTS
# ...............   

new_model = True # FLAG

if new_model==True:
    # plot the accuracy and loss functions for each fold
    color = ['blue', 'black', 'red', 'green','orange', 'cyan', 'grey', 'yellow', 'fuchsia']
    f, ax = plt.subplots(2, k, figsize=(35,6))
    for i in range(k):
        ax[0][i].plot(model_history[i].history['accuracy'], label='train acc', color=color[i])
        ax[0][i].plot(model_history[i].history['val_accuracy'], label='val acc', linestyle= ':', color=color[i])
        ax[0][i].axis([-10,epochs, .2, 1.1])
        ax[0][i].legend()

        subplot_title = 'k = ' + str(i+1)
        ax[0][i].title.set_text(subplot_title)

    for i in range(k):
        ax[1][i].plot(model_history[i].history['loss'], label='train loss', color=color[i])
        ax[1][i].plot(model_history[i].history['val_loss'], label='val loss', linestyle= ':', color=color[i])
        ax[1][i].axis([-10,epochs, .0, 1.1])
        ax[1][i].legend()

# ---------------------------------
#   LOAD PRE-EXISTING MODEL MODEL
# ---------------------------------
def importModel(filename, testAug, testAugLabel):
    modelPath = 'savedModels/saved_h5_models/' + filename
    testModel = tf.keras.models.load_model(modelPath)

    loss, acc = testModel.evaluate(np.array(testAug), testAugLabel, verbose=2)
    print("Loss: ", loss, "| Accuracy: ", acc)

    return testModel

# load existing model and evaluate the test data
testmodel = importModel('20221130_rn50_kfold_handPD_bal_HT_bs32_rs42.h5', test_feat, test_lbls)

# %%

# SHOW THE MISCLASSIFIED IMAGES

def plotMisclassImgs(testModel, test_feat, test_label, test_array):
    test_label = np.array(test_label)
    incorrectImgs = []
    incorrectImgIdx = []

    count = 0
    fig, axes = plt.subplots(3, 17, figsize=(40,10))
    axes = axes.flatten()
    for img, ax in zip(test_array, axes):
        ax.imshow(np.squeeze(img), cmap="gray") # plot image

        # use the model to predict the label
        predImg = testModel.predict(np.expand_dims(test_feat[count], axis=0), verbose=0) # use for grayscale
        # predImg = testModel.predict(test_feat[count])                       # use for RGB
        predLabel = np.argmax(predImg[0])       
        
        if test_label[count] != predLabel:
            ax.set_title('Label: ' + str(test_label[count]) + ' | Pred: ' + str(predLabel), color='red')
            # save off image to array
            incorrectImgs.append(test_array[count])
            incorrectImgIdx.append(count)
        else:
            ax.set_title('Label: ' + str(test_label[count]) + ' | Pred: ' + str(predLabel), color = 'blue')  

        count = count + 1
        
    plt.tight_layout()

    return np.array(incorrectImgs), np.array(incorrectImgIdx), testModel.predict(test_feat)

# plot the results
misClass_test, misClass_idx, predictions = plotMisclassImgs(testmodel, test_feat, np.argmax(test_lbls, axis=1), testArray)

# %%
# =============================
#             SVM
# =============================

# create SVM classifier
def SVM_classifier(train_data, train_labels, val_data, val_labels):

    # clf = svm.SVC(kernel='poly', degree=5)
    # param_grid={'C':[0.1,5,10,100],'degree':[2,5,7,10],'kernel':['rbf','poly']}
    clf = svm.SVC(kernel='rbf', C=.1)
    # clf = GridSearchCV(clf, param_grid)

    # train model
    # clf.fit(train_data, train_labels)
    clf.fit(train_data, train_labels)

    # clf.best_params_
    # print(clf.best_params_)

    # predict the model
    pred = (clf.predict(val_data))

    # calculate accuracy
    acc = round(metrics.accuracy_score(pred, val_labels),4)

    print("The predicted data is: ", pred)
    print("The actual data is: ", np.array(val_labels))
    print(f"The model is {acc*100}% accurate")

    return acc, pred

# svm_acc = []

# for ii in range(0,10):
#     temp_acc, svm_pred = SVM_classifier(train_feat_flat, train_lbls)
#     svm_acc.append(temp_acc)

# print("Average accurage: ", np.mean(svm_acc))

# %%

# train_feat, test_feat, train_lbls, test_lbls = train_test_split(train_feat, train_lbls, test_size=0.2, random_state=42)
# trainArray, testArray, _,_ = train_test_split(trainArray, trainLbls, test_size=0.2, random_state=42)

# fig, ax = plt.subplots(1,2, figsize=(8,3))
# sns.countplot(train_lbls[:,1], ax=ax[0])
# sns.countplot(test_lbls[:,1], ax=ax[1])

# preprocess feature maps
train_feat_flat = []
for i in range(len(train_feat)):
    train_feat_flat.append(train_feat[i].flatten())

train_feat_flat, train_lbls = utils.shuffle(train_feat_flat, trainLbls, random_state=42)

num_val_samples = int(np.ceil(len(trainArray) * 0.20))
k = int(np.floor(len(trainArray) / num_val_samples))

for i in range(k):
    print("Training on fold K = ", i+1)
    startPt = i * num_val_samples
    endPt   = (i+1) * num_val_samples

    if endPt > len(train_feat_flat):
        endPt = len(train_feat_flat)

    val_x = np.array(train_feat_flat[startPt:endPt])
    val_y = train_lbls[startPt:endPt]
    train_x = np.delete(train_feat_flat, np.linspace(startPt, endPt-1, num_val_samples).astype(np.int), axis=0)
    train_y = np.delete(train_lbls, np.linspace(startPt, endPt-1, num_val_samples).astype(np.int), axis=0)

    acc, pred = SVM_classifier(train_x, train_y, val_x, val_y)

    print("======="*12, end="\n")
# %%
