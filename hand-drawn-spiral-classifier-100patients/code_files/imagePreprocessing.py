# this file contains functions to preprocess the images and augment as needed to 
# feed into the model for training

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # statistical data vis, used to plot the total count for each label
import os 
import warnings

# libraries used to split the data and generate augmented images
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn import utils

import tensorflow as tf

# image conversion and processing libraries
import cv2
import PIL
from PIL import Image

warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# function to import the images and create a dataframe
def importImages(dataset_folder):
    img_path = [] # store image paths for all images (original image size: 256x256)
    label = []    # healthy (0) vs parkinsons (1)

    # iterate through all images and create a binary array corresponding to the image labels
    for labeled_folder in os.listdir(dataset_folder):
        for img in os.listdir(dataset_folder + "/" + labeled_folder):
            if labeled_folder == 'healthy':
                label.append(0)
            else:
                label.append(1)
            img_path.append(os.path.join(dataset_folder, labeled_folder, img))

    # total number of images and labels should match
    print("total number of labels: ", len(label))
    print("total number of images: ", len(img_path))

    # create the dataframe
    df = pd.DataFrame()
    df['images'] = img_path
    df['label']  = label

    df = df.sample(frac=1).reset_index(drop=True) # randomize images
    df = utils.shuffle(df)

    return df
# --------------------------------------------------------------------------------------------------

# function to split data into train, validation, and test
def splitData(df):
    # randomly split data in train and validation subsets (70-30 split)
    # stratify attempts to keep the labels 50-50 in the validation data (i.e. 7 total 0's and 8 total 1's)
    train_feature, val_feature, train_label, val_label = train_test_split(df['images'], df['label'], test_size=0.30, stratify=df['label'])

    # shuffle data
    train_feature, train_label = utils.shuffle(train_feature, train_label)
    val_feature, val_label = utils.shuffle(val_feature, val_label)

    ## (OPTIONAL) split validation data into validation and testing data
    val_feature, test_feature, val_label, test_label = train_test_split(val_feature, val_label, test_size=0.5, shuffle=False)

    # sort the test array so that all healthy images are first and PD images are last
    # this is useful for later when plotting
    testDF = pd.DataFrame()
    testDF['img'] = test_feature
    testDF['lbl'] = test_label
    testDF = testDF.sort_values('lbl')

    test_feature = testDF['img']
    test_label   = testDF['lbl']

    print("total validation samples: ", len(val_label))
    print("total testing samples: ", len(test_label))
    print('total training samples: ', len(train_label))

    return train_feature, train_label, val_feature, val_label, test_feature, test_label
# --------------------------------------------------------------------------------------------------

# function to imgAug preprocessing, need to get data in desired form
def imgAug_preprocessing(train_feature, val_feature, test_feature):
    # for each of the data sets (train, val, and test), convert from rgb to grayscale
    # then convert to an array of pixels [0-255] --> this returns an array of size (256,256,1)
    # resize to (128x128) for faster image processing and a more efficient model, data doesn't get lost in this resize
    # need to account for the batch dimension (used in tensorflow), so expand dim to shape (1,128,128,1)
    def img2array(dataset):
        storage_array = []
        for img_path in dataset:
            openImg = PIL.Image.open(img_path)
            image = openImg.convert("L") # covert to grayscale (L), color use (P)
            imgArray = np.array(image)
            imgArray = cv2.resize(imgArray, (128,128))
            imgArray = np.expand_dims(imgArray, axis=2) # if keeping rgb, use axis=0

            # store in array
            storage_array.append(imgArray)
        
        return storage_array

    train_array = img2array(train_feature)
    val_array   = img2array(val_feature)
    test_array  = img2array(test_feature)

    return train_array, val_array, test_array
# --------------------------------------------------------------------------------------------------

# function to perform data augmentation to have more data
def imgAug(train_array, train_label, val_array, val_label, test_array, test_label):
    ## -------------------------------------------------------------------
    #       Artificially create more images for a bigger dataset
    ## -------------------------------------------------------------------
    # define functions to generate batches of data containing augmented images
    # use for training data only
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=180,
        fill_mode='nearest',
        brightness_range=[.4,1.4],
        vertical_flip = True,
        horizontal_flip = True
    )

    # use for validation and testing data (OPTIONAL: can make this different than training)
    test_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=180,
        fill_mode='nearest',
        brightness_range=[.4,1.4],
        vertical_flip = True,
        horizontal_flip = True
    )

    ## define function to artificially add more training, validation, and testing images
    # takes as inputs the dataset array, dataset labels, and the number of additional augmented images per each original image
    def generateAdditionalData(dataset_array, dataset_label, numImgs):
        dataAug = []
        dataAugLabel = []

        # iterate through each image in the data_array and create more images with the features specified by ImageDataGenerator
        for (idx,Lbl) in enumerate(dataset_label):
            tempImg = np.expand_dims(dataset_array[idx], axis=0) # use for grayscale
            # tempImg = train_array[idx]                         # use for rgb
            aug = train_gen.flow(tempImg, batch_size=1, shuffle=True)
            for addImages in range(numImgs):
                augImg = next(aug)[0] #.astype('uint8')
                if np.size(augImg) == 128**2:
                    dataAug.append(augImg)
                    dataAugLabel.append(Lbl)

        return dataAug, dataAugLabel

    trainAug, trainAugLabel = generateAdditionalData(train_array, train_label, 90)
    valAug, valAugLabel     = generateAdditionalData(val_array, val_label, 90)
    testAug, testAugLabel   = generateAdditionalData(test_array, test_label, 90)

    # covert label array to binary class matrix (healthy, PD)
    trainAugLabel = tf.keras.utils.to_categorical(np.array(trainAugLabel))
    valAugLabel = tf.keras.utils.to_categorical(np.array(valAugLabel))
    testAugLabel = tf.keras.utils.to_categorical(np.array(testAugLabel))

    # shuffle data one last time
    trainAug, trainAugLabel = utils.shuffle(trainAug, trainAugLabel)
    valAug, valAugLabel = utils.shuffle(valAug, valAugLabel)
    testAug, testAugLabel = utils.shuffle(testAug, testAugLabel)

    return trainAug, trainAugLabel, valAug, valAugLabel, testAug, testAugLabel
# --------------------------------------------------------------------------------------------------

# define a function to plot augmented images
def plotAugImgs(dataAug, dataAugLabel, dataTitle):
    count = 0 # counter to iterate through labels
    fig, axes = plt.subplots(1, 5, figsize=(12,2.5)) # create the figure window
    axes = axes.flatten()
    for img, ax in zip(dataAug, axes):
        ax.imshow(np.squeeze(img), cmap="gray") # plot image
        ax.set_title('Label: ' + str(dataAugLabel[count])) # display the associated label

        count = count + 1
        
    fig.suptitle(dataTitle)
    plt.tight_layout()
    plt.show()
# --------------------------------------------------------------------------------------------------

# define a function to plot the total number of each label for each dataset
def plotAugLabels(trainAugLabel, valAugLabel, testAugLabel):
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(18,4)
    ax0 = sns.countplot(trainAugLabel[:,1], ax=ax[0]); ax0.title.set_text("training data")
    ax1 = sns.countplot(valAugLabel[:,1], ax=ax[1]); ax1.title.set_text("validation data")
    ax2 = sns.countplot(testAugLabel[:,1], ax=ax[2]); ax2.title.set_text("test data")
    plt.show()

    print("Total training data samples: ", len(trainAugLabel))
    print("Total validation data samples: ", len(valAugLabel))
    print("Total test data samples: ", len(testAugLabel))
    print("Training-to-validation ratio: ", np.round(len(valAugLabel)/len(trainAugLabel),2)*100 , "%")

    

