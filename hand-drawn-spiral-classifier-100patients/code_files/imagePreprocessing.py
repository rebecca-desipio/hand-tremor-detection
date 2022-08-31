# this file contains functions to preprocess the images and augment as needed to 
# feed into the model for training

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # statistical data vis, used to plot the total count for each label
import os 
import warnings

warnings.filterwarnings('ignore')

# libraries used to split the data and generate augmented images
from sklearn.model_selection import train_test_split
from sklearn import utils

# image conversion and processing libraries
import cv2
import PIL
from PIL import Image


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

    

