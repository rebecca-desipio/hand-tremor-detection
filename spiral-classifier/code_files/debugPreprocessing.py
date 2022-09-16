# this file contains functions to preprocess the images and augment as needed to 
# feed into the model for training

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # statistical data vis, used to plot the total count for each label
import os 
import warnings
from sklearn import utils

# image conversion and processing libraries
import cv2
import PIL
from PIL import Image

warnings.filterwarnings('ignore')

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
    #df = utils.shuffle(df)

    return df

# function to split data into train, validation, and test
def splitData(df):
    data_array, labels = [df['images'], df['label']]

    imgData = []
    imgLbls = []
    itr=0
    for i in range(len(data_array)):
        image = PIL.Image.open(data_array[i])
        image = image.convert("L")
        image_array = np.array(image)

        # check length of array to make sure size 256x256, if not, add white pixels onto the end
        img_size = np.size(image_array)
        if img_size == 256**2:
            imgData.append(image_array.flatten().astype(float))  
            imgLbls.append(labels[i])

        itr = itr+1

    num_trn = 78
    num_test = 10

    imgData, imgLbls = utils.shuffle(imgData, imgLbls)

    # split into training and validation data
    train_temp = imgData[:num_trn]
    trainLbls = imgLbls[:num_trn]

    valTest = np.delete(imgData, np.linspace(0,num_trn, num_trn+1, dtype=int), axis=0)
    valTestLbls = np.delete(imgLbls, np.linspace(0,num_trn, num_trn+1, dtype=int), axis=0)

    val_temp = np.delete(valTest, np.linspace(0,num_test-1, num_test, dtype=int), axis=0)
    valLbls = np.delete(valTestLbls, np.linspace(0,num_test, num_test, dtype=int), axis=0)

    test_temp = np.delete(valTest, np.linspace(num_test,num_test*2-1, num_test, dtype=int), axis=0)
    testLbls = np.delete(valTestLbls, np.linspace(num_test,num_test*2-1, num_test, dtype=int), axis=0)

    print("Validation labels: ", valLbls)
    print("Test labels: ", testLbls)


    train = []
    for i in range(len(train_temp)):
        train.append(np.reshape(train_temp[i], (256,256,1)))
    val = []
    for i in range(len(val_temp)):
        val.append(np.reshape(val_temp[i], (256,256,1)))
    test = []
    for i in range(len(test_temp)):
        test.append(np.reshape(test_temp[i], (256,256,1)))


    return train, trainLbls, val, valLbls, test, testLbls
# --------------------------------------------------------------------------------------------------

# imgDF = importImages('datasets/Spiral_DataSet1_relabelled')
# _, _, _, _, test, testLbls = splitData(imgDF)