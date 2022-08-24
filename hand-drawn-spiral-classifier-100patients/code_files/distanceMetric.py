#!/usr/bin/python

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2

# import functions from other Python files
from imagePreprocessing import *
from postModelAnalysis import *
from CNNmodels import *


# generate the ideal archimedean spiral
def generateIdealSpiral():
    # define time and radius 
    t = np.linspace(0, 6*np.pi, 200) # controls number of revolutions 
    r = 1

    # define the x and y components      
    x = r/ (2*np.pi) * (t) * np.cos(t)
    y = r/ (2*np.pi) * (t) * np.sin(t)

    # save the figure and generate a
    # plot of 2d line plot
    plt.figure(figsize=(5,5))
    plt.plot(x,y)
    plt.axis('off')
    plt.tight_layout
    plt.savefig('idealSpiral', pad_inches=0, bbox_inches='tight')
    plt.close()

    openSpiral = PIL.Image.open('idealSpiral.png')
    idealSpiral = openSpiral.convert("L") # convert to grayscale (L), color use (P)
    idealSpiral = cv2.resize(np.array(idealSpiral), (128,128))

    _, thresh_img = cv2.threshold(idealSpiral, 250, 255, cv2.THRESH_BINARY)

    return thresh_img

idealSpiral = generateIdealSpiral()
print(np.shape(idealSpiral))
# plt.matshow(idealSpiral, cmap='gray')
# plt.show()

# ----------------------------------------------------------------------------------
# obtain test images as arrays and the associated labels
imgDF = importImages('datasets/Spiral_DataSet1_relabelled')
train_ft, train_lbl, val_ft, val_lbl, test_ft, test_lbl = splitData(imgDF)
trainMat, valMat, testMat = imgAug_preprocessing(train_ft, val_ft, test_ft)
_,_,_,_, test, testLbl = imgAug(trainMat, train_lbl, valMat, val_lbl, testMat, test_lbl)

# plot the test images overlaid with the ideal archimedean spiral
model = importModel('20220601_run2.h5', test, testLbl)
plotArchSpiralOverlay(test, testLbl, idealSpiral, model, avgError=None, varError=None)




