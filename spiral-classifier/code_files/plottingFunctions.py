# import libraries
from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ------------------------
#      CNN analysis
# create a function to plot the test images along with their
# predicted vs. actual labels

def plotImages(testModel, test_array, test_label):
    test_label = np.array(test_label)
    incorrectImgs = []
    incorrectImgIdx = []
    if len(test_label) == 16:
        rows = 2
        cols = 8
    else:
        rows = 3
        cols = 5

    count = 0
    fig, axes = plt.subplots(rows, cols, figsize=(20,8))
    axes = axes.flatten()
    for img, ax in zip(test_array, axes):
        ax.imshow(np.squeeze(img), cmap="gray") # plot image

        # use the model to predict the label
        predImg = testModel.predict(np.expand_dims(test_array[count], axis=0), verbose=0) # use for grayscale
        # predImg = testModel.predict(test_array[count])                       # use for RGB
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
    plt.show()

    return np.array(incorrectImgs), np.array(incorrectImgIdx)

# -------------------------
## Arch Spiral Analysis

# define function to plot the test images overlaid with ideal archimedean spiral
def plotArchSpiralOverlay(test_array, label, archSpiral, testModel, avgError=None, varError=None):
    label = np.array(label)
    if len(label) == 16:
        rows = 2
        cols = 8
    else:
        rows = 3
        cols = 5

    count = 0
    fig, axes = plt.subplots(rows, cols, figsize=(20,6))
    if (avgError!=None) & (varError!=None):
        fig.suptitle("label | pred ~ mean distance error | distance variance", fontsize=16)
    axes = axes.flatten()
    for img, ax in zip(test_array, axes):
        # overlay the simulated archimedean spiral with one of the original test images
        testImg = np.squeeze(test_array[count])
        testImg = cv2.adaptiveThreshold(testImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 15)
        dispOverlay = archSpiral/255 + testImg/255
        ax.imshow(dispOverlay*255) # plot image 
        
        # use the model to predict the label
        predImg = testModel.predict(np.expand_dims(test_array[count], axis=0), verbose=0) # use for grayscale
        # predImg = testModel.predict(test_array[count])                       # use for RGB
        predLabel = np.argmax(predImg[0])       
        
        if label[count] != predLabel:
            if (avgError==None) & (varError==None):
                ax.set_title('Label: ' + str(label[count]) + ' | Pred: ' + str(predLabel), color='red')
            else:
                ax.set_title(str(label[count]) + ' | ' + str(predLabel) + ' ~ E: ' + str(avgError[count]) + ' | V: ' + str(varError[count]), color='red')  
        else:
            if (avgError==None) & (varError==None):
                ax.set_title('Label: ' + str(label[count]) + ' | Pred: ' + str(predLabel), color = 'blue')  
            else:
                ax.set_title(str(label[count]) + ' | ' + str(predLabel) + ' ~ E: ' + str(avgError[count]) + ' | V: ' + str(varError[count]), color='blue')


        count = count + 1
        
    plt.tight_layout()
    plt.show()

def plotSkelImgs(origImg, skelImg):
    numrows = len(origImg)

    fig, axes = plt.subplots(numrows,2, figsize=(24,24))
    ax = axes.ravel()
    j = 0
    for i in range(0, numrows*2, 2):
        ax[i].matshow(origImg[j], cmap='gray')
        ax[i+1].matshow(skelImg[j], cmap='gray')
        fig.tight_layout()
        j = j+1

    fig.savefig('skeletonized_images.png')
    plt.close()


# ------------------------------------------------------
# Plot center detection images
def plotCenters(img_array, centers):
    rows = len(img_array)
    cols = 1
    count = 0
    fig, axes = plt.subplots(rows, cols, figsize=(20,8))
    axes = axes.flatten()
    for img, ax in zip(img_array, axes):
        
        ax.imshow(np.squeeze(img), cmap="gray") # plot image
        count = count + 1
        
    plt.tight_layout()
    plt.show()


    


