#!/usr/bin/python

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2

from skimage.morphology import medial_axis, skeletonize

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
# ----------------------------------------------------------------------------------

# define a function to skeletonize the images

def skeletonize_imgs(img):
    img =  np.squeeze(img)
    thresh = np.mean(img)
    error = 255 - thresh
    thresh = thresh - error

    # binarize the image
    _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    img = img / 255
    img = 1 - img

    skeleton = skeletonize(img) * 255

    return skeleton


    



