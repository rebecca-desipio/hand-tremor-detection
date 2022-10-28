# This file is used to create a "new" dataset from the handPD_new data
# Set the folder path of the data to be extracted, and set the path for where the new images will be saved
# Set the flag to the type of images being run (either HSV or heatmap)

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ----------------------------------------------------
# iterate through all the images in the set folder
plotting = 0 # set plotting flag

path_origin = os.getcwd()
data_path = '/spiral-classifier/datasets/handPD_HT/'
folderpath = path_origin + data_path + 'train/healthy/' # set the folder path
save_folder_path = path_origin + data_path + 'trainHT/healthy/'

for img_name in os.listdir(folderpath):
    img_path = os.path.join(folderpath, img_name)
    print(img_name)

    save_name = img_name

    # ----------------------------------------
    # obtain hsv (or colorjet) representation
    # ----------------------------------------
    img = cv2.imread(img_path, 1)
    origImg = img.copy()
    # # iterate through all points in the image and threshold (set the paper background to all white)
    nx = np.shape(origImg)[0]
    ny = np.shape(origImg)[1]

    for y in range(ny): # col
        for x in range(nx): # row
            color = origImg[x][y]
            if ((color[0] > 180) & (color[1] > 180) & (color[2] > 180)): # | (color[0] < 70):
                origImg[x][y][0] = 255
                origImg[x][y][1] = 255
                origImg[x][y][2] = 255

    # convert to hsv
    hsvImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2HSV)

    if plotting == 1:
        fig, ax = plt.subplots(1,3)
        ax[0].matshow(hsvImg[:,:,0])
        ax[1].matshow(hsvImg[:,:,1])
        ax[2].matshow(hsvImg[:,:,2])
        ax[0].title.set_text('Hue')
        ax[1].title.set_text('Saturation')
        ax[2].title.set_text('Value')
        plt.show()

    # ----------------------------------------
    # isolate saturation channel, then threshold to remove template
    # ----------------------------------------
    sImg = hsvImg[:,:,1] # saturation channel image

    # remove template by thresholding
    for y in range(ny):
        for x in range(nx):
            if (sImg[x][y] <= 75) & (sImg[x][y] != 0):
                sImg[x][y] = 0
    
    if plotting == 1:
        plt.matshow(sImg)

    # ----------------------------------------
    # remove leftover noise (i.e. template pixels that were missed by thresholding)
    # ----------------------------------------
    # set the sizes and elements to perform erosion and dilation
    size1 = 1
    size2 = 2
    element1 = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2*size1+1,2*size1+1), anchor=(size1,size1))
    element2 = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2*size2+1,2*size2+1), anchor=(size2,size2))

    # perform erosion and dilation
    # this was decided through trial and error
    sImg = cv2.dilate(sImg, element1)
    sImg = cv2.erode(sImg, element2)
    sImg = cv2.dilate(sImg, element1)

    # ----------------------------------------
    # convert back to grayscale, then BGR (cv2 default is BGR, not RGB)
    # ----------------------------------------
    img_grayscale = cv2.bitwise_not(sImg)
    final_img = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2BGR)

    # ----------------------------------------
    # save final image in the set location
    # ----------------------------------------
    save_location = save_folder_path + save_name
    cv2.imwrite(save_location, final_img)

