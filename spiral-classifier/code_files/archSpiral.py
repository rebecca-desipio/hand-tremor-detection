# *******************************
# FILE DESCRIPTION

# This file contains various functions to perform spiral manipulation (i.e. thinning, unravelling, etc.)
# Functions in this file are:
#   - generateIdealSpiral --> creates an ideal spiral from the archimedean spiral math model
#   - skeletonize_imgs    --> performs skeletonization via binarization threshold
#   - unravelSpiral       --> takes skeletonized images as input finds the center starting point
#   - calcDist_to_center  --> takes skeletonized image as input and the starting point calculated from 'unravelSpiral'
#                             and unravels the spiral where the x-axis would be sample points and the y-axis is 
#                             R, or distance to the center
# *******************************
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
# ----------------------------------------------------------------------------------

# define a function to unravel the spiral
def unravelSpiral(skel_img): # not using obj detection
    # find a first point on the spiral
    notOnSpiral = True

    halfpt = int(np.shape(skel_img)[0] / 2) # go halfway down the image, and iterate over until hit the spiral
    for i in range(len(skel_img)):
        if skel_img[halfpt][i] == 255:
            c = [halfpt, i]
            break
    

    row = c[0]
    col = c[1]
    # while notOnSpiral:
    #     if skel_img[row][col] == 255:
    #         onSpiral = [row, col]
    #         notOnSpiral = False
    #     else:
    #         row = row-1

    onSpiral = [row, col]
    # ----------------------
    # find the starting point of the spiral
    notAtStart = True
    checkRight = True
    row = onSpiral[0]
    col = onSpiral[1]
    prevRow = 0
    prevCol = 0
    while notAtStart:
        # print('Current RowCol: ', [row, col])

        # check the top row (L to R)
        if (skel_img[row-1][col-1] == 255) & ([row-1,col-1] != [prevRow, prevCol]):
            prevRow = row
            prevCol = col
            row = row-1
            col = col-1
        elif (skel_img[row-1][col] == 255) & ([row-1,col] != [prevRow, prevCol]):
            prevRow = row
            prevCol = col
            row = row-1
            col = col
        elif (skel_img[row-1][col+1] == 255) & ([row-1,col+1] != [prevRow, prevCol]):
            prevRow = row
            prevCol = col
            row = row-1
            col = col+1

        # check to the right (R to BR)
        elif (skel_img[row][col+1] == 255) & ([row,col+1] != [prevRow, prevCol]):
            prevRow = row
            prevCol = col
            row = row
            col = col+1
        elif (skel_img[row+1][col+1] == 255) & ([row+1,col+1] != [prevRow, prevCol]):
            prevRow = row
            prevCol = col
            row = row+1
            col = col+1

        # check below (R to L)
        elif (skel_img[row+1][col] == 255) & ([row+1,col] != [prevRow, prevCol]):
            prevRow = row
            prevCol = col
            row = row+1
            col = col
        elif (skel_img[row+1][col-1] == 255) & ([row+1,col-1] != [prevRow, prevCol]):
            prevRow = row
            prevCol = col
            row = row+1
            col = col-1

        # check to the left
        elif (skel_img[row][col-1] == 255) & ([row,col-1] != [prevRow, prevCol]):
            prevRow = row
            prevCol = col
            row = row
            col = col-1

        # if none of the above, it is the end point
        else:
            notAtStart = False

    startPt = np.array([row,col])
    print('Center Point: ', startPt)
    # ----------------------

    # # ----------------------
    # xy_coords = np.argwhere(skel_img == 255)
    # distance_to_center = []

    # # compute the distances to the starting point of the spiral
    # for i in range(len(xy_coords)):

    #     # calculate euclidean distance
    #     dist = np.sqrt((xy_coords[i][0] - startPt[0])**2 + (xy_coords[i][1] - startPt[1])**2)
    #     distance_to_center.append(dist)
    # # ----------------------
    # test = np.sort(distance_to_center)

    return startPt #test

# ----------------------------------------------------------------------------------
def calcDist_to_center(img, c):
    dist_to_center = []
    # find the starting point of the spiral
    notAtEnd = True
    row = c[0]
    col = c[1]
    prevRow = 0
    prevCol = 0

    pixelLoc = []
    while notAtEnd: # iterate around CCW
        # print('Current RowCol: ', [row, col])

        # check the top row (R to L)
        if (img[row-1][col+1] == 255) & ([row-1,col+1] != [prevRow, prevCol]):
            pixelLoc.append('TR')
        if (img[row-1][col] == 255) & ([row-1,col] != [prevRow, prevCol]):
            pixelLoc.append('T')
        if (img[row-1][col-1] == 255) & ([row-1,col-1] != [prevRow, prevCol]):
            pixelLoc.append('TL')

        # check to the left (L to BL)
        if (img[row][col-1] == 255) & ([row,col-1] != [prevRow, prevCol]):
            pixelLoc.append('L')
        if (img[row+1][col-1] == 255) & ([row+1,col-1] != [prevRow, prevCol]):
            pixelLoc.append('BL')

        # check below (B to BR)
        if (img[row+1][col] == 255) & ([row+1,col] != [prevRow, prevCol]):
            pixelLoc.append('B')
        if (img[row+1][col+1] == 255) & ([row+1,col+1] != [prevRow, prevCol]):
            pixelLoc.append('BR')

        # check to the right
        if (img[row][col+1] == 255) & ([row,col+1] != [prevRow, prevCol]):
            pixelLoc.append('R')

        # if none of the above, it is the end point
        if pixelLoc == []:
            notAtEnd = False

        else:
            # set the prev row and col pixel locations
            prevRow = row
            prevCol = col

            if len(pixelLoc) == 1:
                loc = pixelLoc[0]
            else:
                loc = pixelLoc[1]

            if loc == 'TR':
                row = row-1
                col = col+1
            elif loc == 'T':
                row=row-1
                col=col
            elif loc == 'TL':
                row=row-1
                col=col-1
            elif loc == 'L':
                row=row
                col=col-1
            elif loc == 'BL':
                row=row+1
                col=col-1
            elif loc == 'B':
                row=row+1
                col=col
            elif loc == 'BR':
                row=row+1
                col=col+1
            elif loc == 'R':
                row=row
                col=col+1

        # calculate the distance to the center and store in array
        # row is x, col is y
        dist = np.sqrt((row - c[0])**2 + (col - c[1])**2)
        dist_to_center.append(dist)   

        pixelLoc = []
        

    return dist_to_center


    



