# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

# read in the file
file = cv2.imread('C:/Users/Rebecca/Documents/Virginia_Tech/Research/git-repos/hand-tremor-detection/spiral-classifier/datasets/handPD_new/test/healthy/sp1-H1.jpg', 1)
print(np.shape(file))
# plt.matshow(file)
# plt.show()

# blur the image
fileBlur = cv2.blur(file, (5,5))
img = cv2.medianBlur(fileBlur, 11)

# plt.matshow(img)
# plt.show()

# iterate through all points in the image and threshold
nx = np.shape(img)[0]
ny = np.shape(img)[1]

for y in range(ny): # col
    for x in range(nx): # row
        color = img[x][y]
        if ((color[0] > 200) & (color[1] > 200) & (color[2] > 200)) | (color[0] < 70):
            img[x][y][0] = 255
            img[x][y][1] = 255
            img[x][y][2] = 255

# plt.matshow(img)
# plt.show()

# perform erosion and dilation
dilation_size = 1
dilateElement = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2*dilation_size+1,2*dilation_size+1), anchor=(dilation_size,dilation_size))
img = cv2.dilate(img, dilateElement)
img = cv2.dilate(img, dilateElement)

# plt.matshow(img)
# plt.show()

erosion_size = 2
erodeElement = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2*erosion_size+1,2*erosion_size+1), anchor=(erosion_size,erosion_size))
img = cv2.erode(img, erodeElement)

# plt.matshow(img)
# plt.show()

for y in range(ny):
    for x in range(nx):
        color = img[x][y]
        difRG = abs(color[0] - color[1])
        difRB = abs(color[0] - color[2])
        difGB = abs(color[1] - color[2])

        if (difGB < 30) & (difRB < 30) & (difRG < 30):
            img[x][y][0] = 255
            img[x][y][1] = 255
            img[x][y][2] = 255

# plt.matshow(img)
# plt.show()

img = cv2.blur(img, (5,5))
img = cv2.medianBlur(img, 3)

plt.matshow(img)
plt.show()



