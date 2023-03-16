
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

dir = os.getcwd()
# imgPath = dir + '/spiral-classifier/skel_1_6_PE0107.jpg'
imgPath = 'C:/Users/Rebecca/Documents/Virginia_Tech/Research/database-images/database-images/skel/spiral/skel_1_2_HE097.jpg' #dir + '/datasets/folador_skeletonize/skeletons/waves/skel_V03HO03.png'
img = cv2.imread(imgPath, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# for some reason it is not always binary. 
# iterate through all pixels and if != 255, push to zero
for x in range(1000):
    for y in range(1000):
        if img[x][y] <= 200:
            img[x][y] = 0
        else:
            img[x][y] = 255

plt.figure(figsize=(15,15))
plt.matshow(img, fignum=1, cmap='gray')
plt.show()
