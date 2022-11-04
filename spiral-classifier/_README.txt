Folder Contents:
_archive: contains old code and saved images
	- spiral_array_classifier.ipynb: contains original KNN and SVM implementations
	- image_conversion.py: also has KNN (revist at some point)
code_files: contrains all the python files with organized functions 
	  (originally written in spiral_image_classifier_rev2.ipynb
	- these cannot go through debugging because VScode does not seem to like debug 
	  mode on API (keras and tensorflow)
	- reorganized all tensorflow related functions into a JupyterNotebook 
	  (spiralClassifier_rev3.ipynb), but still pull other functions as needed

Jupyter Notebooks: 
 - spiral_image_classifier_rev2: contains all original code from Summer 2022 
 - objectDetection: locates the centers of the spirals (later to be used in unraveling the spiral)
 
Files of Interest for ECE-6554 Final Project:
 - feature_classifier.ipynb: 
	Uses the pretrained models (VGG16 or ResNet50) to obtain the features of the 
	images right before the fully connected layers (they have been removed). 
 - extractHTfromST_rev3.ipynb: 
	You can run one image at a time to get the break-down of how I extracted the 
	hand-trace (HT) from the spiral-template (ST). The data used as input for this 
	can be either the  handPD_new or Spiral_HandPD data folders. 
 - extractHT.py: 
	I just copied over the code from 'extractHTfromST_rev3.ipynb' and put it in a 
	for-loop to bulk-run the images, rather than doing them one at a time. 
	The results from this file are saved in the handPD_HT data folder. 


And like I said yesterday, it'd probably be good to create your own branch so we don't have to worry about anything accidentally getting overwritten. Just make sure you comment out any "save image" calls, which can be found by CTRL-f "imwrite"
