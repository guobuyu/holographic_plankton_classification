# holographic_plankton_classification
Automated Plankton Classification from Holographic Imagery with Deep Convolutional Neural Networks
Citation
If you find this work or code is helpful in your research, please cite:

title={Automated Plankton Classification from Holographic Imagery with Deep Convolutional Neural Networks}, 
author={Buyu Guo and Lisa Nyman and Aditya R. Nayak and David Milmore and Malcolm McFarland and Michael S. Twardowski and James M. Sullivan and Jia Yu and Jiarong Hong},

## Table of Contents
* [Installation](#installation)
* [Instructions for Use](#instructions-for-use)
* [How to use](#how-to-use)

# Installation

The following Python libraries should be installed with `pip`:
* thorch
* torchvision
* matplotlib
* trackpy
* opencv 
* sklearn

Example of how to install with pip:
```
$ pip3 install --user sklearn
```

# Instructions for Use

##Training
1. Create training and validation datasets. These should be a txt file with columns for image ID and their correspoding classes. 
2. Modify training parameters in [teShufflenet15.py.py]
3. Train the model

Parameters in [teShufflenet15.py]:
* batchSize: he number of training examples utilized in one iteration.
* numClasses: how many classes we have.
* numEpoch: the number of passes of the entire training dataset the machine learning algorithm has completed.
* testSize: the ratio to split your data into training and testing 
* learningRate: a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a * * minimum of a loss function.
* dropNum: ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random.

* featureExtract = True
* usePretrained = False
* dropout = True
* mono: image type is grayscale or RGB 
* savePreAndRealFlag: a parameter for 

* inputSize: size of training images. If their length/width is smaller the inputSize, it will be padded to inputSize
* imgType: type of the image
* txtFil: txt file generated in step 1
* modelName: the name of output model
* filePath: set the path
* savePath：set the saving path


##Classifying
1. Modify training parameters in [groupProcessing.py]
2. Processing

* holoPath: path of holograms
* netPath: path of the network
* classNum: how many classes we need to classify
* fileType: image type

* segmentSavePath: the save path for segments (cropped from holograms)
* cropSizeLimit: segments with length or width less than cropSizeLimit will not be considered
* morphFlag: whether to use morphological analysis
* adaptiveThreshold: whether to use adaptive threshold
* noSuperviewThre: whether to use morphological analysis
* outputRawImg: whether to output the original holograms
* outputCoordinateFlag:  whether to output the coordinates of segments
* saveWithBackgroundFlag = True
* paddingNum: add several pixels around segments

Please refer to the opencv documentation for the following parameters：
* kernalSizeBlur: https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
* adaptPare1: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html 
* adaptPare2: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
* morphPara1: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html

* outputPath: output path
* inputSize: no need to be same with training para
* batchSize: 
* saveClsaaFlag: whether to save classes separatly
* printScoreFlag: whether to print the classification score 
* paddingFlag: whether to pad the images
* topFlag: whether to save the top score
* respectivelyCountFlag: whether to save the class of segments separatly


# How to use
* Train: 
1. Mark species as numbers (e.g., C. debils is 0; Diatom sp. is 1; D. brightwelli is 2 etc.)  
2. Create a txt file map the images path with their species (shown in the figure) 
3. Set parameters in ‘teShufflenet15.py’ 
4. Run ‘teShufflenet15.py’ 


* Classify: 

** Classify the full-size holograms 
1. Set parameters in ‘groupProcessing.py’ 
2. Run ‘groupProcessing.py’ 

** Classify the segments 
1. Comment out ‘imgCrop.imgSegment(…)’ in ‘groupProcessing.py’ 
2. Set parameters in ‘groupProcessing.py’ 
3. Run ‘groupProcessing.py’ 
