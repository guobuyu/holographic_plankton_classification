import cv2 as cv
import numpy as np
import os

def findMax(imgPath,fileType):
    maxPixel = 0
    tempPixel = 0
    for mainPath, dirs, file in os.walk(imgPath, topdown=False):
        for subFolderName in dirs:   #read subfolder
            pathNow = os.path.join(mainPath, subFolderName)
            for subMainPath, subDirs,subFile in os.walk(pathNow, topdown=False):  #find data file
                for dataFile in subFile:   #read data file
                    if os.path.splitext(dataFile)[1] == fileType:    #find data with specific suffix
                        img = cv.imread(subMainPath+'/'+dataFile)
                        try:
                            tempPixel = max(img.shape)
                        except:
                            # print(dataFile)
                            print(subMainPath,dataFile)
                            print(img.shape,max(img.shape))
                        maxPixel = max(tempPixel,maxPixel)
    return maxPixel

def pad(maxPixel, img,imgName):
    a, b = img.shape[0], img.shape[1]
    BLACK = [0,0]
    if a <= maxPixel:
        if a % 2 == 0:
            padOne = padTwo = (maxPixel - a) / 2
        else:
            padOne = (maxPixel - a - 1) / 2
            padTwo = padOne + 1
    if b <= maxPixel:
        if b % 2 == 0:
            padThree = padFour = (maxPixel - b) / 2
        else:
            padThree = (maxPixel - b) / 2 -0.5
            padFour = padThree + 1
    padOne,padTwo,padThree,padFour = int(padOne),int(padTwo),int(padThree),int(padFour)
    img = cv.copyMakeBorder(img, padOne, padTwo, padThree, padFour, cv.BORDER_CONSTANT, value=BLACK)
    cv.imwrite(imgName,img)


def imgPrePadding(imgPath,fileType):
    if os.path.isdir(imgPath) == True:
        maxPixel = findMax(imgPath, fileType)
        for mainPath, dirs, file in os.walk(imgPath, topdown=False):
            for subFolderName in dirs:   #read subfolder
                pathNow = os.path.join(mainPath, subFolderName)
                for subMainPath, subDirs,subFile in os.walk(pathNow, topdown=False):  #find data file
                    for dataFile in subFile:   #read data file
                        if os.path.splitext(dataFile)[1] == fileType:    #find data with specific suffix
                            imgName = subMainPath + '/' + dataFile
                            img = cv.imread(imgName)
                            numOfshape = len(img.shape)
                            if numOfshape == 3:
                                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                            pad(maxPixel, img,imgName)



                            # dataName = os.path.join(subMainPath, dataFile)  #obtain the filename
                            # dataName = dataName.replace('\\', '/')  #change the formate
                            # dataWrite = dataName+' '+subFolderName+ '\n'   #write




imgMainPath = '/home/hongj/bguo/holoTrain/trainData2/padOriginal/'
imgType = '.tif'
imgSaveType = '.tif'
imgPrePadding(imgMainPath,imgType)




