import cv2 as cv
import numpy as np
import os
import shutil
import easyTips as et

folderTotalNum = 10
inputFolder = '/home/hongj/bguo/holoTrain/trainData/trainData0411/'
outputFolder = '/home/hongj/bguo/holoTrain/trainData/trainData0411_modify/'
pixelSizeLimit = 60
chooseFileType = '.tif'

for foldNum in np.arange(folderTotalNum):

    path = inputFolder+str(foldNum)
    savePath = outputFolder+str(foldNum)
    fileType = chooseFileType
    pixelLimit = pixelSizeLimit
    fileList = et.fileList(path,fileType)
    savePath = et.creatFolder(savePath,path)

    for n in np.arange(len(fileList)):
        img = cv.imread(fileList[n])
        if img.shape[0] > pixelLimit or img.shape[1] > pixelLimit:
            imgSavePath = os.path.join(savePath, os.path.basename(fileList[n]))
            imgOpenPath = os.path.join(path, fileList[n])
            shutil.copyfile(imgOpenPath, imgSavePath)