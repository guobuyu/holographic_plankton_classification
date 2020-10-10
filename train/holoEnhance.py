import numpy as np
import cv2 as cv
from math import pi
from numpy import fft
import os

filePathOri = '/home/hongj/bguo/holoTrain/trainData2/noCutHoloEnh/'
classNum = 5
# saveFolder = 'E:/UMN/planktonProgram/'
# saveFolder = filePath
waveLen = 0.660
reso = 4.584
zMax = 21000
stepNum = 10
saveWay = 0  # real:0 real,  imaginary:1 imag,  amplitude:2 abs,  phase:3 angle
imgType = 'tif'
splitFolder = False

def makeFolder(subFolderName, saveFolder):
    if splitFolder:
        try:
            os.mkdir(saveFolder + subFolderName)
            savePath = os.path.join(saveFolder, subFolderName) + '/'
        except:
            savePath = os.path.join(saveFolder, subFolderName) + '/'

    else:
        savePath = saveFolder

    return savePath

def holoRe(holo,imgSavePath,imgName,imgDotType):
    imgX = holo.shape[1]
    imgY = holo.shape[0]
    matx = np.arange(0, imgX)
    maty = np.arange(0, imgY)
    [matX, matY] = np.meshgrid(matx, maty)
    fx = (np.mod(matX + imgX / 2, imgX) - np.floor(imgX / 2)) / imgX
    fy = (np.mod(matY + imgY / 2, imgY) - np.floor(imgY / 2)) / imgY
    f2 = fx * fx + fy * fy
    sqrtInput = 1 - f2 * np.square(waveLen/reso)
    sqrtInput = np.maximum(sqrtInput, 0)
    H = -2 * pi * 1j * np.sqrt(sqrtInput) / waveLen

    step = zMax/stepNum
    for z in np.arange(0,zMax+1,step):
        fftHolo = fft.fft2(holo)
        phase = np.exp(2*pi*1j*z/waveLen)
        Hz = np.exp(z*H)
        imgOutput = fft.ifft2(Hz*fftHolo)*phase
        imgSaveName = imgSavePath+ imgName +'-'+'%07.2f'%z+imgDotType
        if saveWay==0:
            imgSave = np.real(imgOutput)
        elif saveWay==1:
            imgSave = np.imag(imgOutput)
        elif saveWay ==2:
            imgSave = np.abs(imgOutput)
        elif saveWay ==3:
            imgSave = np.angle(imgOutput)
        imgSave = (imgSave - np.min(imgSave)) / (np.max(imgSave) - np.min(imgSave)) * 255
        imgSave = np.uint8(imgSave)
        # print(imgSavePathName)
        # print(imgSaveName)
        cv.imwrite(imgSaveName, imgSave)

for n in np.arange(0,classNum):
    filePath = filePathOri+str(n)+'/'
    imgList = os.listdir(filePath)
    for imgs in imgList:
        fileType = os.path.splitext(imgs)[1]
        fileName = os.path.splitext(imgs)[0]
        imgDotType = '.' + imgType
        if fileType == imgDotType:

            imgGoing = filePath + imgs
            # print(imgGoing)
            imgGray = cv.imread(imgGoing)
            numOfshape = len(imgGray.shape)
            if numOfshape == 3:
                imgGray = cv.cvtColor(imgGray, cv.COLOR_BGR2GRAY)
            # imgGray = np.float32(imgGray)

            imgSavePath = makeFolder(fileName,filePath)

            reconHolo = holoRe(imgGray,imgSavePath,fileName,imgDotType)
        else:
            continue
