import cv2 as cv
import numpy as np
import easyTips as et
from skimage.filters import threshold_isodata
import matplotlib.pyplot as plt
import time

def imgSegment(filePath,fileType,matchFlag,outputCoordinateFlag, saveFolder,adaptiveThreshold,outputBigImg ,morphFlag ,
               noSuperviewThre,sigmaBlur, kernalSizeBlur ,saveWithBackgroundFlag,sizeLimit, paddingNum, adaptPare1,adaptPare2,morphPara1):

    sizeLimit += paddingNum
    fileList = et.fileList(filePath, fileType)

    savePath = et.creatFolder(saveFolder, filePath= filePath)
    print(filePath,saveFolder,savePath)

    imgNum = len(fileList)


    for n in np.arange(imgNum):
        if matchFlag:
            subFolderName = "%05d" % n
            saveSubPath = et.creatFolder(subFolderName, filePath=savePath)
        else:
            saveSubPath = savePath
        img = cv.imread(fileList[n])
        img = et.rgb2gray(img)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if morphFlag:
            element = cv.getStructuringElement(cv.MORPH_RECT, morphPara1)  #(16,16)
            mask = cv.morphologyEx(img, cv.MORPH_OPEN, element,iterations=2)

        mask = cv.GaussianBlur(mask, kernalSizeBlur, sigmaBlur)
    # ----------------------way 1 no supervise, the threshold num is calculated by the isodata method
        if noSuperviewThre:
            threNum = threshold_isodata(mask)
            (_, mask) = cv.threshold(mask, int(threNum), 255, cv.THRESH_BINARY_INV)

    # -----------------------way2 adaptive threshold , you need to enter the num by yourself
        elif adaptiveThreshold:
            mask = cv.adaptiveThreshold(mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, adaptPare1, adaptPare2) # 121, 4

        mask[ 1087:1187, 984:1084] = 0

        (objPoints, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        objPoints = sorted(objPoints, key=cv.contourArea, reverse=True)
        objNum = len(objPoints)

        if outputCoordinateFlag:
            coordinateName = open(saveSubPath + '/coordinate.txt', 'w')

        if outputBigImg:
            boxList = []

        coorN = 0
        for m in np.arange(objNum):
            rect = cv.minAreaRect(objPoints[m])
            box = np.int0(cv.boxPoints(rect))



            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            xMin =np.int( min(Xs) - paddingNum/2)
            xMax = np.int(max(Xs) + paddingNum/2)
            yMin = np.int(min(Ys) - paddingNum/2)
            yMax = np.int(max(Ys) + paddingNum/2)
            height = yMax - yMin
            width = xMax - xMin

            if width > sizeLimit and height> sizeLimit :

                if saveWithBackgroundFlag:
                    if xMin < 0:
                        xMin = 0
                    if yMin < 0:
                        yMin = 0
                    if xMax > imgWidth - 1:
                        xMax = imgWidth - 1
                    if yMax > imgHeight - 1:
                        yMax = imgHeight - 1
                    cropImg = img[yMin:yMax, xMin:xMax]
                else:
                    width = int(rect[1][0])
                    height = int(rect[1][1])

                    src_pts = box.astype("float32")
                    dst_pts = np.array([[paddingNum, height + paddingNum],
                                        [paddingNum, paddingNum],
                                        [width + paddingNum, paddingNum],
                                        [width + paddingNum, height + paddingNum]], dtype="float32")
                    M = cv.getPerspectiveTransform(src_pts, dst_pts)
                    cropImg = cv.warpPerspective(img, M, (width+2*paddingNum, height+2*paddingNum))
                imgName = et.outputName(fileList[n])

                saveFullPath = saveSubPath+'/'+imgName+'_'+"%04d" % coorN +'.tif'
                cv.imwrite(saveFullPath, cropImg)
                if outputBigImg:
                    boxList.append(m)
                if outputCoordinateFlag:
                    coordinate = "%04d" % coorN + ':' + str(yMin)+',' + str(yMax)+',' + str(xMin)+',' + str(xMax) + '\n'
                    coordinateName.write(coordinate)
                coorN += 1
        coordinateName.close()



        if outputBigImg:
            for boxNum in np.array(boxList):
                rect = cv.minAreaRect(objPoints[boxNum])

                box = np.int0(cv.boxPoints(rect))

                cv.drawContours(img, [box], -1, 255, 3)  # img is the image, which you want to draw a box on,  secound para is the border,
            # forth is the color, fifth is the bold

            cv.imwrite(savePath+'/'+imgName+'.tif', img)

    # print(time.time()-startTime)
