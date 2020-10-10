import torch
from torchvision import transforms,datasets
import numpy as np
import os
from torch.utils.data import DataLoader
import time
import torch.nn as nn
from dataLoadProc import microPlankton

def netClassify(softCalFlag, paddingFlag, groupNum, netPath,imgPath,outputPath,labelLib,fileType, inputSize,batchSize,classes,saveClsaaFlag,respectivelyCountFlag,printScoreFlag, topFlag):
    # netPath = 'E:/UMN/planktonProgram/paper/file/net/resnet34_03280444_res01_batch16.pt'
    # imgPath = r'E:\UMN\planktonProgram\paper\pic\crop/'
    # outputPath = imgPath
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # labelLib=['Diatom_1', 'Diatom_2','Diatom_3','Diatom_4','Diatom_5','Copepod','Copepod_Nauplii','Ciliate','Ceratium_sp1','Ceratium_sp2']
    # fileType = '.tif'
    # inputSize = 400
    # batchSize = 20
    drawImgFlag = True
    # classes = 10
    # saveClsaaFlag = True
    # respectivelyCountFlag = True
    modelUse = torch.load(netPath)
    modelUse = modelUse.to(device)
    if paddingFlag:
        imgTransform = transforms.Compose([#transforms.Pad(4, fill=0, padding_mode='constant'),
                                       transforms.ToTensor(),
                                       # transforms.Normalize(),
                                       ])
    else:
        imgTransform = transforms.ToTensor()

    # ----------------------get the img list START
    def getImg(directory):
        txtfile  = []
        for f in sorted(list(os.listdir(directory))):
            if f.endswith(fileType):
                if os.path.isfile(os.path.join(directory, f)):
                    a = os.path.join(directory, f)
                    txtfile.append(a)
        return txtfile
    # ----------------------get the img list END

    file = getImg(imgPath)         #get the img list
    imgDataset = microPlankton(paddingFlag,inputSize, imgPath, file, imgTransform)
    imgDataloader = torch.utils.data.DataLoader(imgDataset,batch_size=batchSize,
                                                shuffle=False, num_workers=0, pin_memory=True)

    def predictClass(model, imgDataloader,softMaxCal, drawImgFlag, savePath):
        topRecord = []
        printOutList = []
        topList = []
        for input in imgDataloader:
            input = input.to(device)
            model.eval()
            with torch.autograd.set_grad_enabled(False):
                out = model(input)
                if softMaxCal:
                    softMax = nn.Softmax(dim=1)
                    out = softMax(out)
                if printScoreFlag:
                    printOut = out.cpu().numpy()
                # ps = torch.exp(out)
                top = out.topk(1, dim=1)
                if topFlag:
                    topScore = top[0].cpu().numpy()
                top = top[1].cpu().numpy()
                topRecord.append(str(top))
                topList.append(topScore)
                printOutList.append(printOut)
        return topRecord,printOutList,topList

    def printFile(outputPath, classFile,classTotalNum, respectivelyCountFlag,scoreList,scoreFlag,topsList,topsFlag):
        txtName = open(outputPath +'/'+ "%05d" % groupNum+'.txt', 'w')  # creat txt

        for n in np.arange(0,classTotalNum):

            classNum = str(classFile).replace(' ','').count('['+str(n)+']')
            dataWrite = 'Number of '+ labelLib[n]+ ' is '+ str(classNum) + '\n'  # write
            txtName.write(dataWrite)
        txtName.close()
        if topsFlag:
            topFile = open(outputPath + '/' + "%05d" % groupNum + '_topScore.txt', 'w')
            topsList = str(topsList).replace(' ', '').replace(',dtype=float32)','').replace("\n",'').replace('array(','').replace('[','').replace('],', ' ').replace('],', ' ').replace(']', '').split(' ')

            if scoreFlag:
                scoreFile = open(outputPath + '/' + "%05d" % groupNum + '_score.txt', 'w')
                scoreList  = str(scoreList).replace(' ', '').replace(',dtype=float32)','').replace("\n",'').replace('array(','').replace('[','').replace('],', ' ').replace('],', ' ').replace(']', '').split(' ')

            if respectivelyCountFlag:
                txtNameRes = open(outputPath + '/' + "%05d" % groupNum + '_Respectively.txt', 'w')  # creat txt
                classFile = str(classFile).replace(' ','').replace('[','').replace("'",'').replace(r']\n',' ').replace(']', '').replace(',', ' ').split(' ')


            for sortNum in np.arange(len(topsList)):
                # tops = topsList[sortNum]
                tops = "%04d" % sortNum + ': ' + str(topsList[sortNum])+ '\n'
                topFile.write(tops)
                if scoreFlag:
                    # score = scoreList[sortNum]
                    score = "%04d" % sortNum + ': ' + str(scoreList[sortNum]) + '\n'
                    scoreFile.write(score)
                if respectivelyCountFlag:
                    a = classFile[sortNum]
                    txtNameRes.write("%04d" % sortNum + ': ' + labelLib[np.int(a)] + '\n')

            topFile.close()
            if scoreFlag:
                scoreFile.close()
            if respectivelyCountFlag:
                txtNameRes.close()


    pre, scoreList, topList = predictClass(modelUse, imgDataloader,softCalFlag, drawImgFlag, imgPath)

    # print( time.time() - beginT)



    if saveClsaaFlag:
        printFile(outputPath, pre, classes,respectivelyCountFlag,scoreList,printScoreFlag,topList, topFlag)
    else:
        print(pre)
