import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
import time
import os
import copy
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from data import microPlankton
import cv2 as cv
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from dataTransformNew import dataTran
import time

batchSize = 32  # 11111111111
numClasses = 8
numEpoch = 300
testSize = 0.2  # 11111111111
learningRate = 0.01  # 11111111111
dropNum = 0.5
momentum = 0.9

featureExtract = True
usePretrained = False
dropout = True
mono = True
savePreAndRealFlag = True

samplePath = '/home/hongj/bguo/holoTrain/trainData/'
inputSize = 400
imgType = '.tif'
txtFile = "dataFileWb7Merge.txt"  # 11111111111
modelName = "shufflenet_v2_x1_5"
filePath = "/home/hongj/bguo/holoTrain/"
savePath = '/home/hongj/bguo/holoTrain/modelGen/' + modelName + '_'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------find the input size for padding START
def findMax(imgPath, fileType):
    maxPixel = 0
    for mainPath, dirs, file in os.walk(imgPath, topdown=False):
        for subFolderName in dirs:  # read subfolder
            pathNow = os.path.join(mainPath, subFolderName)
            for subMainPath, subDirs, subFile in os.walk(pathNow, topdown=False):  # find data file
                for dataFile in subFile:  # read data file
                    if os.path.splitext(dataFile)[1] == fileType:  # find data with specific suffix
                        img = cv.imread(subMainPath + '/' + dataFile)
                        try:
                            tempPixel = max(img.shape)
                        except:
                            print(dataFile)
                        maxPixel = max(tempPixel, maxPixel)
    return maxPixel


# inputSize = findMax(samplePath, imgType)
# --------------------find the input size for padding END

dataTransforms = {
    "trainData": transforms.Compose([
        # transforms.Resize(inputSize),
        # transforms.RandomCrop(size=inputSize, pad_if_needed=True, padding_mode='constant'),
        transforms.RandomHorizontalFlip(p=0.5),  # horizontal flip and 0.5 is the position
        transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(45, resample=False, expand=False,
                                              center=None),
        transforms.ToTensor(),
        # transforms.Normalize([],[])
    ]),
    "testData": transforms.Compose([
        # transforms.Resize(inputSize),
        # transforms.RandomCrop(size=inputSize, pad_if_needed=True, padding_mode='constant'),
        transforms.RandomHorizontalFlip(p=0.5),  # horizontal flip and 0.5 is the position
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize([],[])
    ])
}

imagDataset = microPlankton(root=filePath, dataTXT=txtFile, )
train_set, test_set = train_test_split(imagDataset, test_size=testSize, random_state=42)
trainSet = dataTran(inputSize,train_set, transform=dataTransforms["trainData"])
testSet = dataTran(inputSize,test_set, transform=dataTransforms["testData"])
testDataloader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=True, num_workers=24,
                                             pin_memory=True)
trainDataloader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True, num_workers=24,
                                              pin_memory=True)


# -------------copy from resnet


# --------------------Show imgs in the dataset Start
# imgs = next(iter(trainDataloader['trainData']))[0]
# unloader = transforms.ToPILImage()
#
# def imgShow( tensor, title):
#     img = tensor.cpu().clone()
#     img = img.squeeze(0)
#     img = unloader(img)
#     plt.imshow(img, cmap="gray")
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)
#
# # print(imgs.shape)
# plt.figure()
# imgShow(imgs[3],title="img")

# --------------------Show imgs in the dataset End


def setParameterRequiresGrad(model, featureExtract):
    if featureExtract:
        for param in model.parameters():
            param.requires_grad = False


def initialModel(modelName, numClasses, featureExtract, usePretrained):
    if modelName == "vgg19":
        modelUse = models.vgg19(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        if mono:
            modelUse.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        modelUse.classifier._modules['6'] = nn.Linear(in_features=4096, out_features=numClasses, bias=True)

    elif modelName == "shufflenet_v2_x2_0":
        modelUse = models.shufflenet_v2_x2_0(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        if mono:
            modelUse.conv1._modules['0'] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if dropout:
            modelUse.fc = nn.Sequential(nn.Dropout(dropNum),
                                        nn.Linear(2048, numClasses, bias=True))
        else:
            modelUse.fc = nn.Linear(2048, numClasses, bias=True)

    elif modelName == "shufflenet_v2_x0_5":
        modelUse = models.shufflenet_v2_x0_5(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        if mono:
            modelUse.conv1._modules['0'] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if dropout:
            modelUse.fc = nn.Sequential(nn.Dropout(dropNum),
                                        nn.Linear(1024, numClasses, bias=True),
                                        )
        else:
            modelUse.fc = nn.Linear(1024, numClasses, bias=True)


    elif modelName == "shufflenet_v2_x1_5":
        modelUse = models.shufflenet_v2_x1_5(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        if mono:
            modelUse.conv1._modules['0'] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if dropout:
            modelUse.fc = nn.Sequential(nn.Dropout(dropNum),
                                        nn.Linear(1024, numClasses, bias=True),)
        else:
            modelUse.fc = nn.Linear(1024, numClasses, bias=True)


    elif modelName == "shufflenet_v2_x1_0":
        modelUse = models.shufflenet_v2_x1_0(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        if mono:
            modelUse.conv1._modules['0'] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if dropout:
            modelUse.fc = nn.Sequential(nn.Dropout(dropNum),
                                        nn.Linear(1024, numClasses, bias=True))
        else:
            modelUse.fc = nn.Linear(1024, numClasses, bias=True)


    elif modelName == "squeezenet1_1":
        modelUse = models.squeezenet1_1(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        if mono:
            modelUse.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))

        modelUse.classifier = nn.Sequential(nn.Dropout(p=dropNum, inplace=False),
                                nn.Conv2d(512, numClasses, kernel_size=(1, 1), stride=(1, 1)),
                                nn.ReLU(inplace=True),
                                nn.AdaptiveAvgPool2d(output_size=(1, 1)))



    elif modelName == "densenet121":
        modelUse = models.densenet121(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        if mono:
            modelUse.features._modules['conv0'] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if dropout:
            modelUse.classifier = nn.Sequential(nn.Dropout(dropNum),
                                        nn.Linear(1024, numClasses, bias=True))
        else:
            modelUse.classifier = nn.Linear(1024, numClasses, bias=True)


    elif modelName == "vgg16":
        modelUse = models.vgg16(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        if mono:
            modelUse.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        modelUse.classifier._modules['6'] = nn.Linear(in_features=4096, out_features=numClasses, bias=True)

    elif modelName == 'mobilenet_v2':
        modelUse = models.mobilenet_v2(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        if mono:
            modelUse.features._modules['0']._modules['0'] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2),
                                                                      padding=(1, 1), bias=False)

        modelUse.classifier = nn.Sequential(
            nn.Dropout(p=dropNum, inplace=False),
            nn.Linear(in_features=1280, out_features=numClasses, bias=True)
        )

    elif modelName == "resnet18":
        modelUse = models.resnet18(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        numFeatures = modelUse.fc.in_features
        if mono:
            modelUse.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if dropout:
            modelUse.fc = nn.Sequential(nn.Dropout(dropNum),
                                        nn.Linear(numFeatures, numClasses))
        else:
            modelUse.fc = nn.Linear(numFeatures, numClasses)

    elif modelName == "resnet34":
        modelUse = models.resnet34(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        numFeatures = modelUse.fc.in_features
        if mono:
            modelUse.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if dropout:
            modelUse.fc = nn.Sequential(nn.Dropout(dropNum),
                                        nn.Linear(numFeatures, numClasses))
        else:
            modelUse.fc = nn.Linear(numFeatures, numClasses)

    elif modelName == "resnet50":
        modelUse = models.resnet50(pretrained=usePretrained)
        setParameterRequiresGrad(modelUse, featureExtract)
        numFeatures = modelUse.fc.in_features
        if mono:
            modelUse.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if dropout:
            modelUse.fc = nn.Sequential(nn.Dropout(dropNum),
                                        nn.Linear(numFeatures, numClasses))
        else:
            modelUse.fc = nn.Linear(numFeatures, numClasses)

    else:
        print("model not implemented")
        return None, None

    return modelUse


def trainModel(model, trainLoader, testLoader, lossFn, optimizer, numEpochs):
    # if len(trainLoader) > len(testLoader):
    #     phase = 'trai'
    bestAccuracy = 0
    bestModuleWeight = copy.deepcopy(model.state_dict())
    trainAccuracyHistory = []
    trainLossHistory = []
    testAccuracyHistory = []
    testLossHistory = []
    for epoch in np.arange(numEpochs):
        runningLoss = 0.
        runningAccuracy = 0.
        model.train()
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.autograd.set_grad_enabled(True):
                outputs = model(inputs)  # bsize * 2
                loss = lossFn(outputs, labels)

            preds = outputs.argmax(dim=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runningLoss += loss.item() * inputs.size(0)
            runningAccuracy += torch.sum(preds.view(-1) == labels.view(-1)).item()
        epochLoss = runningLoss / len(trainLoader.dataset)
        epochAcc = runningAccuracy / len(trainLoader.dataset)
        print("Epoch: {} Phase: {} loss: {}, acc: {}".format(epoch, 'train', epochLoss, epochAcc))

        trainLossHistory.append(epochLoss)
        trainAccuracyHistory.append(epochAcc)

        # bestAccuracy = 0
        # bestModuleWeight = copy.deepcopy(model.state_dict())
        # trainAccuracyHistory = []
        # trainLossHistory = []
        # testAccuracyHistory = []
        # testLossHistory = []

        runningLoss = 0.
        runningAccuracy = 0.
        model.eval()
        if savePreAndRealFlag:
            preResult = []
            realResult = []

        topList = []
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.autograd.set_grad_enabled(False):
                outputs = model(inputs)  # bsize * 2
                loss = lossFn(outputs, labels)

            if epoch>295:
                softMax = nn.Softmax(dim=1)
                out = softMax(outputs)
                top = out.topk(1, dim=1)
                topScore = top[0].cpu().numpy()
                topList.append(topScore)
                topFile = open(filePath+'fileOutput/' + '7wb_topScore.txt', 'w')
                topsList = str(topList).replace(' ', '').replace(',dtype=float32)', '').replace("\n", '').replace(
                    'array(', '').replace('[', '').replace('],', ' ').replace('],', ' ').replace(']', '').split(' ')
                for sortNum in np.arange(len(topsList)):
                    # tops = topsList[sortNum]
                    tops = "%06d" % sortNum + ': ' + str(topsList[sortNum]) + '\n'
                    topFile.write(tops)
                topFile.close()



            preds = outputs.argmax(dim=1)
            runningLoss += loss.item() * inputs.size(0)
            runningAccuracy += torch.sum(preds.view(-1) == labels.view(-1)).item()

            if epoch>295:
                if savePreAndRealFlag:
                    preResult.append(str(preds.view(-1).cpu().numpy()))
                    realResult.append(str(labels.view(-1).cpu().numpy()))
        epochLoss = runningLoss / len(testLoader.dataset)
        epochAcc = runningAccuracy / len(testLoader.dataset)
        print("Epoch: {} Phase: {} loss: {}, acc: {}".format(epoch, 'test', epochLoss, epochAcc))

        if epochAcc > bestAccuracy:
            bestAccuracy = epochAcc
            bestModuleWeight = copy.deepcopy(model.state_dict())

        testLossHistory.append(epochLoss)
        testAccuracyHistory.append(epochAcc)
        if epoch % 5 == 0:
            savePath2 = savePath + str(epoch) + '.pt'
            torch.save(model.load_state_dict(bestModuleWeight), savePath2)
    if epoch >295:
        if savePreAndRealFlag:
            preFile = open(filePath +'fileOutput/' + modelName + time.strftime('%m%d%H%M')+ '_wb12_pre.txt', 'w')
            preFile.write(str(preResult))
            realFile = open(filePath+'fileOutput/' + modelName + time.strftime('%m%d%H%M')+ '_wb12_real.txt', 'w')
            realFile.write(str(realResult))
            preFile.close()
            realFile.close()
    model.load_state_dict(bestModuleWeight)
    return model, trainLossHistory, trainAccuracyHistory, testLossHistory, testAccuracyHistory


modelUse = initialModel(modelName, numClasses, featureExtract=False, usePretrained=False)
modelUse = modelUse.to(device)
optimizer = torch.optim.Adam(modelUse.parameters(), lr=learningRate)
lossFn = nn.CrossEntropyLoss()

modelReturn, trHis, trAcc, teHis, teAcc = trainModel(modelUse, trainDataloader, testDataloader, lossFn, optimizer,
                                                     numEpoch)
torch.save(modelReturn, savePath + time.strftime('%m%d%H%M') + 'wb12_res01_batch16.pt')
torch.save(modelReturn.state_dict, savePath + time.strftime('%m%d%H%M') + '_Para_res01_batch16.pt')
torch.cuda.empty_cache()