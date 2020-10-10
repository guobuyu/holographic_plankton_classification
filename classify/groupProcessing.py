import os
import numpy as np
import imgCrop
import easyTips as et
import loadUse_v2
import time


# ------------------------parameters setting
holoPath = r'E:\UMN\FAU\test/'
netPath = r'E:\UMN\FAU\shufflenet_v2_x1_5_06120202wb12_res01_batch16.pt'
classNum = 3
fileType = '.tif'

# -----------------------preprocessing parameters
segmentSavePath = holoPath+'\crop_withBackground/'
cropSizeLimit = 20
sigmaBlur= 0
morphFlag = True
adaptiveThreshold = True
noSuperviewThre = False
outputRawImg = False
outputCoordinateFlag = True
saveWithBackgroundFlag = True
paddingNum = 20
kernalSizeBlur = (101,101)
matchFlag = True
adaptPare1 = 211
adaptPare2 = 2
morphPara1 = (20,20)

# ---------------------------net parameters
outputPath = holoPath+r'\result_withBackground10'
resultOutPath = ''
inputSize = 400
batchSize = 20
saveClsaaFlag = True
printScoreFlag = True
paddingFlag = True
topFlag = True
softCalFlag = True
respectivelyCountFlag = True
# labelLib = ['Diatom_1', 'Diatom_2', 'Diatom_3', 'Diatom_4', 'Diatom_5', 'Copepod', 'Copepod_Nauplii', 'Ciliate',
#             'Ceratium_sp1','Ceratium_sp2', 'nothing']
# labelLib = ['Diatom_1', 'Diatom_2', 'Diatom_3', 'Diatom_4', 'Diatom_5', 'Copepod', 'Copepod_Nauplii', 'Null',
#             'Null','Null','Null']
# labelLib = ['Diatom_1', 'Diatom_2', 'Diatom_3', 'Diatom_4', 'Diatom_5', 'Copepod', 'Copepod_Nauplii', 'Null',
#             ]
labelLib = ['background', 'fuca_d', 'fuca_s', 'fusus_d', 'fusus_s', 'muelleri_d', 'muelleri_s', 'ciliate', 'diatom',
            'copeped','nauplii', 'round','square']

beginTime = time.time()

# imgCrop.imgSegment(holoPath,fileType,matchFlag, outputCoordinateFlag,segmentSavePath,adaptiveThreshold,outputRawImg ,morphFlag ,
#                noSuperviewThre,sigmaBlur, kernalSizeBlur ,saveWithBackgroundFlag,cropSizeLimit, paddingNum, adaptPare1,adaptPare2,morphPara1)

timeMark1 = time.time()
print('segmentationTime =', timeMark1-beginTime)

folderNum = len([lists for lists in os.listdir(segmentSavePath) if os.path.isdir(os.path.join(segmentSavePath, lists))])

et.creatFolder(outputPath, )

timeMark2 = time.time()

for groupNum in np.arange(folderNum):
    imgPath = segmentSavePath + '/'+"%05d" % groupNum
    loadUse_v2.netClassify(softCalFlag, paddingFlag, groupNum, netPath,imgPath,outputPath,labelLib,fileType, inputSize,batchSize,classNum,saveClsaaFlag,respectivelyCountFlag,printScoreFlag ,topFlag)

timeMark3 = time.time()
print('classification =', timeMark3-timeMark2)
print('totalTime=', timeMark3-beginTime)
