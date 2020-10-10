import os
import numpy as np
import easyTips as et

filePath = r'E:\UMN\planktonProgram\insituData\test\crop\新建文件夹\score.txt'
a = open(filePath)
a = a.readlines()
a = str(a)
a = a.replace('[','').replace(']','').replace(' ','').replace("'",'').replace('"','').replace(',',' ')
b = a.split(' ')
for x in range(len(b)):
    b[x] = float(b[x])

b.sort()


