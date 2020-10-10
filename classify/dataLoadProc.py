from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('F')

class microPlankton(Dataset):
    # 构造函数带有默认参数
    def __init__(self,paddingFlag, sizeLimit,root, dataFile, transform=None, loader=default_loader,):
        # fileTXT = open(root + dataTXT, 'r')  # read the txt file,which has imgPath and label
        imgs = []  # creat a empty list named imgs, which is used to baggage some imgs
        for line in dataFile:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()  # get a list file of fileTXT
            imgs.append(words[0])  # thus imgs is the list of img and its label

        self.padTo = sizeLimit
        self.padFlag = paddingFlag
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        imgDataPath = self.imgs[item]
        img = self.loader(imgDataPath)
        if self.transform is not None:
            if self.padFlag:
                padTo = self.padTo
                a = (padTo - img.size[0]) // 2
                b = (padTo - img.size[1]) // 2

                transform1 = transforms.Pad((a, b, padTo - a - img.size[0], padTo - b - img.size[1]), fill=0,
                                    padding_mode='constant')

                try:
                    img = transform1(img)
                    img = self.transform(img)
                except:
                    print(imgDataPath)
            else:
                try:
                    img = self.transform(img)
                except:
                    print(imgDataPath)
        return img

    def __len__(self):
        return len(self.imgs)

