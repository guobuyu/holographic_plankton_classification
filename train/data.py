from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('F')

class microPlankton(Dataset):
    # 构造函数带有默认参数
    def __init__(self, root, dataTXT, transform=None, target_transform=None, loader=default_loader):
        fileTXT = open(root + dataTXT, 'r')  # read the txt file,which has imgPath and label
        imgs = []  # creat a empty list named imgs, which is used to baggage some imgs
        for line in fileTXT:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()  # get a list file of fileTXT
            imgs.append((words[0], int(words[1])))  # thus imgs is the list of img and its label

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, item):
        imgDataPath, label = self.imgs[item]
        img = self.loader(imgDataPath)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

