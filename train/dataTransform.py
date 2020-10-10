from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

# def default_loader(path):
#     return Image.open(path).convert('F')

class dataTran(Dataset):
    # 构造函数带有默认参数
    def __init__(self, dataSet, transform=None, target_transform=None):
        self.dataSet = dataSet
        self.transform = transform
        self.target_transform = target_transform
        # self.loader = loader

    def __getitem__(self, item):
        imgData, label = self.dataSet[item]
        # img = self.loader(imgData)
        if self.transform is not None:
            try:
                img = self.transform(imgData)
            except:
                print(imgData)
        return img, label

    def __len__(self):
        return len(self.dataSet)

