import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
TRAIN_X_DATA_SOURCE = "./data/train/train/"
TRAIN_Y_DATA_SOURCE = "./data/train_masks/train_masks/"
TRAIN_X_FILES = os.listdir(TRAIN_X_DATA_SOURCE)
TRAIN_Y_FILES = os.listdir(TRAIN_Y_DATA_SOURCE)

def displayImage(tensor):
    tensor = tensor.permute(0, 2, 3, 1)
    plt.imshow(tensor[0])
    plt.show()

class ImageDataset(Dataset):

    def __init__(self, scale = 0.5):
        self.trainXFiles = TRAIN_X_FILES
        self.trainYFiles = TRAIN_Y_FILES
        self.trainSize = len(TRAIN_X_FILES)
        self.scale = scale
        self.imShape = (1280, 1918, 1)

    def preprocess(self, img, isMask = False):
        y, x, _ = self.imShape
        newX, newY = int(x * self.scale), int(y * self.scale)
        img = img.resize((newX, newY), resample = Image.BICUBIC) # test with linear for X later for performance, CONSIDER RESIZING TO (512, 512) or some power of 2


        # data processing final
        img = np.asarray(img)

        if isMask:
            img = img.reshape((newY, newX , 1))
        else:
            img = img / 255.0
        # want 3 matrices one for each rgb value, so we then need to transpose
        img = img.transpose((2, 0, 1)) # 3 nxn matrices, one for red one for green one for blue
        return torch.as_tensor(img.copy()).float().contiguous()
        
    def __len__(self):
        return self.trainSize
    
    def __getitem__(self, index):
        X = Image.open(TRAIN_X_DATA_SOURCE + self.trainXFiles[index])
        Y = Image.open(TRAIN_Y_DATA_SOURCE + self.trainYFiles[index])

        X = self.preprocess(X)
        Y = self.preprocess(Y, True)
        
        return {
            'X': X,
            'Y': Y
        }
        # print(Y)
        # print(np.sum(Y))

dataset = ImageDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



# Get one random sample

# sample = next(iter(dataloader))
# x = sample['X']
# y = sample['Y']
# print(torch.unique(x))
# print(y.shape)
# displayImage(x)
# displayImage(y)
