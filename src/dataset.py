import torch 
from config import * 

import pandas as pd 
import numpy as np

import albumentations as A 
import cv2

class SetiDataset(torch.utils.data.Dataset):

    def __init__(self, df, augmentations=None):
        self.df = df
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        filepath = "/content/train/{}/{}.npy".format(row.id[0], row.id)
        image = np.load(filepath)
        #image = image[::2]
        image = image.astype(np.float32)
        #image = np.vstack(image).transpose((1, 0)) 

        if self.augmentations:
            image = self.augmentations(image = image)['image']
        else:
            #image = image[np.newaxis,:,:]
            image = torch.from_numpy(image).float()

        label = torch.unsqueeze(torch.tensor(row.target).float(),-1)

        return {
            'images' : image,
            'labels' : label
        }