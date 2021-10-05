import torch 
from config import *
from dataset import *
from augmentations import *


class DataModule():

    def __init__(self, df, folds):

        df_train = df[df['fold'] != folds]
        df_valid = df[df['fold'] == folds]

        self.trainset = SetiDataset(df_train)
        self.validset = SetiDataset(df_valid)

    def get_dataloaders(self):

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=params['BATCH_SIZE'],
            shuffle = True,
            drop_last = True,
            num_workers=4,
            pin_memory=True
        )

        validloader = torch.utils.data.DataLoader(
            self.validset,
            batch_size = params['BATCH_SIZE'],
            num_workers=4,
            pin_memory=True
        )

        return trainloader, validloader
