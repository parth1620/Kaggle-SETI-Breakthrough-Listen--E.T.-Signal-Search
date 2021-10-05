from torchvision import transforms as T
from spec_augment import TimeWarp, TimeMask, FreqMask
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

from config import *

def get_train_transform():
    return A.Compose([
        A.Resize(params['IMG_SIZE'], params['IMG_SIZE']),
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5),
        A.ShiftScaleRotate(rotate_limit=0, p=.25),
        A.MotionBlur(p=.2),
        A.IAASharpen(p=.25),
        A.RandomBrightnessContrast(always_apply=False, brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), brightness_by_max=True, p=0.5),
        A.Cutout(max_h_size=int(params['IMG_SIZE'] * 0.05), max_w_size=int(params['IMG_SIZE'] * 0.05), num_holes=10, p=0.5),
        ToTensorV2(),
    ])

def get_valid_transform():
    return A.Compose([
        A.Resize(params['IMG_SIZE'], params['IMG_SIZE']),
        ToTensorV2(),
    ])


