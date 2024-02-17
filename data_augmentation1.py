import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import kornia.augmentation as K
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    CenterCrop,
    Affine,
    RandomBrightnessContrast,
)


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, augment):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.augment = augment

    def __getitem__(self, index):
        """Reading image"""
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        size = (64, 64)
        if self.augment == True:
            aug = HorizontalFlip(p=0.5)
            augmented = aug(image=image, mask=mask)
            x = augmented["image"]
            y = augmented["mask"]

            aug = VerticalFlip(p=0.5)
            augmented = aug(image=x, mask=y)
            x = augmented["image"]
            y = augmented["mask"]

            aug = Rotate(limit=5, p=0.3)
            augmented = aug(image=x, mask=y)
            x = augmented["image"]
            y = augmented["mask"]

            aug = Affine(translate_px={"x": (1, 10), "y": (1, 10)}, p=0.3)
            augmented = aug(image=x, mask=y)
            x = augmented["image"]
            y = augmented["mask"]

            # aug = RandomBrightnessContrast(
            #   brightness_limit=0.05, contrast_limit=0.05, p=0.5
            # )
            # augmented = aug(image=x, mask=y)
            # x = augmented["image"]
            # y = augmented["mask"]
            #
            # transform = transforms.ToTensor()
            # tensor = transform(x)
            # tensor1 = transform(y)
            # aug = K.RandomElasticTransform(
            #   kernel_size=(31, 31), p=0.5, alpha=(0.4, 0.4), sigma=(10, 10)
            # )
            # out = aug(tensor)
            # out1 = aug(tensor1, params=aug._params)
            # np_arr = out.detach().cpu().numpy().squeeze()
            # np_arr1 = out1.detach().cpu().numpy().squeeze()
            # np_arr = np.transpose(np_arr, (1, 2, 0))
            # np_arr = 255 * np_arr
            # np_arr1 = 255 * np_arr1
            # img = np.array(np_arr).astype(np.uint8)
            # img1 = np.array(np_arr1).astype(np.uint8)

            # cv2.imshow('a', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # image = cv2.resize(img, size)
            # mask = cv2.resize(img1, size)
            #
            # aug = CenterCrop(height=size[0] - 4, width=size[0] - 4, p=0)
            # augmented = aug(image=image, mask=mask)
            # image = augmented["image"]
            # mask = augmented["mask"]
            #
            # image = cv2.resize(image, size)
            # mask = cv2.resize(mask, size)

            image = cv2.resize(x, size)
            mask = cv2.resize(y, size)

            image = image / 255.0  ## (512, 512, 3)
            image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            mask = mask / 255.0  ## (512, 512)
            mask = np.expand_dims(mask, axis=0)  ## (1, 512, 512)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
        else:
            image = cv2.resize(image, size)
            mask = cv2.resize(mask, size)
            image = image / 255.0  ## (512, 512, 3)
            image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            mask = mask / 255.0  ## (512, 512)
            mask = np.expand_dims(mask, axis=0)  ## (1, 512, 512)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
