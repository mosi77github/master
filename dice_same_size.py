import torch
from torch import Tensor
import cv2
import os
import numpy as np


def dice_coeff(
    inputs: Tensor,
    targets: Tensor,
    reduce_batch_first: bool = False,
    smooth: float = 1e-8,
):
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice


out = []
cou = 0
su = 0
path_j = "simple1/test/labels"
path_i = "test_output1"
size = 64
for i in os.listdir(path_i):

    x = cv2.imread(os.path.join(path_i, i), cv2.IMREAD_GRAYSCALE)
    y = cv2.imread(os.path.join(path_j, i[:-4] + "_mask.png"), cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (size, size))
    x = cv2.resize(x, (size, size))

    x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)[1]
    y = cv2.threshold(y, 1, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite(os.path.join(path_i, i), x)

    out.append(dice_coeff(Tensor(x), Tensor(y)))
    cou += 1

print(np.mean(out))
