import torch
from torch import Tensor
import cv2
import os
import numpy as np


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-8,
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    if input.sum(dim=sum_dim) == 0 and target.sum(dim=sum_dim) == 0:
        dice = 255
    else:
        dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice


out = []
cou = 0
su = 0
path_j = "simple1/val/labels"
path_i = "test_output1"
for i in os.listdir(path_i):
    try:
        x = cv2.imread(os.path.join(path_i, i), cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(os.path.join(path_j, i[:-4] + "_mask.png"), cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error loading image {i}: {e}")
        continue

    if y.shape:
        x = cv2.resize(x, (y.shape[1], y.shape[0]))
    else:
        print(f"Skipping image {i}: empty image")
        continue

    x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)[1]
    y = cv2.threshold(y, 1, 255, cv2.THRESH_BINARY)[1]
    #cv2.imwrite(os.path.join(path_i, i), x)

    out.append(dice_coeff(Tensor(x), Tensor(y)))
    cou += 1

print(np.mean(out))
