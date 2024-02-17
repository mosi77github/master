import numpy as np
import os
import cv2
from torch import Tensor


def calculate_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt)
    union = np.logical_or(mask_pred, mask_gt)
    if np.sum(union) == 0 and np.sum(mask_pred) == 0 and np.sum(mask_gt) == 0:
        iou = 1
    else:
        iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    return iou


out = []
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
    iou = calculate_iou(np.array(Tensor(x)), np.array(Tensor(y)))
    out.append(iou)

mean_iou = np.mean(out)
print(f"Mean IoU: {mean_iou}")
