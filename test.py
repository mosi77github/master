import cv2
import os
from attention_unet.im2.model import NestedUNet
from model4 import U_Transformer
from model5 import TransUNet
import torch
import numpy as np
import shutil
from unext import UNext
from torch import nn
from unet.im3.model import build_unet
from attention_unet.im1.model import AttU_Net
from unext import UNext
#from cmu.CMUNet import CMUNet
from albumentations import Normalize
from mymodel.mymodel import CMUNet

folder_to_delete = "test_output1"
if os.path.exists(folder_to_delete):
    shutil.rmtree(folder_to_delete)
os.mkdir("test_output1")

image_mean = [0.38420745, 0.38427339, 0.3842764]
image_std = [0.19626831, 0.19627472, 0.19627159]

checkpoint_path = "files/t99.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#model = CMUNet(3, 1).to(device)
model = AttU_Net(3,1).to(device)
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
path_i1 = "simple1/test/images"
for i in os.listdir(path_i1):
    image0 = cv2.imread(path_i1 + "/" + f"{i}")  ## (512, 512, 3)
    image = cv2.resize(image0, (64,64))

    # image_aug = Normalize(mean=image_mean, std=image_std)
    # x = image_aug(image=image)["image"]

    x = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
    x = x / 255.0

    x = np.expand_dims(x, axis=0)  ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    pred_y = model(x)
    pred_y = torch.sigmoid(pred_y)
    pred_y = pred_y[0].cpu().detach().numpy()  ## (1, 512, 512)
    pred_y = np.squeeze(pred_y, axis=0)  ## (512, 512)
    pred_y = pred_y > 0.5
    pred_y = 255 * np.array(pred_y, dtype=np.uint8)
    pred_y = cv2.resize(pred_y, (image0.shape[1], image0.shape[0]))
    cv2.imwrite("test_output1/" + f"{i}", pred_y)
