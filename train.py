import time
from glob import glob
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_augmentation1 import DriveDataset
import cv2
# from cmu.CMUNet import CMUNet
from mymodel.mymodel import CMUNet
from torch import Tensor
import os
import numpy as np
from loss import DiceBCELoss, DiceScore
from utils import seeding, create_dir, epoch_time
from unet.im3.model import build_unet
from attention_unet.im1.model import AttU_Net


size = 64

def save_loss_plot(train_losses, valid_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        # print(x.shape)
        y_pred = model(x)
        # print(y_pred.shape)
        y_pred = torch.sigmoid(y_pred)
        # y_pred = model(x)['out']
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, Score, device):
    epoch_loss = 0.0
    # score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)
            # y_pred = model(x)['out']
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            # sc = Score(y_pred, y)
            # score += sc.item()
        epoch_loss = epoch_loss / len(loader)
        # score = score / len(loader)
    return epoch_loss


def dice_coeff(
    inputs: Tensor,
    targets: Tensor,
    smooth: float = 1e-8,
):
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice


def find_best_model():
    global size
    model.eval()
    out = []
    cou = 0
    path_j = "simple1/val/labels"
    path_i = "test_output"
    path_i1 = "simple1/val/images"
    for i in os.listdir(path_i1):
        image0 = cv2.imread(path_i1 + "/" + f"{i}")
        image = cv2.resize(image0, (size, size))
        x = np.transpose(image, (2, 0, 1))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().detach().numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = pred_y > 0.5
        pred_y = 255 * np.array(pred_y, dtype=np.uint8)
        pred_y = cv2.resize(pred_y, (image0.shape[1], image0.shape[0]))
        cv2.imwrite("test_output/" + f"{i}", pred_y)

    for i in os.listdir(path_i):
        x = cv2.imread(os.path.join(path_i, i), cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(os.path.join(path_j, i[:-4] + "_mask.png"), cv2.IMREAD_GRAYSCALE)
        y = cv2.resize(y, (size, size))
        x = cv2.resize(x, (size, size))
        x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)[1]
        y = cv2.threshold(y, 1, 255, cv2.THRESH_BINARY)[1]
        out.append(dice_coeff(Tensor(x), Tensor(y)))
        cou += 1

    print(np.mean(out))
    return np.mean(out)


if __name__ == "__main__":
    """Seeding"""
    seeding(313)

    """ Directories """
    create_dir("files")
    for ii in range(1):
        if ii == 0:
            """Load dataset"""
            train_x = sorted(glob("simple1/train/images/*"))
            train_y = sorted(glob("simple1/train/labels/*"))
            valid_x = sorted(glob("simple1/val/images/*"))
            valid_y = sorted(glob("simple1/val/labels/*"))
            checkpoint_path = "files/t1.pth"
        if ii == 1:

            """Load dataset"""
            train_x = sorted(glob("two_stage_p1_data/images/train2/*"))
            train_y = sorted(glob("two_stage_p1_data/labels/train2/*"))
            valid_x = sorted(glob("two_stage_p1_data/images/val2/*"))
            valid_y = sorted(glob("two_stage_p1_data/labels/val2/*"))
            checkpoint_path = "files/t2.pth"
        if ii == 2:
            """Load dataset"""
            train_x = sorted(glob("two_stage_p1_data/images/train3/*"))
            train_y = sorted(glob("two_stage_p1_data/labels/train3/*"))
            valid_x = sorted(glob("two_stage_p1_data/images/val3/*"))
            valid_y = sorted(glob("two_stage_p1_data/labels/val3/*"))
            checkpoint_path = "files/t3.pth"
        if ii == 3:

            """Load dataset"""
            train_x = sorted(glob("two_stage_p1_data/images/train4/*"))
            train_y = sorted(glob("two_stage_p1_data/labels/train4/*"))
            valid_x = sorted(glob("two_stage_p1_data/images/val4/*"))
            valid_y = sorted(glob("two_stage_p1_data/labels/val4/*"))
            checkpoint_path = "files/t4.pth"

        data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
        print(data_str)

        """ Hyperparameters """
        batch_size = 8
        num_epochs = 100
        lr = 0.0001
        weight_decay = 0.0001

        """ Dataset and loader """

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        print(device)
        # model = CustomDeepLab(num_classes=1)
        load_pretrained_model = 0
        checkpoint_path = "files/t1.pth"
        model = AttU_Net(3,1).to(device)
        #model = CMUNet(3, 1).to(device)
        if load_pretrained_model == 1:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.9, patience=10, verbose=True, min_lr=0.00001
        )
        loss_fn = DiceBCELoss()
        Score = DiceScore()

        """ Training the model """
        best_valid_loss = float("inf")
        best_valid_score = 0
        train_losses = []
        valid_losses = []
        try:
            for epoch in range(num_epochs):
                checkpoint_path = f"files/t{epoch}.pth"
                start_time = time.time()
                train_dataset = DriveDataset(train_x, train_y, True)
                valid_dataset = DriveDataset(valid_x, valid_y, False)
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                )

                valid_loader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                print(
                    f"Learning rate for epoch {epoch + 1}: {optimizer.param_groups[0]['lr']}"
                )
                train_loss = train(model, train_loader, optimizer, loss_fn, device)
                valid_loss = evaluate(model, valid_loader, loss_fn, Score, device)
                score = find_best_model()
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                scheduler.step(valid_loss)

                """ Saving the model """
                if best_valid_score < score:
                    best_valid_score = score
                    torch.save(model.state_dict(), checkpoint_path)

                if valid_loss < best_valid_loss:
                    data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
                    print(data_str)

                    best_valid_loss = valid_loss
                    # torch.save(model.state_dict(), checkpoint_path)

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                data_str = (
                    f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
                )
                data_str += f"\tTrain Loss: {train_loss:.3f}\n"
                data_str += f"\t Val. Loss: {valid_loss:.3f}\n"
                print(data_str)
        except KeyboardInterrupt:
            print("Training interrupted. Saving loss plot...")
            save_loss_plot(train_losses, valid_losses, "loss_plot.jpg")
            raise KeyboardInterrupt

        print("Training completed. Saving loss plot...")
        save_loss_plot(train_losses, valid_losses, "loss_plot.jpg")
        print("Loss plot saved successfully.")
