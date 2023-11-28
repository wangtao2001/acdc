import numpy as np
import torch

def dice_(y_true, y_pred, smooth=1):  
    intersection = torch.mul(y_true, y_pred).sum()
    return (2 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def batch_dice(y_true, y_pred, classes=4, smooth=1, ):
    # size: B, H, W
    batch_dice = np.zeros((y_true.shape[0], classes-1)) # B, Class
    for i in range(y_true.shape[0]):
        for j in range(1, classes):
            t = torch.where(y_true[i,:,:] ==j, 1, 0)
            p = torch.where(y_pred[i,:,:] ==j, 1, 0)
            batch_dice[i][j-1] = dice_(t, p, smooth)
    return np.mean(batch_dice, axis=0).tolist()