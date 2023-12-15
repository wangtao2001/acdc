import numpy as np
import torch

def dice_(y_true, y_pred, smooth=1):  
    intersection = torch.mul(y_true, y_pred).sum()
    return (2 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def batch_dice(y_true, y_pred, classes=4, smooth=1):
    # size: B, H, W
    batch_dice = np.zeros((y_true.shape[0], classes-1)) # B, Class
    for i in range(y_true.shape[0]): # 每个batch
        for j in range(1, classes): # 每个类 Ignore the background class("0")
            t = torch.where(y_true[i,:,:] ==j, 1, 0)
            p = torch.where(y_pred[i,:,:] ==j, 1, 0)
            batch_dice[i][j-1] = dice_(t, p, smooth)
    return np.mean(batch_dice, axis=0).tolist() # (1, classes-1) 表示各类单独的dice

def iou_mean(pred, target, n_classes=4):
    ious = []
    iousSum = 0
    pred = pred.view(-1)
    target = np.array(target.cpu())
    target = torch.from_numpy(target)
    target = target.view(-1)
    # Ignore Iou for background class("0")
    for cls in range(1, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum / (n_classes - 1)