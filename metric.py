import numpy as np
import torch

def dice_(label, pred, smooth=1):  
    intersection = torch.mul(label, pred).sum()
    return (2 * intersection + smooth) / (label.sum() + pred.sum() + smooth)

def dice_mean(label, pred, classes=4, smooth=1):
    # size: (batch, H, W)
    batch_dice = np.zeros((label.shape[0], classes-1)) # (batch, classes)
    for i in range(label.shape[0]): # 每个batch
        for j in range(1, classes): # 每个类 ignore the background class 0
            t = torch.where(label[i,:,:] ==j, 1, 0)
            p = torch.where(pred[i,:,:] ==j, 1, 0)
            batch_dice[i][j-1] = dice_(t, p, smooth)
    return np.mean(batch_dice)

def iou_mean(label, pred, classes=4):
    ious_sum = 0
    pred = pred.view(-1)
    label = np.array(label.cpu())
    label = torch.tensor(label)
    label = label.view(-1)
    # ignore Iou for background class 0
    for cls in range(1, classes):
        pred_inds = pred == cls
        label_inds = label == cls
        intersection = (pred_inds[label_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + label_inds.long().sum().data.cpu().item() - intersection
        ious_sum += float(intersection) / float(max(union, 1))
    return ious_sum / (classes - 1)