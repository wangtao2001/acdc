import numpy as np
import torch
from scipy.spatial.distance import cdist

def _dice(label, pred, smooth=1):
    # 对矩阵计算
    intersection = torch.mul(label, pred).sum()
    return (2 * intersection + smooth) / (label.sum() + pred.sum() + smooth)

def dice_mean(label, pred, classes=4, smooth=1):
    # size: (batch, H, W) 0/1/2/3...
    batch_dice = np.zeros((label.shape[0], classes-1)) # (batch, classes)
    for i in range(label.shape[0]): # 每个batch
        for j in range(1, classes): # 每个类 ignore the background class 0
            t = torch.where(label[i,:,:] ==j, 1, 0)
            p = torch.where(pred[i,:,:] ==j, 1, 0)
            batch_dice[i][j-1] = _dice(t, p, smooth)
    return np.mean(batch_dice)

def iou_mean(label, pred, classes=4):
    ious_sum = 0
    pred = pred.view(-1)
    label = label.view(-1)
    # ignore Iou for background class 0
    for cls in range(1, classes):
        pred_inds = pred == cls
        label_inds = label == cls
        intersection = (pred_inds[label_inds]).sum()
        union = pred_inds.sum() + label_inds.sum() - intersection
        ious_sum += float(intersection) / float(max(union, 1))
    return ious_sum / (classes - 1)

def _directed_hd95(c1, c2):
    if len(c1) == 0 or len(c2) == 0:
        return 0
    dists = cdist(c1, c2)  # 所有点对之间的距离
    max_dists = np.max(dists, axis=1)  # 第一个点集中每个点到第二个点集的最大距离
    return np.percentile(max_dists, 95)  # 取95%分位数

def _hd95(label, pred):
    # 转换非0元素的坐标
    label_axis = np.transpose(np.nonzero(label))
    pred_axis = np.transpose(np.nonzero(pred))
    forward_hd95 = _directed_hd95(label_axis, pred_axis)
    reverse_hd95 = _directed_hd95(pred_axis, label_axis)
    # Hausdorff distance 95 是双向距离的最大值
    return max(forward_hd95, reverse_hd95)


def hd95_mean(label, pred, classes=4):
    batch_hd = np.zeros((label.shape[0], classes-1)) 
    for i in range(label.shape[0]):
        for j in range(1, classes):
            t = torch.where(label[i,:,:] ==j, 1, 0)
            p = torch.where(pred[i,:,:] ==j, 1, 0)
            batch_hd[i][j-1] = _hd95(t.cpu().numpy(), p.cpu().numpy())
    return np.mean(batch_hd)
