from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
from metric import batch_dice, iou_mean
import numpy as np

def train(epoch, model, iterator, optimizer, device):
    model = model.to(device)
    model.train()
    criterion = CrossEntropyLoss()
    losses = []
    ious = []

    with tqdm(total=len(iterator), desc=f'epoch {epoch}') as pbar:
        for data, label in iterator:
            # data: (batch, 1, H, W)
            # label: (batch, H, W)
            data, label = data.to(device), label.to(device)
            pbar.update(1)
            hat = model(data)
            # hat:(batch, 4, H, W) 4：类别数
            l = criterion(hat, label.long())
            losses.append(l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(hat, dim=1)
                ious.append(iou_mean(pred, label)) # 每个batch下的iou
            
    print(f'epoch: {epoch}, train loss: {round(l.item(), 4)}, train mean iou: {np.mean(ious)}')
    return losses, ious

def test(epoch, model, iterator, device):
    model = model.to(device)
    model.eval()
    ious = []
    with torch.no_grad():
        with tqdm(total=len(iterator), desc=f'epoch {epoch}') as pbar:
            for data, label in iterator:
                data, label = data.to(device), label.to(device)
                pbar.update(1)
                hat = model(data)
                pred = torch.argmax(hat, dim=1)
                ious.append(iou_mean(label, pred))
    print(f'epoch: {epoch}, test mean iou: {np.mean(ious)}')
    return ious

