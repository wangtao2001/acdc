from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
from metric import dice_mean, iou_mean
import numpy as np

def train(epoch, model, iterator, optimizer, device):
    model = model.to(device)
    model.train()
    criterion = CrossEntropyLoss()
    losses = []
    dices = []

    with tqdm(total=len(iterator), desc=f'epoch {epoch}') as pbar:
        for img, label in iterator:
            # data: (batch, 1, H, W)
            # label: (batch, H, W)
            img, label = img.to(device), label.to(device)
            pbar.update(1)
            hat = model(img)
            # hat:(batch, 4, H, W) 4：类别数
            l = criterion(hat, label.long())
            losses.append(l.item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(hat, dim=1)
                dices.append(dice_mean(label, pred)) # 每个batch下的iou
            
    print(f'epoch: {epoch}, train loss: {round(l.item(), 4)}, train mean iou: {np.mean(dices)}')
    return losses, dices

def test(epoch, model, iterator, device):
    model = model.to(device)
    model.eval()
    dices = []
    with torch.no_grad():
        with tqdm(total=len(iterator), desc=f'epoch {epoch}') as pbar:
            for img, label in iterator:
                img, label = img.to(device), label.to(device)
                pbar.update(1)
                hat = model(img)
                pred = torch.argmax(hat, dim=1)
                dices.append(dice_mean(label, pred))
    print(f'epoch: {epoch}, test mean iou: {np.mean(dices)}')
    return dices

