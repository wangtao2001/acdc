from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
import numpy as np

def train(epoch, model, iterator, optimizer, metric, scheduler, accelerator):
    model.train()
    criterion = CrossEntropyLoss()
    losses = []
    m = []

    with tqdm(total=len(iterator), desc=f'epoch {epoch+1}') as pbar:
        for img, label in iterator:
            # data: (batch, 1, H, W)
            # label: (batch, H, W)
            pbar.update(1)
            hat = model(img)
            # hat:(batch, 4, H, W) 4：类别数
            l = criterion(hat, label.long())
            losses.append(l.item())
            accelerator.backward(l)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            with torch.no_grad():
                pred = torch.argmax(hat, dim=1)
                m.append(metric(label, pred)) # 每个batch下的iou
            
    print(f'epoch: {epoch+1}, train loss: {round(l.item(), 4)}, train mean metric: {np.mean(m)}')
    return losses, m

def test(epoch, model, iterator, metric):
    model.eval()
    m = []
    with torch.no_grad():
        with tqdm(total=len(iterator), desc=f'epoch {epoch+1}') as pbar:
            for img, label in iterator:
                pbar.update(1)
                hat = model(img)
                pred = torch.argmax(hat, dim=1)
                m.append(metric(label, pred))
    print(f'epoch: {epoch+1}, test mean metric: {np.mean(m)}')
    return m

