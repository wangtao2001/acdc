from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
from metric import batch_dice
import numpy as np

def train(epoch, model, iterator, optimizer, device):
    model = model.to(device)
    model.train()
    criterion = CrossEntropyLoss()
    losses = []
    dices = []

    with tqdm(total=len(iterator), desc=f'epoch {epoch}') as pbar:
        for data, label in iterator:
            # data:(batch, 1, H, W)
            # label:(batch, H, W)
            data, label = data.to(device), label.to(device)
            pbar.update(1)
            hat = model(data)
            # hat:(batch, 4, H, W) 4：类别数
            l = criterion(hat, label.long())
            losses.append(l.item())
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                pred = torch.argmax(hat, dim=1)
                dices.append(batch_dice(label, pred))
            
    print(f'epoch: {epoch}, loss: {round(l.item(), 4)}, mean dice: {np.mean(dices)}')
    return losses, dices

