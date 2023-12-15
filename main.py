from data import ACDCDataset
from torch.utils.data import DataLoader
import torch
from run import train, test
import os
from model import UNet
from torch.optim import AdamW
from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
smooth = lambda data: sm.nonparametric.lowess (
    data, list(range(len(data))), 0.05
)[:, 1] # 曲线平滑处理

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data_loader, test_data_loader = ACDCDataset("./acdc_challenge_20170617/training/*")

model = UNet()
optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-6)
train_all_loss = []
train_all_iou = []
test_all_iou = []
for epoch in range(30):
    losses, ious = train(epoch, model, train_data_loader, optimizer, device)
    train_all_loss.extend(losses)
    train_all_iou.extend(ious)
    test_all_iou.extend(test(epoch, model, test_data_loader, device))


torch.save(model, 'models/model.pt')

plt.plot(smooth(train_all_loss))
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('img/train/loss.png')
plt.clf()
plt.plot(smooth(train_all_iou))
plt.xlabel('step')
plt.ylabel('train mean iou')
plt.savefig('img/train/iou.png')
plt.clf()
plt.plot(smooth(test_all_iou))
plt.xlabel('step')
plt.ylabel('test mean iou')
plt.savefig('img/test/iou.png')
plt.close()