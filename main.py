from data import Dataset
from torch.utils.data import DataLoader
import torch
from run import train
import os
from model import UNet
from torch.optim import AdamW
from matplotlib import pyplot as plt
import statsmodels.api as sm
smooth = lambda data: sm.nonparametric.lowess (
    data, list(range(len(data))), 0.05
)[:, 1] # 曲线平滑处理

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data = Dataset('dataset/train/data', 'dataset/train/label')
train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)

model = UNet()
optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-6)
all_loss = []
all_dice = []
for epoch in range(50):
    losses, dices = train(epoch, model, train_data_loader, optimizer, device)
    all_loss.extend(losses)
    all_dice.extend(dices)  # step, 3

torch.save(model, 'models/model.pt')

plt.plot(smooth(all_loss))
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('img/loss.png')
plt.clf()
for d in range(3):
    plt.plot(smooth([a[d] for a in all_dice]))
plt.savefig('img/dice.png')
plt.close()