from data import ACDCDataset
from torch.utils.data import DataLoader
import torch
from run import train, test
import os
from models import FCN8s, UNet, UNetPlusPlus
from torch.optim import AdamW
from matplotlib import pyplot as plt
import statsmodels.api as sm
import argparse
from metric import dice_mean, iou_mean, hd95_mean

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    required=True,
    choices=['fcn', 'unet', 'unetpp']
)
parser.add_argument(
    '--metric',
    type=str,
    required=True,
    choices=['dice', 'iou', 'hd95']
)
args = parser.parse_args()

modelset = {
    'fcn': FCN8s, 'unet': UNet, 'unetpp': UNetPlusPlus
}
metrics = {
    'dice': dice_mean, 'iou': iou_mean, 'hd95': hd95_mean
}

smooth = lambda data: sm.nonparametric.lowess (
    data, list(range(len(data))), 0.05
)[:, 1] # 曲线平滑处理

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
train_data_loader, test_data_loader = ACDCDataset("./dataset/training/*", batch_size=batch_size)

model = modelset[args.model]()
optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-6)
train_all_loss = []
train_all_metric = []
test_all_metric = []
for epoch in range(30):
    losses, m = train(epoch, model, train_data_loader, optimizer, metrics[args.metric], device)
    train_all_loss.extend(losses)
    train_all_metric.extend(m)
    test_all_metric.extend(test(epoch, model, test_data_loader, metrics[args.metric], device))


torch.save(model, f'models/model-{args.model}-{losses[-1]:.6f}.pt')

plt.plot(smooth(train_all_loss))
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('img/train/loss.png')
plt.clf()
plt.plot(smooth(train_all_metric))
plt.xlabel('step')
plt.ylabel(f'train mean {args.metric}')
plt.savefig(f'img/train/{args.metric}.png')
plt.clf()
plt.plot(smooth(test_all_metric))
plt.xlabel('step')
plt.ylabel(f'test mean {args.metric}')
plt.savefig(f'img/test/{args.metric}.png')
plt.close()