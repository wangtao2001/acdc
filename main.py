from data import ACDCDataset
import torch
from run import train, test
import os
from models import FCN8s, UNet, UNetPlusPlus
from torch.optim import AdamW
from matplotlib import pyplot as plt
import statsmodels.api as sm
import argparse
from metric import dice_mean, iou_mean, hd95_mean
from transformers import get_linear_schedule_with_warmup

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
train_iterator, test_iterator = ACDCDataset("./dataset/training/*", batch_size=batch_size)
epochs = 30
total_steps = len(train_iterator) * epochs # 总步数
warm_up_ratio = 0.1 # 预热10%
model = modelset[args.model]()
optimizer = AdamW(model.parameters(), lr=1e-3, eps=1e-6)
# 线性学习率预热
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*warm_up_ratio, num_training_steps=total_steps)

train_all_loss = []
train_all_metric = []
test_all_metric = []
for epoch in range(epochs):
    losses, m = train(epoch, model, train_iterator, optimizer, metrics[args.metric], scheduler ,device)
    train_all_loss.extend(losses)
    train_all_metric.extend(m)
    test_all_metric.extend(test(epoch, model, test_iterator, metrics[args.metric], device))


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