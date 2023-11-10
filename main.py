from data import Dataset
from torch.utils.data import DataLoader
import torch
from run import train
import os
from model import UNet


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data = Dataset('dataset/train/data', 'dataset/train/label')
train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
lr = 0.01

model = UNet()
train(1, model, train_data_loader, None, None, device)