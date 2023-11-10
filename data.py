import os
import cv2
import torch

class Dataset:
    def __init__(self, data_path, label_path):
        self.data, self.targets = [], []
        for d in os.listdir(data_path):
            self.data.append(torch.tensor(cv2.imread(os.path.join(data_path, d), cv2.IMREAD_GRAYSCALE)))
        for d in os.listdir(label_path):
            self.targets.append(torch.tensor(cv2.imread(os.path.join(label_path, d), cv2.IMREAD_GRAYSCALE)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        return sample, target