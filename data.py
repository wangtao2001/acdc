import os
import cv2
import torch
import numpy as np

class Dataset:
    def __init__(self, data_path, label_path):
        self.data, self.targets = [], []
        for d in os.listdir(data_path):
            img = cv2.normalize(cv2.imread(os.path.join(data_path, d), cv2.IMREAD_GRAYSCALE), None, 0, 1, cv2.NORM_MINMAX)
            self.data.append(torch.tensor(img, dtype=torch.float64))
        for d in os.listdir(label_path):
            img = cv2.normalize(cv2.imread(os.path.join(label_path, d), cv2.IMREAD_GRAYSCALE), None, 0, 1, cv2.NORM_MINMAX)
            self.targets.append(torch.tensor(img, dtype=torch.float64))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        return sample, target