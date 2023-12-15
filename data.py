import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
from matplotlib import pyplot as plt

seed = torch.random.seed()

def ACDCDataset(dataset_path, num=0.9, batch_size=4):
    root = glob.glob(dataset_path)
    imgs_path = []
    labels_path = []

    for i in root:
        for j in glob.glob(i + '/*'):
            if j.endswith('_frame01.nii.gz'):
                imgs_path.append(j)
            elif j.endswith('_frame01_gt.nii.gz'):
                labels_path.append(j)
    imgs = []
    labels = []

    for img, label in zip(imgs_path, labels_path):
        img, label = nib.load(img), nib.load(label)
        img, label = img.dataobj, label.dataobj
        for i in range(img.shape[-1]): # 最后一个维度是slice
            # 去除无病灶和分类数超过四的数据
            if np.max(label[:, :, i]) > 0 and np.max(label[:, :, i]) < 5:
                imgs.append(img[:, :, i])
                labels.append(label[:, :, i])

    # 切分训练集，测试集
    state = np.random.get_state()
    np.random.shuffle(imgs)
    np.random.set_state(state)
    np.random.shuffle(labels) # 保证同时打乱但对应关系不变
    s = int(len(imgs) * num)

    train_imgs = imgs[:s]
    train_labels = labels[:s]

    test_imgs = imgs[s:]
    test_labels = labels[s:]

    non_transforms = transforms.Compose([
        transforms.CenterCrop((128, 128))
    ])

    train_loader = ACDCData(train_imgs, train_labels, non_transforms)
    test_loader = ACDCData(test_imgs, test_labels, non_transforms)

    # 数据增强
    change_transforms = transforms.Compose([
        transforms.CenterCrop((128, 128)),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomVerticalFlip(1)

    ])
    enhance_loader = ACDCData(train_imgs, train_labels, change_transforms)
    train_loader.imgs.extend(enhance_loader.imgs) # 将增强后的数据也添加进来
    train_loader.labels.extend(enhance_loader.labels)

    train_data = DataLoader(train_loader, batch_size, shuffle=True)
    test_data = DataLoader(test_loader, batch_size, shuffle=True)

    return train_data, test_data


class ACDCData(Dataset):
    def __init__(self, imgs, labels, transforms):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms
        # self.idx = {0: 0, 1: 85, 2: 170, 3: 255} # 没有将label的值均化到0-255所以这一步是不必要的

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img_tensor = torch.tensor(img, dtype=torch.float)
        torch.random.manual_seed(seed)
        img_tensor = self.transforms(img_tensor)
        label_tensor = torch.tensor(label)
        torch.random.manual_seed(seed)
        label_tensor = self.transforms(label_tensor)

        return torch.unsqueeze(img_tensor, 0), label_tensor

    def __len__(self):
        return len(self.imgs)