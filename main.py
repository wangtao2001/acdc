from data import Dataset
from torch.utils.data import DataLoader

train_data = Dataset('dataset/train/data', 'dataset/train/label')
data_loader = DataLoader(train_data, batch_size=1, shuffle=True)