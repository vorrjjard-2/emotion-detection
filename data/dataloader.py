import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary

class FAR2013Data(Dataset):
    def __init__(self, X, y, transform=None):
        self.features = X 
        self.label = y  
        self.transform = transform 

    def __len__(self):
        return len(self.features) 
        
    def __getitem__(self, idx):
        sample = self.features[idx].reshape(48, 48)
        sample = Image.fromarray(sample.astype(np.uint8), mode='L')
        label = self.label[idx]
        
        if self.transform:
            sample = self.transform(sample)

        assert isinstance(sample, torch.Tensor), f"Expected torch.Tensor, got {type(sample)}"
        assert sample.shape == (1, 48, 48), f"Unexpected shape: {sample.shape}"
        
        return sample, label
    
def load_dataloader(path, split, batch_size, shuffle=True, transform=None):
    df = pd.read_csv(path)
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    X = np.stack(pixels.to_numpy())
    y = df['emotion'].to_numpy()

    if split == 'train':
        train_idx = df['Usage'] == 'Training'
        X = X[train_idx]
        y = y[train_idx]
    elif split == 'val':
        val_idx = df['Usage'] == 'PublicTest'
        X = X[val_idx]
        y = y[val_idx]
    else:
        test_idx = df['Usage'] == 'PrivateTest'
        X = X[test_idx]
        y = y[test_idx]

    dataset = FAR2013Data(X, y, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader