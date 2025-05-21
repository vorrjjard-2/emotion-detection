# PyTorch Core
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# NumPy for data handling
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn for metrics
from sklearn.metrics import confusion_matrix, classification_report

# Misc utilities
import random
import os
import time
import copy
from tqdm import tqdm

def line_graph(data_tensors, labels=None, title=None, xlabel="Epochs", ylabel="Value", colors=None):
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(data_tensors))]
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(data_tensors)))  
    
    plt.figure(figsize=(10, 6))
    
    for i, data in enumerate(data_tensors):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()  # safer with .detach()
        plt.plot(data, label=labels[i], color=colors[i], lw=2)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()