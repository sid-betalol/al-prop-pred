import random
import numpy as np
import torch
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Subset

def train_al(model, loader, optimizer, device, label_dict, indices):
    model.train()
    total_loss = 0
    for i, data in enumerate(tqdm(loader, desc="Training")):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.z, data.pos, data.batch)
        batch_indices = indices[i * loader.batch_size:(i + 1) * loader.batch_size]
        y = torch.stack([label_dict[idx] for idx in batch_indices]).to(device)
        loss = nn.MSELoss()(out, y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test_al(model, loader, device, label_dict, indices):
    model.eval()
    total_mae = 0
    for i, data in enumerate(loader):
        data = data.to(device)
        with torch.no_grad():
            out = model(data.z, data.pos, data.batch)
        batch_indices = indices[i * loader.batch_size:(i + 1) * loader.batch_size]
        y = torch.stack([label_dict[idx] for idx in batch_indices]).to(device)
        total_mae += (out - y.squeeze()).abs().sum().item()
    return total_mae / len(loader.dataset)

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.z, data.pos, data.batch)
        loss = nn.MSELoss()(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    total_mae = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.z, data.pos, data.batch)
        total_mae += (out - data.y.squeeze()).abs().sum().item()
    return total_mae / len(loader.dataset)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        best_filename = 'best_model.pth'
        torch.save(state, best_filename)

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, return_indices=False):
    if isinstance(dataset, Subset):
        indices = dataset.indices
        original_dataset = dataset.dataset
    else:
        indices = list(range(len(dataset)))
        original_dataset = dataset

    num_samples = len(indices)
    np.random.shuffle(indices)
    
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    if return_indices:
        return train_indices, val_indices, test_indices
    else:
        train_dataset = Subset(original_dataset, train_indices)
        val_dataset = Subset(original_dataset, val_indices)
        test_dataset = Subset(original_dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset