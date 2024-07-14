import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomNodeSplit
from tqdm import tqdm
import os
import numpy as np

from models.painn_model import FinalModel as PaiNN
from entalpic_al import HOME, TARGET

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        best_filename = 'best_model.pth'
        torch.save(state, best_filename)
        
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    num_graphs = len(dataset)
    indices = np.random.permutation(num_graphs)
    
    train_size = int(num_graphs * train_ratio)
    val_size = int(num_graphs * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]
    
    return train_dataset, val_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = QM9(HOME)
    data_set = dataset[-1000:]
    data_set.data.y = data_set.data.y[:, [TARGET]]
    print(f"Target Range: {data_set.data.y.min().item()} to {data_set.data.y.max().item()}")

    train_dataset, val_dataset, test_dataset = split_dataset(data_set)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = PaiNN(hc=128, num_layers=4).to(device)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_mae = float('inf')
    smoothed_val_loss = None
    smoothing_factor = 0.9
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_val_mae = checkpoint['best_val_mae']
            smoothed_val_loss = checkpoint['smoothed_val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_mae = test(model, val_loader, device)
        
        if smoothed_val_loss is None:
            smoothed_val_loss = val_mae
        else:
            smoothed_val_loss = smoothing_factor * smoothed_val_loss + (1 - smoothing_factor) * val_mae
        
        scheduler.step(smoothed_val_loss)
        
        is_best = val_mae < best_val_mae
        best_val_mae = min(val_mae, best_val_mae)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_mae': best_val_mae,
            'smoothed_val_loss': smoothed_val_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}, Smoothed Val MAE: {smoothed_val_loss:.4f}')

    # Load the best model and evaluate on test set
    best_checkpoint = torch.load('best_model.pth')
    model.load_state_dict(best_checkpoint['state_dict'])
    test_mae = test(model, test_loader, device)
    print(f'Test MAE: {test_mae:.4f}')

if __name__ == '__main__':
    main()