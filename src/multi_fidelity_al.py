import torch
from torch_geometric.nn import SchNet
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from models.painn_model import FinalModel as PaiNN
from pretrained_model import load_pretrained_model_dimenet, load_pretrained_model_schnet, predict_with_pretrained
from utils import set_seed, train, test
from entalpic_al import HOME, TARGET
import random


def multi_fidelity_acquisition(low_fidelity_model, high_fidelity_model, unlabeled_loader, device, k):
    
    low_preds = predict_with_pretrained(low_fidelity_model, unlabeled_loader, device)
    high_preds = predict_with_pretrained(high_fidelity_model, unlabeled_loader, device)
    
    disagreement = torch.abs(low_preds - high_preds)
    
    return disagreement.argsort(descending=True)[:k]

def multi_fidelity_active_learning_loop(args, device):
    set_seed(args.seed)
    
    dataset = QM9(HOME)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset._data.y = dataset._data.y[:, idx]
    full_dataset = dataset[-10000:].copy()
    full_dataset._data.y = full_dataset._data.y[:, [TARGET]]
    
    initial_indices = random.sample(range(len(full_dataset)), 1000)
    labeled_indices = set(initial_indices)
    unlabeled_indices = set(range(len(full_dataset))) - labeled_indices
    
    val_indices = random.sample(list(unlabeled_indices), 200)
    unlabeled_indices -= set(val_indices)
    
    high_fidelity_model = load_pretrained_model_dimenet(QM9(HOME))
    high_fidelity_model.to(device)
    
    low_fidelity_model = load_pretrained_model_schnet(QM9(HOME))
    low_fidelity_model.to(device)
    
    model = PaiNN(hc=128, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    for al_iteration in range(args.al_iterations):
        print(f"Active Learning Iteration: {al_iteration+1}")
        
        train_dataset = Subset(full_dataset, list(labeled_indices))
        val_dataset = Subset(full_dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, device)
            val_mae = test(model, val_loader, device)
            
            scheduler.step(val_mae)
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}')
        
        unlabeled_dataset = Subset(full_dataset, list(unlabeled_indices))
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size)
        
        new_indices = multi_fidelity_acquisition(low_fidelity_model, high_fidelity_model, unlabeled_loader, device, args.points_per_iter)
        
        new_indices = [list(unlabeled_indices)[i] for i in new_indices]
        labeled_indices.update(new_indices)
        unlabeled_indices = unlabeled_indices - set(new_indices)

    test_indices = list(unlabeled_indices)
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    test_mae = test(model, test_loader, device)
    print(f'Final Test MAE: {test_mae:.4f}')
    return model, test_mae