import torch
import numpy as np
import random
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler, SubsetRandomSampler
from torch.utils.data import Subset
from torch_geometric.datasets import QM9
from torch.utils.data import Subset
from tqdm import tqdm
import torch.nn as nn
from models.painn_model import FinalModel as PaiNN
from pretrained_model import load_pretrained_model_dimenet, predict_with_pretrained
from utils import set_seed, split_dataset
from entalpic_al import HOME, TARGET
from utils import train, test
from tqdm import tqdm
from sklearn.cluster import KMeans

def mc_dropout_std(model, loader, device, num_samples=10):
    model.train()
    predictions = []
    for _ in range(num_samples):
        preds = []
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data.z, data.pos, data.batch)
            preds.append(pred)
        predictions.append(torch.cat(preds))
    predictions = torch.stack(predictions)
    return predictions.std(dim=0)

def expected_improvement(model, loader, device, best_f):
    model.eval()
    eis = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.z, data.pos, data.batch)
        improvement = best_f-pred
        sigma = 0.1
        z = improvement/sigma
        ei = improvement*(0.5+0.5*torch.erf(z/2**0.5))+sigma*(1/(2*np.pi)**0.5)*(torch.exp(-0.5*z**2))
        eis.append(ei)
    return torch.cat(eis)

def badge_sampling(model, loader, device, k):
    model.eval()  # Set model to evaluation mode
    gradients = []
    for data in loader:
        data = data.to(device)
        model.zero_grad()
        output = model(data.z, data.pos, data.batch)
        
        grad_embeddings = torch.autograd.grad(output.sum(), model.out[-1].weight, retain_graph=True)[0]
        grad_embeddings = grad_embeddings.view(-1)
        grad_embeddings = grad_embeddings.repeat(output.size(0), 1)
        
        gradients.append(grad_embeddings)
    
    gradients = torch.cat(gradients, dim=0)
    gradients = gradients / torch.norm(gradients, dim=1, keepdim=True)
    
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
    kmeans.fit(gradients.cpu().numpy())
    
    centers = torch.from_numpy(kmeans.cluster_centers_).to(device)
    distances = torch.cdist(gradients, centers)
    indices = distances.argmin(dim=0)
    
    return indices.tolist()

def random_acquisition(unlabeled_indices, k):
    return random.sample(unlabeled_indices, k)

def active_learning_loop(args, device):
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
    
    if not args.use_al_true_labels:
        pretrained_model = load_pretrained_model_dimenet(QM9(HOME))
        pretrained_model.to(device)
        full_loader = DataLoader(full_dataset, batch_size=args.batch_size)
        all_labels = predict_with_pretrained(pretrained_model, full_loader, device)
    else:
        all_labels = full_dataset._data.y.squeeze()
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, labels):
            self.dataset = dataset
            self.labels = labels
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            data = self.dataset[idx]
            data.y = self.labels[idx].unsqueeze(0)
            return data
    custom_dataset = CustomDataset(full_dataset, all_labels)
    
    model = PaiNN(hc=128, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_mae = float("inf")
    
    for al_iteration in range(args.al_iterations):
        print(f"Active Learning Iteration: {al_iteration+1}")
        
        train_dataset = torch.utils.data.Subset(custom_dataset, list(labeled_indices))
        val_dataset = torch.utils.data.Subset(custom_dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
        
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, device)
            val_mae = test(model, val_loader, device)
            
            scheduler.step(val_mae)
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}')
        
        best_val_mae = min(val_mae, best_val_mae)
        
        # Sample from unlabeled data
        unlabeled_dataset = torch.utils.data.Subset(custom_dataset, list(unlabeled_indices))
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size)
        
        if args.acquisition == 'uncertainty':
            uncertainties = mc_dropout_std(model, unlabeled_loader, device)
            new_indices = uncertainties.argsort(descending=True)[:args.points_per_iter]
        elif args.acquisition == 'ei':
            eis = expected_improvement(model, unlabeled_loader, device, best_val_mae)
            new_indices = eis.argsort(descending=True)[:args.points_per_iter]
        elif args.acquisition == 'badge':
            new_indices = badge_sampling(model, unlabeled_loader, device, args.points_per_iter)
        elif args.acquisition == 'random':
            new_indices = random_acquisition(list(unlabeled_indices), args.points_per_iter)
        else:
            raise ValueError("Invalid Acquisition Function")
        
        if args.acquisition != 'random':
            new_indices = [list(unlabeled_indices)[i] for i in new_indices]
        labeled_indices.update(new_indices)
        unlabeled_indices = unlabeled_indices - set(new_indices)

    # Final evaluation
    if not args.use_al_true_labels:
        test_indices = list(unlabeled_indices)
        test_dataset = torch.utils.data.Subset(custom_dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        test_mae_pretrained = test(model, test_loader, device)
        print(f'Final Test MAE (Pretrained labels): {test_mae_pretrained:.4f}')
        
        actual_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        actual_loader = DataLoader(actual_dataset, batch_size=args.batch_size)
        test_mae_actual = test(model, actual_loader, device)
        print(f'Final Test MAE (Actual QM9 labels): {test_mae_actual:.4f}')
        
        return model, test_mae_pretrained, test_mae_actual
    
    else:
        test_indices = list(unlabeled_indices)
        test_dataset = torch.utils.data.Subset(custom_dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
        test_mae = test(model, test_loader, device)
        print(f'Final Test MAE: {test_mae:.4f}')
        return model, test_mae