"""
This script performs a sanity check on the DimeNetPlusPlus model for QM9 dataset.
It compares the performance of a pretrained model with a randomly initialized one.
"""
import argparse
from typing import List, Tuple

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus
from tqdm import tqdm

from entalpic_al import HOME, TARGET

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing batch_size and num_batches.
    """
    parser = argparse.ArgumentParser(description="Sanity check for DimeNetPlusPlus on QM9 dataset")
    parser.add_argument("--batch_size", type=int, default=32, help = "Batch size for the inference loop")
    parser.add_argument("--num_batches", type=int, default=32, help = "Number of batches to test on")
    return parser.parse_args()

def load_dataset() -> QM9:
    """Load and preprocess the QM9 dataset.

    Returns:
        QM9: Preprocessed QM9 dataset.
    """
    dataset = QM9(HOME)
    # Select specific target properties
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset.data.y = dataset.data.y[:, idx]
    return dataset

def load_model_data(dataset:QM9) -> Tuple[DimeNetPlusPlus, List]:
    """Load the pretrained DimeNetPlusPlus model and datasets.

    Args:
        dataset (QM9): The QM9 dataset.

    Returns:
        Tuple[DimeNetPlusPlus, List]: Pretrained model and list of datasets (train, val, test).
    """
    model, datasets = DimeNetPlusPlus.from_qm9_pretrained(HOME, dataset, TARGET)
    return model, datasets

def eval_model(model: DimeNetPlusPlus, loader:DataLoader, num_batches:int, device:torch.device) -> float:
    """Evaluate the model and compute mean absolute error.

    Args:
        model (DimeNetPlusPlus): The model to evaluate.
        loader (DataLoader): DataLoader for the dataset.
        num_batches (int): Number of batches to evaluate.
        device (torch.device): Device to run the evaluation on.

    Returns:
        float: Mean Absolute Error (MAE) of the model predictions.
    """
    mae_list = []
    for d, data in enumerate(tqdm(loader, total=num_batches)):
        if d ==  num_batches:
            break
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.z, data.pos, data.batch)
        mae = (pred.view(-1) - data.y[:, TARGET]).abs()
        mae_list.append(mae)
    return torch.cat(mae_list, dim=0).mean().item()

def main():
    
    args = parse_arguments()
    print(f"Using  batch size: {args.batch_size}")
    
    dataset = load_dataset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pre-trained model
    model, (_, _, test_dataset) = load_model_data(dataset)
    model = model.to(device)
    
    loader = DataLoader(test_dataset, batch_size = args.batch_size)
    
    # Evaluate the pre-trained model
    mae_pretrained = eval_model(model, loader, args.num_batches, device)
    print(f"Pretrained MAE: {mae_pretrained: .4f} eV")
    
    # Reset parameters and evaluate the randomly intialized model
    model.reset_parameters()
    mae_random = eval_model(model, loader, args.num_batches, device)
    print(f"Random init MAE: {mae_random: .4f} eV")
    
if __name__ == "__main__":
    main()
    
    