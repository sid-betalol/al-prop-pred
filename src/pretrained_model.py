import torch
from torch_geometric.nn import DimeNetPlusPlus
from entalpic_al import HOME, TARGET

def load_pretrained_model(dataset):
    model, _ = DimeNetPlusPlus.from_qm9_pretrained(HOME, dataset, TARGET)
    return model

def predict_with_pretrained(model, loader, device):
    model.eval()
    preds = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.z, data.pos, data.batch)
        preds.append(pred.view(-1))
    preds_tensor = torch.cat(preds)
    return preds_tensor