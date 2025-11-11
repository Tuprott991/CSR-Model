"""
Utility functions for CSR model
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import yaml
from sklearn.cluster import KMeans


def save_config(config: Dict, save_path: str):
    """Save configuration to YAML file"""
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_prototypes_kmeans(
    vectors_per_concept: List[torch.Tensor],
    num_prototypes_per_concept: int,
    random_state: int = 42
) -> torch.Tensor:
    """
    Initialize prototypes using K-means clustering
    
    Args:
        vectors_per_concept: List of K tensors, each (N_k, proj_dim)
        num_prototypes_per_concept: M prototypes per concept
        random_state: Random seed
        
    Returns:
        prototypes: (K, M, proj_dim)
    """
    K = len(vectors_per_concept)
    proj_dim = vectors_per_concept[0].shape[1]
    prototypes = torch.zeros(K, num_prototypes_per_concept, proj_dim)
    
    for k, vectors in enumerate(vectors_per_concept):
        if len(vectors) >= num_prototypes_per_concept:
            # Use K-means
            kmeans = KMeans(
                n_clusters=num_prototypes_per_concept,
                random_state=random_state,
                n_init=10
            )
            vectors_np = vectors.cpu().numpy()
            kmeans.fit(vectors_np)
            prototypes[k] = torch.from_numpy(kmeans.cluster_centers_)
        else:
            # If fewer vectors, duplicate some
            indices = torch.randint(0, len(vectors), (num_prototypes_per_concept,))
            prototypes[k] = vectors[indices]
    
    return prototypes


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        labels: (N,) class labels
        
    Returns:
        weights: (num_classes,) class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=labels
    )
    return torch.FloatTensor(weights)


def compute_concept_weights(concept_labels: np.ndarray) -> torch.Tensor:
    """
    Compute concept weights for imbalanced multi-label concepts
    
    Args:
        concept_labels: (N, K) binary concept labels
        
    Returns:
        weights: (K,) concept weights
    """
    K = concept_labels.shape[1]
    weights = []
    
    for k in range(K):
        pos_count = concept_labels[:, k].sum()
        neg_count = len(concept_labels) - pos_count
        
        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0
        
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(model, layer_names: List[str]):
    """Freeze specific layers in model"""
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False


def unfreeze_layers(model, layer_names: List[str]):
    """Unfreeze specific layers in model"""
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity matrix between two sets of vectors
    
    Args:
        x: (N, D) tensor
        y: (M, D) tensor
        
    Returns:
        similarity: (N, M) cosine similarity matrix
    """
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return torch.matmul(x_norm, y_norm.t())


def visualize_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = None
):
    """Visualize training and validation loss curves"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def export_model_to_onnx(
    model,
    save_path: str,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cuda'
):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: CSR model
        save_path: Path to save ONNX model
        input_size: (B, C, H, W) input shape
        device: Device
    """
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['task_logits', 'similarities'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'task_logits': {0: 'batch_size'},
            'similarities': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {save_path}")


def load_dataset_splits(
    split_file: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load predefined dataset splits
    
    Args:
        split_file: JSON file with train/val/test splits
        
    Returns:
        train_ids, val_ids, test_ids
    """
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    return splits['train'], splits['val'], splits['test']


def create_dataset_splits(
    all_image_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    save_path: str = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create random dataset splits
    
    Args:
        all_image_ids: List of all image IDs
        train_ratio, val_ratio, test_ratio: Split ratios
        random_state: Random seed
        save_path: Path to save splits JSON
        
    Returns:
        train_ids, val_ids, test_ids
    """
    np.random.seed(random_state)
    
    n = len(all_image_ids)
    indices = np.random.permutation(n)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_ids = [all_image_ids[i] for i in train_indices]
    val_ids = [all_image_ids[i] for i in val_indices]
    test_ids = [all_image_ids[i] for i in test_indices]
    
    if save_path:
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"Splits saved to: {save_path}")
    
    return train_ids, val_ids, test_ids
