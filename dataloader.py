"""
Data loaders for CSR model training
Supports TBX11K, VinDr-CXR, and ISIC datasets
Findings = concepts, Diseases = target classes
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class MedicalImageDataset(Dataset):
    """
    Generic medical image dataset for CSR training
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        image_dir: str,
        concept_columns: List[str],
        class_column: str,
        transform: Optional[transforms.Compose] = None,
        image_col: str = 'image_path'
    ):
        """
        Args:
            data_df: DataFrame with image paths, concept labels, and class labels
            image_dir: Root directory for images
            concept_columns: List of column names for concept labels
            class_column: Column name for disease/class labels
            transform: Image transformations
            image_col: Column name for image paths
        """
        self.data_df = data_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.concept_columns = concept_columns
        self.class_column = class_column
        self.image_col = image_col
        self.transform = transform
        
        self.num_concepts = len(concept_columns)
        
        # Get unique classes and create label mapping
        self.classes = sorted(data_df[class_column].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data_df.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row[self.image_col]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get concept labels (multi-label binary)
        concept_labels = torch.tensor(
            [row[col] for col in self.concept_columns],
            dtype=torch.float32
        )
        
        # Get class label
        class_label = torch.tensor(
            self.class_to_idx[row[self.class_column]],
            dtype=torch.long
        )
        
        return {
            'image': image,
            'concept_labels': concept_labels,
            'class_label': class_label,
            'image_path': str(img_path),
            'idx': idx
        }


class TBX11KDataset(MedicalImageDataset):
    """
    TBX11K dataset loader
    Tuberculosis chest X-ray dataset
    """
    @staticmethod
    def create_from_annotations(
        annotations_file: str,
        image_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        """
        Create TBX11K dataset from annotation file
        
        Expected annotation format:
        {
            "image_name": {
                "findings": ["infiltration", "effusion", ...],
                "disease": "tuberculosis" or "normal"
            }
        }
        """
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Define concept (findings) vocabulary
        all_findings = set()
        for data in annotations.values():
            all_findings.update(data.get('findings', []))
        
        concept_columns = sorted(list(all_findings))
        
        # Build DataFrame
        rows = []
        for img_name, data in annotations.items():
            row = {'image_path': img_name}
            
            # Binary concept labels
            findings = data.get('findings', [])
            for concept in concept_columns:
                row[concept] = 1 if concept in findings else 0
            
            row['disease'] = data.get('disease', 'normal')
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        return TBX11KDataset(
            data_df=df,
            image_dir=image_dir,
            concept_columns=concept_columns,
            class_column='disease',
            transform=transform
        )


class VinDrCXRDataset(MedicalImageDataset):
    """
    VinDr-CXR dataset loader
    Vietnamese chest X-ray dataset
    """
    @staticmethod
    def create_from_csv(
        csv_file: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Create VinDr-CXR dataset from CSV file
        
        Expected CSV columns:
        - image_id: image filename
        - Multiple binary columns for findings (concepts)
        - disease: disease class label
        """
        df = pd.read_csv(csv_file)
        
        # Identify concept columns (all binary columns except disease and image_id)
        concept_columns = [
            col for col in df.columns 
            if col not in ['image_id', 'disease', 'image_path'] 
            and df[col].dtype in [np.int64, np.float64]
        ]
        
        if 'image_path' not in df.columns:
            df['image_path'] = df['image_id']
        
        return VinDrCXRDataset(
            data_df=df,
            image_dir=image_dir,
            concept_columns=concept_columns,
            class_column='disease',
            transform=transform,
            image_col='image_path'
        )


class ISICDataset(MedicalImageDataset):
    """
    ISIC skin lesion dataset loader
    """
    @staticmethod
    def create_from_csv(
        csv_file: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Create ISIC dataset from CSV file
        
        Expected CSV columns:
        - image: image filename
        - Multiple columns for dermoscopic features (concepts)
        - diagnosis: skin condition diagnosis
        """
        df = pd.read_csv(csv_file)
        
        # Common ISIC dermoscopic features as concepts
        concept_columns = [
            col for col in df.columns 
            if col not in ['image', 'diagnosis', 'image_path']
            and df[col].dtype in [np.int64, np.float64]
        ]
        
        if 'image_path' not in df.columns:
            df['image_path'] = df['image']
        
        return ISICDataset(
            data_df=df,
            image_dir=image_dir,
            concept_columns=concept_columns,
            class_column='diagnosis',
            transform=transform,
            image_col='image_path'
        )


def get_transforms(input_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Get image transforms for CSR training
    
    Args:
        input_size: Target image size (224 or 256)
        is_train: Whether training (with augmentation) or evaluation
        
    Returns:
        Composed transforms
    """
    # ImageNet normalization (for pretrained backbones)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if is_train:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.1)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.1)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])


def create_dataloaders(
    dataset_name: str,
    data_root: str,
    train_file: str,
    val_file: str,
    test_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224,
    use_weighted_sampling: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        dataset_name: 'tbx11k', 'vindrcxr', or 'isic'
        data_root: Root directory for images
        train_file: Path to training annotations/CSV
        val_file: Path to validation annotations/CSV
        test_file: Path to test annotations/CSV
        batch_size: Batch size
        num_workers: Number of data loading workers
        input_size: Image input size
        use_weighted_sampling: Whether to use weighted random sampling for class balance
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_transforms(input_size, is_train=True)
    eval_transform = get_transforms(input_size, is_train=False)
    
    # Create datasets based on dataset name
    if dataset_name.lower() == 'tbx11k':
        train_dataset = TBX11KDataset.create_from_annotations(
            train_file, data_root, 'train', train_transform
        )
        val_dataset = TBX11KDataset.create_from_annotations(
            val_file, data_root, 'val', eval_transform
        )
        test_dataset = TBX11KDataset.create_from_annotations(
            test_file, data_root, 'test', eval_transform
        )
    
    elif dataset_name.lower() == 'vindrcxr':
        train_dataset = VinDrCXRDataset.create_from_csv(
            train_file, data_root, train_transform
        )
        val_dataset = VinDrCXRDataset.create_from_csv(
            val_file, data_root, eval_transform
        )
        test_dataset = VinDrCXRDataset.create_from_csv(
            test_file, data_root, eval_transform
        )
    
    elif dataset_name.lower() == 'isic':
        train_dataset = ISICDataset.create_from_csv(
            train_file, data_root, train_transform
        )
        val_dataset = ISICDataset.create_from_csv(
            val_file, data_root, eval_transform
        )
        test_dataset = ISICDataset.create_from_csv(
            test_file, data_root, eval_transform
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create weighted sampler for training if requested
    train_sampler = None
    shuffle_train = True
    
    if use_weighted_sampling:
        # Calculate class weights
        class_labels = [train_dataset[i]['class_label'].item() for i in range(len(train_dataset))]
        class_counts = torch.bincount(torch.tensor(class_labels))
        
        # Weight = 1 / class_count
        class_weights = 1.0 / class_counts.float()
        
        # Assign weight to each sample based on its class
        sample_weights = torch.tensor([class_weights[label] for label in class_labels])
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False  # Don't shuffle when using sampler
        
        print(f"Using weighted sampling for training:")
        print(f"  Class counts: {class_counts.tolist()}")
        print(f"  Class weights: {class_weights.tolist()}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def collate_with_concepts(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that handles concept labels properly
    """
    images = torch.stack([item['image'] for item in batch])
    concept_labels = torch.stack([item['concept_labels'] for item in batch])
    class_labels = torch.stack([item['class_label'] for item in batch])
    
    return {
        'image': images,
        'concept_labels': concept_labels,
        'class_label': class_labels,
        'image_paths': [item['image_path'] for item in batch],
        'indices': [item['idx'] for item in batch]
    }
