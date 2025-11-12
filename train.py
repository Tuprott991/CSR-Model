"""
Training script for CSR model
Implements 3-stage training: Concept -> Prototype -> Task
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple

from model import CSRModel
from dataloader import create_dataloaders
from losses import (
    MultiPrototypeContrastiveLoss, 
    FocalBCELoss, 
    ClassBalancedCrossEntropyLoss,
    PrototypeDiversityLoss,
    PrototypeUsageBalancingLoss
)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current validation metric (loss or accuracy)
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class CSRTrainer:
    """
    Trainer for CSR model implementing all 3 stages
    """
    def __init__(
        self,
        model: CSRModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        save_dir: str = 'checkpoints',
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.001,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_class_balanced_loss: bool = False,
        class_weights: torch.Tensor = None,
        use_prototype_diversity: bool = False,
        diversity_weight: float = 0.1,
        label_smoothing: float = 0.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.use_early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        
        # Advanced loss configurations
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_class_balanced_loss = use_class_balanced_loss
        self.class_weights = class_weights
        self.use_prototype_diversity = use_prototype_diversity
        self.diversity_weight = diversity_weight
        self.label_smoothing = label_smoothing
        
    def train_stage_a(
        self,
        epochs: int = 30,
        lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        concept_loss_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Stage A: Train concept model (F + C)
        Objective: Multi-label concept classification with BCE loss
        """
        print("=" * 60)
        print("STAGE A: Training Concept Model")
        print("=" * 60)
        
        # Optimizer with different LRs for backbone and concept head
        optimizer = optim.AdamW([
            {'params': self.model.feature_extractor.parameters(), 'lr': backbone_lr},
            {'params': self.model.concept_head.parameters(), 'lr': lr}
        ])
        
        # Loss function - Use Focal BCE if requested
        if self.use_focal_loss:
            criterion = FocalBCELoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
            print(f"Using FocalBCELoss (alpha={self.focal_alpha}, gamma={self.focal_gamma})")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("Using BCEWithLogitsLoss")
        
        print(f"Concept loss weight: {concept_loss_weight}")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        best_val_loss = float('inf')
        # Early stopping for Stage A (monitor val_loss: lower is better)
        if self.use_early_stopping:
            es = EarlyStopping(patience=self.patience, min_delta=self.min_delta, mode='min')
        else:
            es = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                images = batch['image'].to(self.device)
                concept_labels = batch['concept_labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                outputs = self.model(images, stage='concept')
                concept_logits = outputs['concept_logits']
                
                # Loss (weighted)
                loss = criterion(concept_logits, concept_labels) * concept_loss_weight
                
                # Backward
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            train_loss /= len(self.train_loader)
            
            # Validation
            val_loss = self.validate_stage_a()
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('stage_a_best.pth', stage='a', epoch=epoch+1, val_loss=val_loss)
                print(f"Saved best model (val_loss: {val_loss:.4f})")

            # Early stopping check
            if es is not None and es(val_loss):
                print(f"Early stopping triggered after epoch {epoch+1} (no improvement in val_loss for {self.patience} epochs)")
                break
        
        # Save final model
        self.save_checkpoint('stage_a_final.pth', stage='a', epoch=epoch+1, val_loss=best_val_loss)
        
        return {'train_loss': train_loss, 'val_loss': val_loss}
    
    def validate_stage_a(self) -> float:
        """Validate Stage A"""
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                concept_labels = batch['concept_labels'].to(self.device)
                
                outputs = self.model(images, stage='concept')
                concept_logits = outputs['concept_logits']
                
                loss = criterion(concept_logits, concept_labels)
                # loss may be a tensor or a float depending on implementation
                if isinstance(loss, torch.Tensor):
                    val_loss += loss.item()
                else:
                    val_loss += float(loss)
        
        return val_loss / len(self.val_loader)
    
    def extract_concept_vectors(self) -> Dict[int, List[torch.Tensor]]:
        """
        Extract local concept vectors from training set for prototype initialization
        Returns dict mapping concept_id -> list of projected vectors
        """
        print("Extracting concept vectors from training set...")
        
        self.model.eval()
        concept_vectors = {k: [] for k in range(self.model.num_concepts)}
        
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Extracting vectors"):
                images = batch['image'].to(self.device)
                concept_labels = batch['concept_labels'].to(self.device)
                
                # Get projected concept vectors
                outputs = self.model(images, stage='prototype', concept_labels=concept_labels)
                v_proj = outputs['v_proj']  # (B, K, proj_dim)
                
                # Store vectors for each concept (only where label = 1)
                B, K = concept_labels.shape
                for k in range(K):
                    mask = concept_labels[:, k] == 1  # Samples with concept k
                    if mask.any():
                        concept_vectors[k].append(v_proj[mask, k, :].cpu())
        
        # Concatenate vectors for each concept
        for k in range(self.model.num_concepts):
            if len(concept_vectors[k]) > 0:
                concept_vectors[k] = torch.cat(concept_vectors[k], dim=0)
            else:
                # If no samples for this concept, use random vectors
                concept_vectors[k] = torch.randn(10, self.model.proj_dim)
        
        return concept_vectors
    
    def train_stage_b(
        self,
        epochs: int = 20,
        lr: float = 1e-3,
        lambda_temp: float = 20.0,
        gamma: float = 20.0,
        margin: float = 0.05,
        diversity_weight: float = 0.1
    ) -> Dict[str, float]:
        """
        Stage B: Learn projector and prototypes with contrastive loss
        Freeze backbone and concept head
        """
        print("=" * 60)
        print("STAGE B: Training Prototypes with Contrastive Learning")
        print("=" * 60)
        
        # Initialize prototypes from concept vectors
        concept_vectors = self.extract_concept_vectors()
        vectors_list = [concept_vectors[k] for k in range(self.model.num_concepts)]
        self.model.prototypes.initialize_from_vectors(vectors_list)
        print("Prototypes initialized from concept vectors")
        
        # Freeze backbone and concept head
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.model.concept_head.parameters():
            param.requires_grad = False
        
        # Optimizer for projector and prototypes
        optimizer = optim.AdamW([
            {'params': self.model.projector.parameters()},
            {'params': self.model.prototypes.parameters()}
        ], lr=lr)
        
        # Contrastive loss
        criterion = MultiPrototypeContrastiveLoss(
            lambda_temp=lambda_temp,
            gamma=gamma,
            margin=margin
        )
        
        # Prototype diversity loss (optional)
        if self.use_prototype_diversity:
            diversity_loss = PrototypeDiversityLoss()
            print(f"Using PrototypeDiversityLoss (weight={diversity_weight})")
        else:
            diversity_loss = None
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        best_val_loss = float('inf')
        # Early stopping for Stage B (monitor val_loss: lower is better)
        if self.use_early_stopping:
            es = EarlyStopping(patience=self.patience, min_delta=self.min_delta, mode='min')
        else:
            es = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                images = batch['image'].to(self.device)
                concept_labels = batch['concept_labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                outputs = self.model(images, stage='prototype', concept_labels=concept_labels)
                v_proj = outputs['v_proj']  # (B, K, proj_dim)
                
                # Get prototypes
                prototypes = self.model.prototypes()  # (K, M, proj_dim)
                
                # Compute contrastive loss
                loss = criterion(v_proj, concept_labels, prototypes)
                
                # Add prototype diversity loss
                if diversity_loss is not None:
                    div_loss = diversity_loss(prototypes)
                    loss = loss + diversity_weight * div_loss
                
                # Backward
                loss.backward()
                optimizer.step()
                
                # Normalize prototypes after update
                self.model.prototypes.normalize_prototypes()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            train_loss /= len(self.train_loader)
            
            # Validation
            val_loss = self.validate_stage_b(criterion)
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('stage_b_best.pth', stage='b', epoch=epoch+1, val_loss=val_loss)
                print(f"Saved best model (val_loss: {val_loss:.4f})")

            # Early stopping
            if es is not None and es(val_loss):
                print(f"Early stopping triggered after epoch {epoch+1} (no improvement in val_loss for {self.patience} epochs)")
                break
        
        # Save final model
        self.save_checkpoint('stage_b_final.pth', stage='b', epoch=epoch+1, val_loss=best_val_loss)
        
        # Unfreeze for stage C
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = True
        for param in self.model.concept_head.parameters():
            param.requires_grad = True
        
        return {'train_loss': train_loss, 'val_loss': val_loss}
    
    def validate_stage_b(self, criterion: nn.Module) -> float:
        """Validate Stage B"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                concept_labels = batch['concept_labels'].to(self.device)
                
                outputs = self.model(images, stage='prototype', concept_labels=concept_labels)
                v_proj = outputs['v_proj']
                prototypes = self.model.prototypes()
                
                loss = criterion(v_proj, concept_labels, prototypes)
                # some loss implementations may return a float
                if isinstance(loss, torch.Tensor):
                    val_loss += loss.item()
                else:
                    val_loss += float(loss)
        
        return val_loss / len(self.val_loader)
    
    def train_stage_c(
        self,
        epochs: int = 30,
        lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        finetune_all: bool = True
    ) -> Dict[str, float]:
        """
        Stage C: Train task head for disease classification
        Optionally fine-tune entire model
        """
        print("=" * 60)
        print("STAGE C: Training Task Head")
        print("=" * 60)
        
        # Optimizer
        if finetune_all:
            optimizer = optim.AdamW([
                {'params': self.model.feature_extractor.parameters(), 'lr': backbone_lr},
                {'params': self.model.concept_head.parameters(), 'lr': lr},
                {'params': self.model.projector.parameters(), 'lr': lr},
                {'params': self.model.prototypes.parameters(), 'lr': lr},
                {'params': self.model.task_head.parameters(), 'lr': lr}
            ])
        else:
            # Freeze prototypes, only train task head
            for param in self.model.prototypes.parameters():
                param.requires_grad = False
            optimizer = optim.AdamW(self.model.task_head.parameters(), lr=lr)
        
        # Loss function - Use Class-Balanced Loss if requested
        if self.use_class_balanced_loss and self.class_weights is not None:
            # Use class-balanced cross-entropy
            num_classes = self.model.num_classes
            # Calculate class counts from training data
            class_counts = torch.zeros(num_classes)
            for batch in self.train_loader:
                class_labels = batch['class_label']
                for label in class_labels:
                    class_counts[label.item()] += 1
            
            criterion = ClassBalancedCrossEntropyLoss(
                class_counts=class_counts.to(self.device),
                beta=0.9999
            )
            print(f"Using ClassBalancedCrossEntropyLoss")
            print(f"  Class counts: {class_counts.tolist()}")
        elif self.class_weights is not None:
            # Use standard cross-entropy with class weights
            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights.to(self.device),
                label_smoothing=self.label_smoothing
            )
            print(f"Using CrossEntropyLoss with class weights and label smoothing={self.label_smoothing}")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            print(f"Using CrossEntropyLoss with label smoothing={self.label_smoothing}")
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        best_val_acc = 0.0
        # Early stopping for Stage C (monitor val_acc: higher is better)
        if self.use_early_stopping:
            es = EarlyStopping(patience=self.patience, min_delta=self.min_delta, mode='max')
        else:
            es = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                images = batch['image'].to(self.device)
                class_labels = batch['class_label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                outputs = self.model(images, stage='task')
                task_logits = outputs['task_logits']
                
                # Loss
                loss = criterion(task_logits, class_labels)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                # Normalize prototypes if they're being trained
                if finetune_all:
                    self.model.prototypes.normalize_prototypes()
                
                # Metrics
                train_loss += loss.item()
                _, predicted = task_logits.max(1)
                train_total += class_labels.size(0)
                train_correct += predicted.eq(class_labels).sum().item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * train_correct / train_total
                })
            
            train_loss /= len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation
            val_loss, val_acc = self.validate_stage_c()
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint('stage_c_best.pth', stage='c', epoch=epoch+1, val_loss=val_loss, val_acc=val_acc)
                print(f"Saved best model (val_acc: {val_acc:.2f}%)")

            # Early stopping (monitor val_acc)
            if es is not None and es(val_acc):
                print(f"Early stopping triggered after epoch {epoch+1} (no improvement in val_acc for {self.patience} epochs)")
                break
        
        # Save final model
        self.save_checkpoint('stage_c_final.pth', stage='c', epoch=epoch+1, val_loss=val_loss, val_acc=val_acc)
        
        return {'train_loss': train_loss, 'train_acc': train_acc, 
                'val_loss': val_loss, 'val_acc': val_acc}
    
    def validate_stage_c(self) -> Tuple[float, float]:
        """Validate Stage C"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                class_labels = batch['class_label'].to(self.device)
                
                outputs = self.model(images, stage='task')
                task_logits = outputs['task_logits']
                
                loss = criterion(task_logits, class_labels)
                if isinstance(loss, torch.Tensor):
                    val_loss += loss.item()
                else:
                    val_loss += float(loss)
                
                _, predicted = task_logits.max(1)
                total += class_labels.size(0)
                correct += predicted.eq(class_labels).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, filename: str, stage: str, epoch: int = None, val_loss: float = None, val_acc: float = None):
        """Save model checkpoint

        Args:
            filename: checkpoint filename to write into save_dir
            stage: 'a'|'b'|'c'
            epoch: optional epoch number
            val_loss: optional validation loss to record
            val_acc: optional validation accuracy to record
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'stage': stage
        }
        if epoch is not None:
            checkpoint['epoch'] = int(epoch)
        if val_loss is not None:
            checkpoint['val_loss'] = float(val_loss)
        if val_acc is not None:
            checkpoint['val_acc'] = float(val_acc)

        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint
        
        Args:
            filename: Path to checkpoint file (can be absolute or relative to save_dir)
        
        Returns:
            checkpoint dict
        """
        # Handle both absolute paths and paths relative to save_dir
        checkpoint_path = Path(filename)
        if not checkpoint_path.exists():
            checkpoint_path = self.save_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filename}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print which stage was loaded
        stage = checkpoint.get('stage', 'unknown')
        epoch = checkpoint.get('epoch', 'unknown')
        
        print(f"✓ Loaded checkpoint from Stage {stage.upper()}, Epoch {epoch}")
        
        # Print metrics if available
        if 'val_loss' in checkpoint:
            print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"  Val Acc: {checkpoint['val_acc']:.2f}%")
        
        return checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train CSR model')
    
    # Dataset args
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['tbx11k', 'vindrcxr', 'isic'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for images')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Training annotations file')
    parser.add_argument('--val_file', type=str, required=True,
                        help='Validation annotations file')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test annotations file')
    
    # Model args
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet34', 'convnextv2_base'])
    parser.add_argument('--num_prototypes', type=int, default=10,
                        help='Number of prototypes per concept')
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='Projection dimension')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    
    # Stage-specific args
    parser.add_argument('--stage_a_epochs', type=int, default=30)
    parser.add_argument('--stage_b_epochs', type=int, default=20)
    parser.add_argument('--stage_c_epochs', type=int, default=30)
    
    # Resume training args
    parser.add_argument('--start_stage', type=str, default='a',
                        choices=['a', 'b', 'c'],
                        help='Start from which stage (a=concept, b=prototype, c=task)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/stage_a_best.pth)')
    
    # Early stopping args
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum improvement threshold for early stopping')
    
    # Advanced loss args
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal BCE Loss for concepts (Stage A)')
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--concept_loss_weight', type=float, default=1.0,
                        help='Weight for concept loss in Stage A')
    
    parser.add_argument('--use_class_balanced_loss', action='store_true',
                        help='Use class-balanced cross-entropy for task (Stage C)')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights in cross-entropy (Stage C)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for cross-entropy (0.1 recommended)')
    
    parser.add_argument('--use_prototype_diversity', action='store_true',
                        help='Use prototype diversity loss (Stage B)')
    parser.add_argument('--diversity_weight', type=float, default=0.1,
                        help='Weight for prototype diversity loss')
    
    parser.add_argument('--use_weighted_sampling', action='store_true',
                        help='Use weighted random sampling for class balance')
    
    args = parser.parse_args()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        use_weighted_sampling=args.use_weighted_sampling
    )
    
    # Get dataset info
    num_concepts = train_loader.dataset.num_concepts
    num_classes = train_loader.dataset.num_classes
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of concepts: {num_concepts}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    if args.early_stopping:
        print(f"Early stopping: enabled (patience={args.patience}, min_delta={args.min_delta})")
    else:
        print("Early stopping: disabled")
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_labels = [train_loader.dataset[i]['class_label'].item() for i in range(len(train_loader.dataset))]
        class_counts = torch.bincount(torch.tensor(class_labels))
        # Inverse frequency weighting
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
        print(f"Class weights: {class_weights.tolist()}")
    
    # Create model
    model = CSRModel(
        num_concepts=num_concepts,
        num_classes=num_classes,
        num_prototypes_per_concept=args.num_prototypes,
        backbone=args.backbone,
        proj_dim=args.proj_dim,
        pretrained=True
    )
    
    # Create trainer
    trainer = CSRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        save_dir=args.save_dir,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_class_balanced_loss=args.use_class_balanced_loss,
        class_weights=class_weights,
        use_prototype_diversity=args.use_prototype_diversity,
        diversity_weight=args.diversity_weight,
        label_smoothing=args.label_smoothing
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {args.resume}")
        print(f"{'='*60}\n")
        trainer.load_checkpoint(args.resume)
    
    # Run training stages based on start_stage
    print(f"\n{'='*60}")
    print(f"Starting training from stage {args.start_stage.upper()}")
    print(f"{'='*60}\n")
    
    if args.start_stage == 'a':
        # Run all three stages
        print("Will run: Stage A → Stage B → Stage C")
        trainer.train_stage_a(
            epochs=args.stage_a_epochs,
            concept_loss_weight=args.concept_loss_weight
        )
        trainer.train_stage_b(
            epochs=args.stage_b_epochs,
            diversity_weight=args.diversity_weight
        )
        trainer.train_stage_c(epochs=args.stage_c_epochs)
        
    elif args.start_stage == 'b':
        # Start from stage B (requires stage A checkpoint)
        if not args.resume:
            # Try to auto-load stage_a_best.pth
            stage_a_checkpoint = Path(args.save_dir) / 'stage_a_best.pth'
            if stage_a_checkpoint.exists():
                print(f"Auto-loading Stage A checkpoint: {stage_a_checkpoint}")
                trainer.load_checkpoint(str(stage_a_checkpoint))
            else:
                raise ValueError(
                    "Starting from stage B requires Stage A checkpoint. "
                    "Either:\n"
                    f"  1. Train Stage A first, or\n"
                    f"  2. Provide --resume path to stage_a_*.pth checkpoint"
                )
        
        print("Will run: Stage B → Stage C")
        trainer.train_stage_b(
            epochs=args.stage_b_epochs,
            diversity_weight=args.diversity_weight
        )
        trainer.train_stage_c(epochs=args.stage_c_epochs)
        
    elif args.start_stage == 'c':
        # Start from stage C (requires stage B checkpoint)
        if not args.resume:
            # Try to auto-load stage_b_best.pth
            stage_b_checkpoint = Path(args.save_dir) / 'stage_b_best.pth'
            if stage_b_checkpoint.exists():
                print(f"Auto-loading Stage B checkpoint: {stage_b_checkpoint}")
                trainer.load_checkpoint(str(stage_b_checkpoint))
            else:
                raise ValueError(
                    "Starting from stage C requires Stage B checkpoint. "
                    "Either:\n"
                    f"  1. Train Stage A and B first, or\n"
                    f"  2. Provide --resume path to stage_b_*.pth checkpoint"
                )
        
        print("Will run: Stage C only")
        trainer.train_stage_c(epochs=args.stage_c_epochs)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nCheckpoints saved in: {args.save_dir}")
    print("Available checkpoints:")
    checkpoint_dir = Path(args.save_dir)
    if checkpoint_dir.exists():
        for ckpt in sorted(checkpoint_dir.glob("*.pth")):
            print(f"  - {ckpt.name}")


if __name__ == '__main__':
    main()
