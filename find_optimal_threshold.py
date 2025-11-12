"""
Find optimal threshold for concept prediction to improve recall
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, f1_score
import argparse
from pathlib import Path

from model import CSRModel
from dataloader import create_dataloaders


def find_optimal_threshold(
    model,
    test_loader,
    device='cuda',
    thresholds=np.arange(0.1, 0.9, 0.05)
):
    """
    Find optimal threshold for concept prediction
    """
    model.eval()
    
    # Collect all predictions and labels
    all_probs = []
    all_labels = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            concept_labels = batch['concept_labels'].to(device)
            
            outputs = model(images, stage='concept')
            concept_logits = outputs['concept_logits']
            concept_probs = torch.sigmoid(concept_logits)
            
            all_probs.append(concept_probs.cpu().numpy())
            all_labels.append(concept_labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)  # (N, K)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, K)
    
    print(f"\nTesting thresholds from {thresholds[0]:.2f} to {thresholds[-1]:.2f}...")
    
    best_threshold = 0.5
    best_f1 = 0.0
    results = []
    
    for threshold in thresholds:
        preds = (all_probs > threshold).astype(float)
        
        # Overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels.flatten(), 
            preds.flatten(), 
            average='binary',
            zero_division=0
        )
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Print results
    print("\n" + "="*60)
    print("Threshold Analysis:")
    print("="*60)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*60)
    
    for r in results:
        marker = " ← BEST" if r['threshold'] == best_threshold else ""
        print(f"{r['threshold']:<12.2f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}{marker}")
    
    print("="*60)
    print(f"\nOptimal threshold: {best_threshold:.2f}")
    print(f"Best F1 score: {best_f1:.4f}")
    
    # Test with best threshold
    print(f"\nPer-concept metrics with threshold={best_threshold:.2f}:")
    print("-"*60)
    
    best_preds = (all_probs > best_threshold).astype(float)
    
    for k in range(all_labels.shape[1]):
        if all_labels[:, k].sum() > 0:
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_labels[:, k], 
                best_preds[:, k], 
                average='binary',
                zero_division=0
            )
            print(f"Concept {k:2d}: Prec={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    
    return best_threshold, results


def main():
    parser = argparse.ArgumentParser()
    
    # Dataset args
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num_prototypes', type=int, default=10)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--input_size', type=int, default=224)
    
    # Eval args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Create dataloaders
    _, _, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size
    )
    
    # Create model
    num_concepts = test_loader.dataset.num_concepts
    num_classes = test_loader.dataset.num_classes
    
    model = CSRModel(
        num_concepts=num_concepts,
        num_classes=num_classes,
        num_prototypes_per_concept=args.num_prototypes,
        backbone=args.backbone,
        proj_dim=args.proj_dim,
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    print(f"Loaded checkpoint: {args.checkpoint}\n")
    
    # Find optimal threshold
    best_threshold, results = find_optimal_threshold(
        model, test_loader, args.device
    )
    
    print(f"\n{'='*60}")
    print("Recommendation:")
    print(f"{'='*60}")
    print(f"Update evaluate.py line 127:")
    print(f"  OLD: concept_preds = (concept_probs > 0.5).float()")
    print(f"  NEW: concept_preds = (concept_probs > {best_threshold:.2f}).float()")


if __name__ == '__main__':
    main()
