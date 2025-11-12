"""
Analyze concept probability distribution to understand why recall is low
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from model import CSRModel
from dataloader import create_dataloaders


def analyze_concept_probabilities(model, test_loader, device='cuda'):
    """
    Analyze the distribution of concept probabilities
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    print("Collecting concept probabilities...")
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
    
    num_concepts = all_labels.shape[1]
    
    print("\n" + "="*80)
    print("Concept Probability Analysis")
    print("="*80)
    
    for k in range(num_concepts):
        positive_mask = all_labels[:, k] == 1
        negative_mask = all_labels[:, k] == 0
        
        num_positive = positive_mask.sum()
        num_negative = negative_mask.sum()
        
        if num_positive == 0:
            print(f"\nConcept {k}: No positive samples in test set - SKIP")
            continue
        
        pos_probs = all_probs[positive_mask, k]
        neg_probs = all_probs[negative_mask, k]
        
        print(f"\nConcept {k}:")
        print(f"  Positive samples: {num_positive}")
        print(f"  Negative samples: {num_negative}")
        print(f"  ")
        print(f"  Probability distribution for POSITIVE (true=1) samples:")
        print(f"    Mean:   {pos_probs.mean():.4f}")
        print(f"    Median: {np.median(pos_probs):.4f}")
        print(f"    Std:    {pos_probs.std():.4f}")
        print(f"    Min:    {pos_probs.min():.4f}")
        print(f"    Max:    {pos_probs.max():.4f}")
        print(f"    Q25:    {np.percentile(pos_probs, 25):.4f}")
        print(f"    Q75:    {np.percentile(pos_probs, 75):.4f}")
        print(f"    % > 0.5: {(pos_probs > 0.5).mean()*100:.1f}%")
        print(f"    % > 0.3: {(pos_probs > 0.3).mean()*100:.1f}%")
        print(f"    % > 0.1: {(pos_probs > 0.1).mean()*100:.1f}%")
        print(f"  ")
        print(f"  Probability distribution for NEGATIVE (true=0) samples:")
        print(f"    Mean:   {neg_probs.mean():.4f}")
        print(f"    Median: {np.median(neg_probs):.4f}")
        print(f"    Max:    {neg_probs.max():.4f}")
        print(f"    % > 0.5: {(neg_probs > 0.5).mean()*100:.1f}%")
        
        # Separation analysis
        print(f"  ")
        print(f"  Separation:")
        mean_diff = pos_probs.mean() - neg_probs.mean()
        print(f"    Mean difference: {mean_diff:.4f}")
        if mean_diff < 0.2:
            print(f"    ⚠️  WARNING: Poor separation! Concepts overlap too much.")
        elif mean_diff < 0.4:
            print(f"    ⚠️  Moderate separation. Consider lower threshold.")
        else:
            print(f"    ✓ Good separation.")
    
    print("\n" + "="*80)
    print("Overall Summary:")
    print("="*80)
    
    # Flatten and analyze
    positive_mask_flat = all_labels.flatten() == 1
    all_pos_probs = all_probs.flatten()[positive_mask_flat]
    all_neg_probs = all_probs.flatten()[~positive_mask_flat]
    
    print(f"All positive concept instances:")
    print(f"  Count: {len(all_pos_probs)}")
    print(f"  Mean prob: {all_pos_probs.mean():.4f}")
    print(f"  Median prob: {np.median(all_pos_probs):.4f}")
    print(f"  % with prob > 0.5: {(all_pos_probs > 0.5).mean()*100:.1f}%")
    print(f"  % with prob > 0.3: {(all_pos_probs > 0.3).mean()*100:.1f}%")
    print(f"  % with prob > 0.1: {(all_pos_probs > 0.1).mean()*100:.1f}%")
    
    print(f"\nAll negative concept instances:")
    print(f"  Count: {len(all_neg_probs)}")
    print(f"  Mean prob: {all_neg_probs.mean():.4f}")
    print(f"  % with prob > 0.5: {(all_neg_probs > 0.5).mean()*100:.1f}%")
    
    print("\n" + "="*80)
    print("Diagnosis:")
    print("="*80)
    
    pos_mean = all_pos_probs.mean()
    pct_above_05 = (all_pos_probs > 0.5).mean() * 100
    
    if pct_above_05 < 30:
        print("❌ PROBLEM: Most positive samples have prob < 0.5!")
        print("   → Model is under-confident in predicting concepts")
        print("   → Recommendations:")
        print("     1. Use lower threshold (try 0.2-0.4)")
        print("     2. Retrain with more Stage A epochs")
        print("     3. Increase concept loss weight")
        print(f"     4. Current mean prob for true concepts: {pos_mean:.3f}")
    elif pct_above_05 < 60:
        print("⚠️  ISSUE: Many positive samples have prob < 0.5")
        print(f"   → Try threshold around {pos_mean:.2f}")
    else:
        print("✓ Distribution looks reasonable")
        print("  → Threshold 0.5 should work, but check per-concept metrics")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num_prototypes', type=int, default=10)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--input_size', type=int, default=224)
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
    
    # Analyze
    analyze_concept_probabilities(model, test_loader, args.device)


if __name__ == '__main__':
    main()
