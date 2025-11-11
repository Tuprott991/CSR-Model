"""
Evaluation script for CSR model
Includes standard metrics and interpretability analysis
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score
)
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from typing import Dict, List

from model import CSRModel
from dataloader import create_dataloaders
from doctor_interaction import DoctorInteraction


class CSREvaluator:
    """
    Evaluator for CSR model with comprehensive metrics
    """
    def __init__(
        self,
        model: CSRModel,
        test_loader: DataLoader,
        device: str = 'cuda',
        class_names: List[str] = None,
        concept_names: List[str] = None
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.concept_names = concept_names
        
    def evaluate_task_performance(self) -> Dict[str, float]:
        """
        Evaluate disease classification performance
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['class_label'].to(self.device)
                
                outputs = self.model(images, stage='task')
                logits = outputs['task_logits']
                
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        # ROC-AUC (for binary or multi-class with OvR)
        try:
            if self.model.num_classes == 2:
                roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist()
        }
        
        return metrics
    
    def evaluate_concept_performance(self) -> Dict[str, float]:
        """
        Evaluate concept prediction performance
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating concepts"):
                images = batch['image'].to(self.device)
                concept_labels = batch['concept_labels'].to(self.device)
                
                outputs = self.model(images, stage='concept')
                concept_logits = outputs['concept_logits']
                
                # Sigmoid for multi-label
                concept_probs = torch.sigmoid(concept_logits)
                concept_preds = (concept_probs > 0.5).float()
                
                all_preds.append(concept_preds.cpu().numpy())
                all_labels.append(concept_labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)  # (N, K)
        all_labels = np.concatenate(all_labels, axis=0)  # (N, K)
        
        # Compute metrics per concept
        per_concept_metrics = []
        for k in range(self.model.num_concepts):
            acc = accuracy_score(all_labels[:, k], all_preds[:, k])
            if all_labels[:, k].sum() > 0:  # If concept appears in test set
                prec, rec, f1, _ = precision_recall_fscore_support(
                    all_labels[:, k], all_preds[:, k], average='binary'
                )
            else:
                prec, rec, f1 = 0.0, 0.0, 0.0
            
            per_concept_metrics.append({
                'concept_id': k,
                'concept_name': self.concept_names[k] if self.concept_names else f"concept_{k}",
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1)
            })
        
        # Overall concept metrics (micro-average)
        overall_acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
        overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(
            all_labels.flatten(), all_preds.flatten(), average='binary'
        )
        
        return {
            'overall_accuracy': float(overall_acc),
            'overall_precision': float(overall_prec),
            'overall_recall': float(overall_rec),
            'overall_f1': float(overall_f1),
            'per_concept': per_concept_metrics
        }
    
    def analyze_prototype_usage(self) -> Dict[str, any]:
        """
        Analyze which prototypes are most frequently activated
        """
        self.model.eval()
        
        # Track max activations for each prototype
        prototype_activations = torch.zeros(
            self.model.num_concepts, self.model.M
        )
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Analyzing prototypes"):
                images = batch['image'].to(self.device)
                
                outputs = self.model(images, stage='task')
                similarities = outputs['similarities']  # (B, K, M)
                
                # Count how many times each prototype is the max
                max_prototypes = similarities.argmax(dim=2)  # (B, K)
                
                for k in range(self.model.num_concepts):
                    for m in range(self.model.M):
                        prototype_activations[k, m] += (max_prototypes[:, k] == m).sum().item()
                
                total_samples += images.size(0)
        
        # Compute statistics
        prototype_usage = []
        for k in range(self.model.num_concepts):
            concept_usage = []
            for m in range(self.model.M):
                usage_freq = prototype_activations[k, m].item() / total_samples
                concept_usage.append({
                    'prototype_id': m,
                    'usage_frequency': float(usage_freq)
                })
            
            prototype_usage.append({
                'concept_id': k,
                'concept_name': self.concept_names[k] if self.concept_names else f"concept_{k}",
                'prototypes': concept_usage
            })
        
        return {
            'total_samples': total_samples,
            'prototype_usage': prototype_usage
        }
    
    def evaluate_with_concept_rejection(
        self,
        rejection_rates: List[float] = [0.1, 0.2, 0.3]
    ) -> Dict[str, any]:
        """
        Evaluate performance when randomly rejecting concepts
        Simulates doctor rejection
        """
        results = {}
        
        for rate in rejection_rates:
            print(f"\nEvaluating with {rate*100}% concept rejection...")
            
            all_preds = []
            all_labels = []
            
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc=f"Rejection rate {rate}"):
                    images = batch['image'].to(self.device)
                    labels = batch['class_label'].to(self.device)
                    
                    B = images.size(0)
                    
                    # Randomly reject concepts
                    num_reject = int(self.model.num_concepts * rate)
                    rejected_concepts = torch.zeros(B, self.model.num_concepts, device=self.device)
                    
                    for b in range(B):
                        reject_ids = np.random.choice(
                            self.model.num_concepts, num_reject, replace=False
                        )
                        rejected_concepts[b, reject_ids] = 1
                    
                    # Forward with rejection
                    outputs = self.model(
                        images, 
                        stage='task',
                        rejected_concepts=rejected_concepts
                    )
                    logits = outputs['task_logits']
                    
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted'
            )
            
            results[f'rejection_{int(rate*100)}pct'] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        
        return results
    
    def generate_report(self, save_path: str = None) -> Dict:
        """
        Generate comprehensive evaluation report
        """
        print("=" * 60)
        print("CSR Model Evaluation Report")
        print("=" * 60)
        
        report = {}
        
        # Task performance
        print("\n[1/4] Evaluating task performance...")
        task_metrics = self.evaluate_task_performance()
        report['task_performance'] = task_metrics
        
        print(f"\nTask Performance:")
        print(f"  Accuracy:  {task_metrics['accuracy']:.4f}")
        print(f"  Precision: {task_metrics['precision']:.4f}")
        print(f"  Recall:    {task_metrics['recall']:.4f}")
        print(f"  F1 Score:  {task_metrics['f1_score']:.4f}")
        if task_metrics['roc_auc']:
            print(f"  ROC-AUC:   {task_metrics['roc_auc']:.4f}")
        
        # Concept performance
        print("\n[2/4] Evaluating concept performance...")
        concept_metrics = self.evaluate_concept_performance()
        report['concept_performance'] = concept_metrics
        
        print(f"\nConcept Performance (Overall):")
        print(f"  Accuracy:  {concept_metrics['overall_accuracy']:.4f}")
        print(f"  Precision: {concept_metrics['overall_precision']:.4f}")
        print(f"  Recall:    {concept_metrics['overall_recall']:.4f}")
        print(f"  F1 Score:  {concept_metrics['overall_f1']:.4f}")
        
        # Prototype usage
        print("\n[3/4] Analyzing prototype usage...")
        prototype_analysis = self.analyze_prototype_usage()
        report['prototype_analysis'] = prototype_analysis
        
        # Concept rejection robustness
        print("\n[4/4] Evaluating robustness to concept rejection...")
        rejection_results = self.evaluate_with_concept_rejection()
        report['rejection_robustness'] = rejection_results
        
        print("\nRobustness to Concept Rejection:")
        for key, metrics in rejection_results.items():
            print(f"  {key}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        # Save report
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {save_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate CSR model')
    
    # Dataset args
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)  # For dataloader setup
    parser.add_argument('--val_file', type=str, required=True)
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet34', 'convnextv2_base'])
    parser.add_argument('--num_prototypes', type=int, default=10)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--input_size', type=int, default=224)
    
    # Evaluation args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results')
    
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
    
    # Get dataset info
    num_concepts = test_loader.dataset.num_concepts
    num_classes = test_loader.dataset.num_classes
    concept_names = test_loader.dataset.concept_columns
    class_names = test_loader.dataset.classes
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of concepts: {num_concepts}")
    print(f"Number of classes: {num_classes}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = CSRModel(
        num_concepts=num_concepts,
        num_classes=num_classes,
        num_prototypes_per_concept=args.num_prototypes,
        backbone=args.backbone,
        proj_dim=args.proj_dim,
        pretrained=False  # Load from checkpoint
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from: {args.checkpoint}")
    
    # Create evaluator
    evaluator = CSREvaluator(
        model=model,
        test_loader=test_loader,
        device=args.device,
        class_names=class_names,
        concept_names=concept_names
    )
    
    # Generate report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / 'evaluation_report.json'
    
    report = evaluator.generate_report(save_path=str(report_path))
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
