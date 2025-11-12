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
                concept_preds = (concept_probs > 0.15).float()
                
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
    
    def analyze_misclassifications(self) -> Dict:
        """
        Analyze which concepts are missing in misclassified samples
        """
        self.model.eval()
        
        misclassified = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Analyzing misclassifications"):
                images = batch['image'].to(self.device)
                labels = batch['class_label'].to(self.device)
                concepts_gt = batch['concept_labels'].to(self.device)
                
                outputs = self.model(images)
                preds = outputs['task_logits'].argmax(dim=1)
                concept_probs = torch.sigmoid(outputs['concept_logits'])
                
                # Find misclassified samples
                wrong_mask = (preds != labels)
                
                for i in range(len(images)):
                    if wrong_mask[i]:
                        misclassified.append({
                            'true_class': self.class_names[labels[i].item()] if self.class_names else labels[i].item(),
                            'pred_class': self.class_names[preds[i].item()] if self.class_names else preds[i].item(),
                            'concepts_gt': concepts_gt[i].cpu().numpy(),
                            'concepts_pred': (concept_probs[i] > 0.5).cpu().numpy(),
                            'concept_probs': concept_probs[i].cpu().numpy()
                        })
        
        # Analyze patterns
        analysis = {
            'total_misclassified': len(misclassified),
            'misclassification_patterns': {},
            'concept_analysis': {}
        }
        
        # Group by (true_class, pred_class) pattern
        for item in misclassified:
            pattern = f"{item['true_class']}_to_{item['pred_class']}"
            if pattern not in analysis['misclassification_patterns']:
                analysis['misclassification_patterns'][pattern] = {
                    'count': 0,
                    'missing_concepts': [],
                    'spurious_concepts': []
                }
            
            analysis['misclassification_patterns'][pattern]['count'] += 1
            
            # Find missing concepts (GT=1 but Pred=0)
            missing = np.where((item['concepts_gt'] == 1) & (item['concepts_pred'] == 0))[0]
            analysis['misclassification_patterns'][pattern]['missing_concepts'].extend(missing.tolist())
            
            # Find spurious concepts (GT=0 but Pred=1)
            spurious = np.where((item['concepts_gt'] == 0) & (item['concepts_pred'] == 1))[0]
            analysis['misclassification_patterns'][pattern]['spurious_concepts'].extend(spurious.tolist())
        
        # Summarize concept importance
        for pattern, info in analysis['misclassification_patterns'].items():
            if len(info['missing_concepts']) > 0:
                from collections import Counter
                most_common_missing = Counter(info['missing_concepts']).most_common(3)
                info['top_missing_concepts'] = [
                    {
                        'concept_id': cid,
                        'concept_name': self.concept_names[cid] if self.concept_names else f"concept_{cid}",
                        'frequency': freq
                    }
                    for cid, freq in most_common_missing
                ]
            
            # Clean up raw lists
            del info['missing_concepts']
            del info['spurious_concepts']
        
        return analysis
    
    def evaluate_calibration(self) -> Dict:
        """
        Measure if predicted probabilities match actual accuracy (calibration)
        """
        from sklearn.calibration import calibration_curve
        
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating calibration"):
                images = batch['image'].to(self.device)
                labels = batch['class_label'].to(self.device)
                
                outputs = self.model(images)
                probs = F.softmax(outputs['task_logits'], dim=1)
                
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
        
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Get predicted probability and predicted class
        pred_probs, pred_classes = all_probs.max(dim=1)
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            (pred_classes == all_labels).numpy(),
            pred_probs.numpy(),
            n_bins=10,
            strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        ece = np.abs(fraction_of_positives - mean_predicted_value).mean()
        
        # Maximum Calibration Error (MCE)
        mce = np.abs(fraction_of_positives - mean_predicted_value).max()
        
        return {
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            },
            'interpretation': 'Good' if ece < 0.05 else ('Moderate' if ece < 0.15 else 'Poor')
        }
    
    def generate_html_report(self, save_path: str, report_data: Dict):
        """
        Generate interactive HTML report with visualizations
        """
        task_perf = report_data.get('task_performance', {})
        concept_perf = report_data.get('concept_performance', {})
        prototype_analysis = report_data.get('prototype_analysis', {})
        misclass_analysis = report_data.get('misclassification_analysis', {})
        calibration = report_data.get('calibration', {})
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CSR Model Evaluation Report</title>
    <meta charset="utf-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 20px; 
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px;
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{ 
            color: #7f8c8d; 
            margin-top: 20px;
        }}
        .metric {{ 
            background: #ecf0f1; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 5px;
            border-left: 4px solid #95a5a6;
        }}
        .metric.good {{ border-left-color: #27ae60; }}
        .metric.warning {{ border-left-color: #f39c12; }}
        .metric.bad {{ border-left-color: #e74c3c; }}
        
        .good {{ color: #27ae60; font-weight: bold; }}
        .bad {{ color: #e74c3c; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            background: white;
        }}
        th, td {{ 
            border: 1px solid #bdc3c7; 
            padding: 12px; 
            text-align: left; 
        }}
        th {{ 
            background-color: #3498db; 
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e8f4f8; }}
        
        .summary-box {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-item {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-item.green {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .summary-item.orange {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .summary-item.red {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        
        .summary-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .summary-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .recommendation {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }}
        .recommendation.critical {{
            background: #f8d7da;
            border-color: #f5c6cb;
        }}
        
        code {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 2px;
        }}
        .badge.success {{ background: #d4edda; color: #155724; }}
        .badge.warning {{ background: #fff3cd; color: #856404; }}
        .badge.danger {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 CSR Model Evaluation Report</h1>
        <p><strong>Generated:</strong> {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <!-- Summary Cards -->
        <div class="summary-box">
            <div class="summary-item {'green' if task_perf.get('accuracy', 0) > 0.9 else 'orange'}">
                <div class="summary-label">Task Accuracy</div>
                <div class="summary-value">{task_perf.get('accuracy', 0)*100:.1f}%</div>
            </div>
            <div class="summary-item {'green' if concept_perf.get('overall_recall', 0) > 0.7 else 'red'}">
                <div class="summary-label">Concept Recall</div>
                <div class="summary-value">{concept_perf.get('overall_recall', 0)*100:.1f}%</div>
            </div>
            <div class="summary-item {'green' if calibration.get('expected_calibration_error', 1) < 0.05 else 'orange'}">
                <div class="summary-label">Calibration (ECE)</div>
                <div class="summary-value">{calibration.get('expected_calibration_error', 0):.3f}</div>
            </div>
            <div class="summary-item orange">
                <div class="summary-label">Misclassified</div>
                <div class="summary-value">{misclass_analysis.get('total_misclassified', 0)}</div>
            </div>
        </div>
        
        <!-- Task Performance -->
        <h2>📊 1. Task Performance (Disease Classification)</h2>
        <div class="metric {'good' if task_perf.get('accuracy', 0) > 0.9 else 'warning'}">
            <p><strong>Accuracy:</strong> <span class="{'good' if task_perf.get('accuracy', 0) > 0.9 else 'warning'}">{task_perf.get('accuracy', 0):.4f}</span></p>
            <p><strong>Precision:</strong> {task_perf.get('precision', 0):.4f}</p>
            <p><strong>Recall:</strong> {task_perf.get('recall', 0):.4f}</p>
            <p><strong>F1 Score:</strong> {task_perf.get('f1_score', 0):.4f}</p>
            <p><strong>ROC-AUC:</strong> <span class="good">{task_perf.get('roc_auc', 0):.4f}</span></p>
        </div>
        
        <h3>Per-Class Performance</h3>
        <table>
            <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Status</th></tr>
"""
        
        # Per-class metrics
        per_class_prec = task_perf.get('per_class_precision', [])
        per_class_rec = task_perf.get('per_class_recall', [])
        per_class_f1 = task_perf.get('per_class_f1', [])
        
        for i in range(len(per_class_prec)):
            class_name = self.class_names[i] if self.class_names and i < len(self.class_names) else f"Class {i}"
            prec = per_class_prec[i]
            rec = per_class_rec[i]
            f1 = per_class_f1[i]
            
            if rec < 0.1:
                status = '<span class="badge danger">❌ Never Predicted</span>'
                row_class = 'bad'
            elif rec < 0.5:
                status = '<span class="badge danger">❌ Critical</span>'
                row_class = 'bad'
            elif rec < 0.8:
                status = '<span class="badge warning">⚠️ Warning</span>'
                row_class = 'warning'
            else:
                status = '<span class="badge success">✅ Good</span>'
                row_class = 'good'
            
            html += f"""
            <tr>
                <td><strong>{class_name}</strong></td>
                <td>{prec:.4f}</td>
                <td class="{row_class}">{rec:.4f}</td>
                <td>{f1:.4f}</td>
                <td>{status}</td>
            </tr>
"""
        
        html += f"""
        </table>
        
        <!-- Concept Performance -->
        <h2>🧩 2. Concept Performance</h2>
        <div class="metric {'bad' if concept_perf.get('overall_recall', 0) < 0.6 else 'warning'}">
            <p><strong>Overall Precision:</strong> <span class="{'good' if concept_perf.get('overall_precision', 0) > 0.8 else 'warning'}">{concept_perf.get('overall_precision', 0):.4f}</span></p>
            <p><strong>Overall Recall:</strong> <span class="{'bad' if concept_perf.get('overall_recall', 0) < 0.6 else 'warning'}">{concept_perf.get('overall_recall', 0):.4f}</span></p>
            <p><strong>Overall F1:</strong> <span class="{'bad' if concept_perf.get('overall_f1', 0) < 0.7 else 'warning'}">{concept_perf.get('overall_f1', 0):.4f}</span></p>
        </div>
        
        <h3>Per-Concept Performance</h3>
        <table>
            <tr><th>Concept</th><th>Precision</th><th>Recall</th><th>F1</th><th>Status</th></tr>
"""
        
        for concept in concept_perf.get('per_concept', []):
            name = concept.get('concept_name', 'Unknown')
            prec = concept.get('precision', 0)
            rec = concept.get('recall', 0)
            f1 = concept.get('f1_score', 0)
            
            if rec == 0:
                status = '<span class="badge danger">❌ Not Predicted</span>'
            elif f1 > 0.7:
                status = '<span class="badge success">✅ Good</span>'
            elif f1 > 0.3:
                status = '<span class="badge warning">⚠️ Warning</span>'
            else:
                status = '<span class="badge danger">❌ Critical</span>'
            
            html += f"""
            <tr>
                <td>{name}</td>
                <td>{prec:.4f}</td>
                <td class="{'bad' if rec < 0.5 else 'warning'}">{rec:.4f}</td>
                <td>{f1:.4f}</td>
                <td>{status}</td>
            </tr>
"""
        
        html += f"""
        </table>
        
        <!-- Misclassification Analysis -->
        <h2>🔍 3. Misclassification Analysis</h2>
        <p><strong>Total Misclassified:</strong> {misclass_analysis.get('total_misclassified', 0)} samples</p>
        
        <h3>Common Patterns:</h3>
        <table>
            <tr><th>Pattern</th><th>Count</th><th>Top Missing Concepts</th></tr>
"""
        
        for pattern, info in misclass_analysis.get('misclassification_patterns', {}).items():
            top_missing = info.get('top_missing_concepts', [])
            missing_str = ', '.join([f"{m['concept_name']} ({m['frequency']}x)" for m in top_missing[:3]])
            
            html += f"""
            <tr>
                <td><code>{pattern}</code></td>
                <td>{info['count']}</td>
                <td>{missing_str if missing_str else 'None'}</td>
            </tr>
"""
        
        html += f"""
        </table>
        
        <!-- Calibration -->
        <h2>📈 4. Model Calibration</h2>
        <div class="metric {'good' if calibration.get('expected_calibration_error', 1) < 0.05 else 'warning'}">
            <p><strong>Expected Calibration Error (ECE):</strong> {calibration.get('expected_calibration_error', 0):.4f}</p>
            <p><strong>Maximum Calibration Error (MCE):</strong> {calibration.get('maximum_calibration_error', 0):.4f}</p>
            <p><strong>Interpretation:</strong> <span class="{'good' if calibration.get('interpretation') == 'Good' else 'warning'}">{calibration.get('interpretation', 'N/A')}</span></p>
        </div>
        <p><em>ECE &lt; 0.05 is excellent, 0.05-0.15 is moderate, &gt;0.15 needs recalibration.</em></p>
        
        <!-- Recommendations -->
        <h2>💡 5. Recommendations</h2>
"""
        
        # Generate recommendations
        recommendations = []
        
        if concept_perf.get('overall_recall', 0) < 0.6:
            recommendations.append({
                'level': 'critical',
                'title': 'CRITICAL: Low Concept Recall',
                'desc': f"Concept recall is only {concept_perf.get('overall_recall', 0)*100:.1f}%. Model is missing most concepts.",
                'actions': [
                    'Run <code>python find_optimal_threshold.py</code> to find better thresholds',
                    'Retrain Stage A with FocalBCELoss (alpha=0.75, gamma=2.0)',
                    'Increase Stage A epochs from 30 to 50+',
                    'Use higher concept loss weight (e.g., 2.0 instead of 1.0)'
                ]
            })
        
        # Check if any class has 0% recall
        zero_recall_classes = [i for i, rec in enumerate(per_class_rec) if rec < 0.1]
        if zero_recall_classes:
            class_names_str = ', '.join([self.class_names[i] if self.class_names else f"Class {i}" for i in zero_recall_classes])
            recommendations.append({
                'level': 'critical',
                'title': 'CRITICAL: Classes Never Predicted',
                'desc': f"Classes [{class_names_str}] are never or rarely predicted.",
                'actions': [
                    'Use ClassBalancedCrossEntropyLoss in Stage C',
                    'Apply class-weighted sampling in data loader',
                    'Consider oversampling minority classes',
                    'Check if training data has sufficient samples for these classes'
                ]
            })
        
        # Check prototype usage
        if prototype_analysis:
            unused_count = sum(
                1 for c in prototype_analysis.get('prototype_usage', [])
                for p in c.get('prototypes', [])
                if p.get('usage_frequency', 0) < 0.01
            )
            total_protos = sum(len(c.get('prototypes', [])) for c in prototype_analysis.get('prototype_usage', []))
            if total_protos > 0 and unused_count / total_protos > 0.5:
                recommendations.append({
                    'level': 'warning',
                    'title': 'Many Prototypes Unused',
                    'desc': f"{unused_count}/{total_protos} prototypes are rarely used.",
                    'actions': [
                        'Add PrototypeDiversityLoss in Stage B',
                        'Use PrototypeUsageBalancingLoss during training',
                        'Consider reducing num_prototypes per concept'
                    ]
                })
        
        if calibration.get('expected_calibration_error', 0) > 0.15:
            recommendations.append({
                'level': 'warning',
                'title': 'Poor Calibration',
                'desc': f"ECE = {calibration.get('expected_calibration_error', 0):.3f}. Predicted probabilities don't match actual accuracy.",
                'actions': [
                    'Apply temperature scaling on validation set',
                    'Use label smoothing during training',
                    'Consider Platt scaling or isotonic regression'
                ]
            })
        
        if not recommendations:
            recommendations.append({
                'level': 'success',
                'title': 'Model Performance Looks Good!',
                'desc': 'All major metrics are within acceptable ranges.',
                'actions': ['Continue monitoring on new data', 'Consider A/B testing against baseline']
            })
        
        for rec in recommendations:
            rec_class = 'critical' if rec['level'] == 'critical' else ''
            html += f"""
        <div class="recommendation {rec_class}">
            <h4>{'🚨' if rec['level'] == 'critical' else '⚠️' if rec['level'] == 'warning' else '✅'} {rec['title']}</h4>
            <p>{rec['desc']}</p>
            <p><strong>Actions:</strong></p>
            <ul>
"""
            for action in rec['actions']:
                html += f"                <li>{action}</li>\n"
            html += """
            </ul>
        </div>
"""
        
        html += f"""
        
        <!-- Next Steps -->
        <h2>🎯 6. Next Steps</h2>
        <ol>
            <li>Review misclassification patterns above and check if concepts are correctly labeled</li>
            <li>Run threshold optimization: <code>python find_optimal_threshold.py --checkpoint [path]</code></li>
            <li>If retraining needed, use enhanced losses from <code>losses.py</code></li>
            <li>Visualize prototypes: <code>python visualize_prototypes.py</code></li>
            <li>Test model with doctor interaction tool</li>
        </ol>
        
        <hr style="margin: 30px 0;">
        <p style="text-align: center; color: #7f8c8d;">
            <em>Generated by CSR Model Evaluation Framework</em>
        </p>
    </div>
</body>
</html>
"""
        
        # Save HTML
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n✅ HTML report saved to: {save_path}")
        print(f"   Open in browser to view interactive report")
    
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
        print("\n[4/7] Evaluating robustness to concept rejection...")
        rejection_results = self.evaluate_with_concept_rejection()
        report['rejection_robustness'] = rejection_results
        
        print("\nRobustness to Concept Rejection:")
        for key, metrics in rejection_results.items():
            print(f"  {key}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        # Misclassification analysis
        print("\n[5/7] Analyzing misclassifications...")
        misclass_analysis = self.analyze_misclassifications()
        report['misclassification_analysis'] = misclass_analysis
        
        print(f"\nMisclassification Analysis:")
        print(f"  Total misclassified: {misclass_analysis['total_misclassified']}")
        if misclass_analysis['misclassification_patterns']:
            print(f"  Top patterns:")
            for pattern, info in list(misclass_analysis['misclassification_patterns'].items())[:3]:
                print(f"    {pattern}: {info['count']} cases")
        
        # Calibration
        print("\n[6/7] Evaluating calibration...")
        calibration = self.evaluate_calibration()
        report['calibration'] = calibration
        
        print(f"\nCalibration:")
        print(f"  ECE: {calibration['expected_calibration_error']:.4f} ({calibration['interpretation']})")
        print(f"  MCE: {calibration['maximum_calibration_error']:.4f}")
        
        # Save JSON report
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nJSON report saved to: {save_path}")
        
        # Generate HTML report
        print("\n[7/7] Generating HTML report...")
        if save_path:
            html_path = str(Path(save_path).with_suffix('.html'))
            self.generate_html_report(html_path, report)
        
        print("\n" + "=" * 60)
        print("✅ Evaluation complete!")
        print("=" * 60)
        
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
