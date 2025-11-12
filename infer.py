"""
Inference script for CSR model
Supports multiple inference modes:
1. Single image inference
2. Batch inference on directory
3. Interactive mode with concept rejection
4. Spatial feedback mode with bounding boxes
5. Visualization of concept activations and similarity maps
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import CSRModel
from doctor_interaction import DoctorInteraction


class CSRInference:
    """
    Inference wrapper for CSR model with multiple modes
    """
    def __init__(
        self,
        checkpoint_path: str,
        num_concepts: int,
        num_classes: int,
        num_prototypes: int = 10,
        backbone: str = 'resnet50',
        proj_dim: int = 128,
        device: str = 'cuda',
        class_names: List[str] = None,
        concept_names: List[str] = None
    ):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            num_concepts: Number of concepts (K)
            num_classes: Number of disease classes
            num_prototypes: Number of prototypes per concept (M)
            backbone: Backbone architecture
            proj_dim: Projection dimension
            device: 'cuda' or 'cpu'
            class_names: List of class names (for display)
            concept_names: List of concept names (for display)
        """
        self.device = device
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.concept_names = concept_names or [f"Concept {i}" for i in range(num_concepts)]
        
        # Create model
        self.model = CSRModel(
            num_concepts=num_concepts,
            num_classes=num_classes,
            num_prototypes_per_concept=num_prototypes,
            backbone=backbone,
            proj_dim=proj_dim,
            pretrained=False
        ).to(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create doctor interaction interface
        self.doctor = DoctorInteraction(self.model, device)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(int(224 * 1.1)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Model loaded successfully!")
        print(f"  Concepts: {num_concepts}")
        print(f"  Classes: {num_classes}")
        print(f"  Prototypes per concept: {num_prototypes}")
        print(f"  Device: {device}")
    
    def load_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            image_tensor: (1, 3, 224, 224) preprocessed image
            original_image: (H, W, 3) original image as numpy array
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Transform
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return image_tensor, original_image
    
    def predict(
        self, 
        image: torch.Tensor,
        return_concepts: bool = True,
        return_similarities: bool = False
    ) -> Dict:
        """
        Standard inference without feedback
        
        Args:
            image: (1, 3, H, W) input image
            return_concepts: Whether to return concept predictions
            return_similarities: Whether to return prototype similarities
            
        Returns:
            Dictionary with predictions and optional extra info
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get concept predictions first
            concept_outputs = self.model(image, stage='concept')
            concept_logits = concept_outputs['concept_logits']
            
            # Get task predictions
            task_outputs = self.model(image, stage='task')
            task_logits = task_outputs['task_logits']
            
            # Process task predictions
            task_probs = F.softmax(task_logits, dim=1)
            pred_class = task_logits.argmax(dim=1).item()
            pred_prob = task_probs[0, pred_class].item()
            
            result = {
                'predicted_class': self.class_names[pred_class],
                'predicted_class_id': pred_class,
                'confidence': pred_prob,
                'all_probabilities': {
                    self.class_names[i]: float(task_probs[0, i])
                    for i in range(len(self.class_names))
                }
            }
            
            # Concept predictions
            if return_concepts:
                concept_probs = torch.sigmoid(concept_logits)[0]
                
                result['concepts'] = {
                    self.concept_names[i]: {
                        'probability': float(concept_probs[i]),
                        'predicted': bool(concept_probs[i] > 0.5)
                    }
                    for i in range(len(self.concept_names))
                }
            
            # Prototype similarities
            if return_similarities:
                similarities = task_outputs['similarities'][0]  # (K, M)
                result['similarities'] = similarities.cpu().numpy().tolist()
        
        return result
    
    def predict_with_rejection(
        self,
        image: torch.Tensor,
        rejected_concepts: List[str]
    ) -> Dict:
        """
        Inference with concept rejection (doctor feedback)
        
        Args:
            image: (1, 3, H, W) input image
            rejected_concepts: List of concept names to reject
            
        Returns:
            Dictionary with modified predictions
        """
        # Convert concept names to IDs
        rejected_ids = []
        for concept_name in rejected_concepts:
            if concept_name in self.concept_names:
                rejected_ids.append(self.concept_names.index(concept_name))
            else:
                print(f"Warning: Unknown concept '{concept_name}', skipping")
        
        if not rejected_ids:
            print("No valid concepts to reject, returning standard prediction")
            return self.predict(image)
        
        # Get prediction with rejection
        task_logits = self.doctor.reject_concepts(image, rejected_ids)
        task_probs = F.softmax(task_logits, dim=1)
        pred_class = task_logits.argmax(dim=1).item()
        pred_prob = task_probs[0, pred_class].item()
        
        return {
            'predicted_class': self.class_names[pred_class],
            'predicted_class_id': pred_class,
            'confidence': pred_prob,
            'rejected_concepts': rejected_concepts,
            'all_probabilities': {
                self.class_names[i]: float(task_probs[0, i])
                for i in range(len(self.class_names))
            }
        }
    
    def predict_with_spatial_feedback(
        self,
        image: torch.Tensor,
        positive_boxes: List[Tuple[int, int, int, int]],
        negative_boxes: List[Tuple[int, int, int, int]],
        alpha: float = 0.2
    ) -> Dict:
        """
        Inference with spatial feedback (bounding boxes)
        
        Args:
            image: (1, 3, H, W) input image
            positive_boxes: List of (x1, y1, x2, y2) for important regions
            negative_boxes: List of (x1, y1, x2, y2) for unimportant regions
            alpha: Suppression factor for negative regions
            
        Returns:
            Dictionary with modified predictions
        """
        task_logits = self.doctor.inference_with_spatial_feedback(
            image, positive_boxes, negative_boxes, alpha
        )
        
        task_probs = F.softmax(task_logits, dim=1)
        pred_class = task_logits.argmax(dim=1).item()
        pred_prob = task_probs[0, pred_class].item()
        
        return {
            'predicted_class': self.class_names[pred_class],
            'predicted_class_id': pred_class,
            'confidence': pred_prob,
            'num_positive_boxes': len(positive_boxes),
            'num_negative_boxes': len(negative_boxes),
            'all_probabilities': {
                self.class_names[i]: float(task_probs[0, i])
                for i in range(len(self.class_names))
            }
        }
    
    def get_top_concepts(self, image: torch.Tensor, top_k: int = 5) -> Dict:
        """
        Get top-k most activated concepts
        
        Args:
            image: (1, 3, H, W) input image
            top_k: Number of top concepts to return
            
        Returns:
            Dictionary mapping concept names to scores
        """
        concept_scores = self.doctor.get_top_concepts(image, top_k)
        
        return {
            self.concept_names[concept_id]: score
            for concept_id, score in concept_scores.items()
        }
    
    def visualize_concepts(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        concept_ids: List[int] = None,
        save_path: str = None
    ):
        """
        Visualize concept activation maps
        
        Args:
            image: (1, 3, H, W) preprocessed image
            original_image: (H, W, 3) original image
            concept_ids: List of concept IDs to visualize (None = top 5)
            save_path: Path to save visualization
        """
        # Get top concepts if not specified
        if concept_ids is None:
            top_concepts = self.doctor.get_top_concepts(image, top_k=5)
            concept_ids = list(top_concepts.keys())
        
        num_concepts = len(concept_ids)
        fig, axes = plt.subplots(1, num_concepts + 1, figsize=(4 * (num_concepts + 1), 4))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Concept activations
        for i, concept_id in enumerate(concept_ids):
            vis = self.doctor.visualize_concept_activation(
                image, concept_id, original_image, alpha=0.5
            )
            axes[i + 1].imshow(vis)
            axes[i + 1].set_title(f'{self.concept_names[concept_id]}')
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_prototypes(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        concept_id: int,
        save_path: str = None
    ):
        """
        Visualize similarity maps for all prototypes of a concept
        
        Args:
            image: (1, 3, H, W) preprocessed image
            original_image: (H, W, 3) original image
            concept_id: Which concept to visualize
            save_path: Path to save visualization
        """
        M = self.model.num_prototypes_per_concept
        
        fig, axes = plt.subplots(2, (M + 1) // 2, figsize=(4 * ((M + 1) // 2), 8))
        axes = axes.flatten()
        
        for proto_id in range(M):
            vis = self.doctor.visualize_similarity_maps(
                image, concept_id, proto_id, original_image, alpha=0.5
            )
            axes[proto_id].imshow(vis)
            axes[proto_id].set_title(f'Prototype {proto_id}')
            axes[proto_id].axis('off')
        
        fig.suptitle(f'Prototypes for {self.concept_names[concept_id]}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def batch_predict(
        self,
        image_dir: str,
        output_file: str = None,
        save_visualizations: bool = False,
        viz_dir: str = None
    ) -> List[Dict]:
        """
        Run inference on all images in a directory
        
        Args:
            image_dir: Directory containing images
            output_file: JSON file to save results
            save_visualizations: Whether to save concept visualizations
            viz_dir: Directory to save visualizations
            
        Returns:
            List of prediction dictionaries
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.jpeg'))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return []
        
        print(f"Found {len(image_files)} images")
        
        if save_visualizations:
            viz_dir = Path(viz_dir or 'visualizations')
            viz_dir.mkdir(exist_ok=True, parents=True)
        
        results = []
        
        for img_file in image_files:
            print(f"\nProcessing: {img_file.name}")
            
            # Load image
            image, original = self.load_image(str(img_file))
            
            # Predict
            result = self.predict(image, return_concepts=True)
            result['image_path'] = str(img_file)
            
            print(f"  Prediction: {result['predicted_class']} ({result['confidence']:.3f})")
            
            # Visualize top concepts
            if save_visualizations:
                viz_path = viz_dir / f"{img_file.stem}_concepts.png"
                self.visualize_concepts(image, original, save_path=str(viz_path))
            
            results.append(result)
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='CSR Model Inference')
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num_prototypes', type=int, default=10)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Dataset info (needed for num_concepts and num_classes)
    parser.add_argument('--num_concepts', type=int, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--class_names', type=str, nargs='+', default=None)
    parser.add_argument('--concept_names', type=str, nargs='+', default=None)
    
    # Inference mode
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch', 'interactive', 'visualize'],
                       help='Inference mode')
    
    # Input
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Directory of images for batch mode')
    
    # Output
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualizations')
    parser.add_argument('--viz_dir', type=str, default='visualizations',
                       help='Directory for visualizations')
    
    # Interactive options
    parser.add_argument('--reject_concepts', type=str, nargs='+', default=None,
                       help='Concept names to reject')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top concepts to show')
    
    args = parser.parse_args()
    
    # Create inference engine
    inferencer = CSRInference(
        checkpoint_path=args.checkpoint,
        num_concepts=args.num_concepts,
        num_classes=args.num_classes,
        num_prototypes=args.num_prototypes,
        backbone=args.backbone,
        proj_dim=args.proj_dim,
        device=args.device,
        class_names=args.class_names,
        concept_names=args.concept_names
    )
    
    print("\n" + "="*60)
    print(f"Running inference in '{args.mode}' mode")
    print("="*60 + "\n")
    
    # Single image mode
    if args.mode == 'single':
        if not args.image:
            parser.error("--image required for single mode")
        
        print(f"Loading image: {args.image}")
        image, original = inferencer.load_image(args.image)
        
        # Standard prediction
        print("\n--- Standard Prediction ---")
        result = inferencer.predict(image, return_concepts=True)
        
        print(f"\nPredicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        print("\nAll Class Probabilities:")
        for cls, prob in result['all_probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
        
        print("\nTop Predicted Concepts:")
        concepts = [(name, info['probability']) 
                   for name, info in result['concepts'].items() 
                   if info['predicted']]
        concepts.sort(key=lambda x: x[1], reverse=True)
        for name, prob in concepts[:args.top_k]:
            print(f"  {name}: {prob:.4f}")
        
        # Concept rejection if specified
        if args.reject_concepts:
            print(f"\n--- Prediction with Rejected Concepts ---")
            print(f"Rejecting: {', '.join(args.reject_concepts)}")
            
            result_rejected = inferencer.predict_with_rejection(
                image, args.reject_concepts
            )
            
            print(f"\nNew Predicted Class: {result_rejected['predicted_class']}")
            print(f"New Confidence: {result_rejected['confidence']:.4f}")
            
            print("\nChange in probabilities:")
            for cls in result['all_probabilities'].keys():
                old_prob = result['all_probabilities'][cls]
                new_prob = result_rejected['all_probabilities'][cls]
                change = new_prob - old_prob
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"  {cls}: {old_prob:.4f} {arrow} {new_prob:.4f} ({change:+.4f})")
        
        # Visualizations
        if args.visualize:
            viz_dir = Path(args.viz_dir)
            viz_dir.mkdir(exist_ok=True, parents=True)
            
            # Visualize top concepts
            print("\nGenerating concept visualizations...")
            viz_path = viz_dir / f"{Path(args.image).stem}_concepts.png"
            inferencer.visualize_concepts(image, original, save_path=str(viz_path))
        
        # Save result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    # Batch mode
    elif args.mode == 'batch':
        if not args.image_dir:
            parser.error("--image_dir required for batch mode")
        
        results = inferencer.batch_predict(
            image_dir=args.image_dir,
            output_file=args.output,
            save_visualizations=args.visualize,
            viz_dir=args.viz_dir
        )
        
        print(f"\nProcessed {len(results)} images")
    
    # Visualize mode (detailed visualization of one image)
    elif args.mode == 'visualize':
        if not args.image:
            parser.error("--image required for visualize mode")
        
        print(f"Loading image: {args.image}")
        image, original = inferencer.load_image(args.image)
        
        # Get top concepts
        top_concepts = inferencer.get_top_concepts(image, top_k=args.top_k)
        
        print("\nTop Activated Concepts:")
        for concept_name, score in top_concepts.items():
            print(f"  {concept_name}: {score:.4f}")
        
        viz_dir = Path(args.viz_dir)
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Visualize all top concepts
        print("\nGenerating concept activation maps...")
        concept_ids = [inferencer.concept_names.index(name) for name in top_concepts.keys()]
        viz_path = viz_dir / f"{Path(args.image).stem}_concepts.png"
        inferencer.visualize_concepts(
            image, original, concept_ids=concept_ids, save_path=str(viz_path)
        )
        
        # Visualize prototypes for top concept
        print("\nGenerating prototype similarity maps for top concept...")
        top_concept_id = concept_ids[0]
        proto_path = viz_dir / f"{Path(args.image).stem}_prototypes_c{top_concept_id}.png"
        inferencer.visualize_prototypes(
            image, original, top_concept_id, save_path=str(proto_path)
        )
    
    # Interactive mode
    elif args.mode == 'interactive':
        if not args.image:
            parser.error("--image required for interactive mode")
        
        print(f"Loading image: {args.image}")
        image, original = inferencer.load_image(args.image)
        
        # Initial prediction
        result = inferencer.predict(image, return_concepts=True)
        
        print(f"\nInitial Prediction: {result['predicted_class']} ({result['confidence']:.4f})")
        
        # Interactive loop
        while True:
            print("\n" + "="*60)
            print("Interactive Mode - Options:")
            print("  1. View top concepts")
            print("  2. Reject concepts")
            print("  3. View current prediction")
            print("  4. Visualize concepts")
            print("  5. Exit")
            print("="*60)
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                top_concepts = inferencer.get_top_concepts(image, top_k=10)
                print("\nTop Activated Concepts:")
                for i, (concept_name, score) in enumerate(top_concepts.items(), 1):
                    print(f"  {i}. {concept_name}: {score:.4f}")
            
            elif choice == '2':
                print("\nAvailable concepts:")
                for i, name in enumerate(inferencer.concept_names):
                    print(f"  {i}. {name}")
                
                reject_input = input("\nEnter concept numbers to reject (comma-separated): ").strip()
                try:
                    indices = [int(x.strip()) for x in reject_input.split(',')]
                    reject_names = [inferencer.concept_names[i] for i in indices]
                    
                    result_rejected = inferencer.predict_with_rejection(image, reject_names)
                    
                    print(f"\nNew Prediction: {result_rejected['predicted_class']}")
                    print(f"New Confidence: {result_rejected['confidence']:.4f}")
                    
                except (ValueError, IndexError):
                    print("Invalid input!")
            
            elif choice == '3':
                print(f"\nCurrent Prediction: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("\nAll probabilities:")
                for cls, prob in result['all_probabilities'].items():
                    print(f"  {cls}: {prob:.4f}")
            
            elif choice == '4':
                viz_dir = Path(args.viz_dir)
                viz_dir.mkdir(exist_ok=True, parents=True)
                viz_path = viz_dir / f"{Path(args.image).stem}_concepts.png"
                inferencer.visualize_concepts(image, original, save_path=str(viz_path))
            
            elif choice == '5':
                print("Exiting...")
                break
            
            else:
                print("Invalid choice!")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)


if __name__ == '__main__':
    main()
