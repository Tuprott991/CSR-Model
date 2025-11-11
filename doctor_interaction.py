"""
Doctor-in-the-loop interaction module
Implements concept-level and spatial-level feedback mechanisms
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DoctorInteraction:
    """
    Handles doctor-in-the-loop interactions:
    1. Concept-level: Reject/accept concepts
    2. Spatial-level: Draw bounding boxes for positive/negative regions
    """
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def get_top_concepts(
        self,
        image: torch.Tensor,
        top_k: int = 5
    ) -> Dict[int, float]:
        """
        Get top-k most activated concepts for an image
        
        Args:
            image: (1, 3, H, W) input image
            top_k: number of top concepts to return
            
        Returns:
            Dict mapping concept_id -> max similarity score
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get similarities
            outputs = self.model(image, stage='task')
            similarities = outputs['similarities']  # (1, K, M)
            
            # Get max similarity per concept
            concept_scores = similarities.max(dim=2)[0].squeeze(0)  # (K,)
            
            # Get top-k
            top_values, top_indices = torch.topk(concept_scores, top_k)
            
            result = {
                int(idx): float(val) 
                for idx, val in zip(top_indices.cpu(), top_values.cpu())
            }
        
        return result
    
    def reject_concepts(
        self,
        image: torch.Tensor,
        rejected_concept_ids: List[int]
    ) -> torch.Tensor:
        """
        Perform inference with rejected concepts (set their similarities to 0)
        
        Args:
            image: (1, 3, H, W) input image
            rejected_concept_ids: List of concept IDs to reject
            
        Returns:
            task_logits: (1, num_classes) modified predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            # Create rejection mask
            K = self.model.num_concepts
            rejected_mask = torch.zeros(1, K, device=self.device)
            for concept_id in rejected_concept_ids:
                rejected_mask[0, concept_id] = 1
            
            # Forward with rejection
            outputs = self.model(
                image, 
                stage='task',
                rejected_concepts=rejected_mask
            )
            task_logits = outputs['task_logits']
        
        return task_logits
    
    def build_importance_map(
        self,
        height: int,
        width: int,
        positive_boxes: List[Tuple[int, int, int, int]],
        negative_boxes: List[Tuple[int, int, int, int]],
        alpha: float = 0.2
    ) -> torch.Tensor:
        """
        Build spatial importance map from bounding boxes (Eq.12)
        
        Args:
            height, width: spatial dimensions of feature map
            positive_boxes: List of (x1, y1, x2, y2) for important regions
            negative_boxes: List of (x1, y1, x2, y2) for unimportant regions
            alpha: suppression factor for negative regions (default 0.2)
            
        Returns:
            importance_map: (1, 1, H, W) spatial importance weights
        """
        # Initialize all to 1.0
        importance_map = torch.ones(1, 1, height, width, device=self.device)
        
        # Set negative regions to alpha
        for x1, y1, x2, y2 in negative_boxes:
            importance_map[:, :, y1:y2, x1:x2] = alpha
        
        # Positive regions remain 1.0 (already initialized)
        # If you want to explicitly set them:
        for x1, y1, x2, y2 in positive_boxes:
            importance_map[:, :, y1:y2, x1:x2] = 1.0
        
        return importance_map
    
    def inference_with_spatial_feedback(
        self,
        image: torch.Tensor,
        positive_boxes: List[Tuple[int, int, int, int]],
        negative_boxes: List[Tuple[int, int, int, int]],
        alpha: float = 0.2
    ) -> torch.Tensor:
        """
        Perform inference with spatial feedback from doctor
        
        Args:
            image: (1, 3, H, W) input image
            positive_boxes: Boxes for important regions (in feature map coords)
            negative_boxes: Boxes for unimportant regions (in feature map coords)
            alpha: suppression factor
            
        Returns:
            task_logits: (1, num_classes) modified predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get feature map size
            f = self.model.feature_extractor(image)
            _, _, H, W = f.shape
            
            # Build importance map
            importance_map = self.build_importance_map(
                H, W, positive_boxes, negative_boxes, alpha
            )
            
            # Forward with importance map
            outputs = self.model(
                image,
                stage='task',
                importance_map=importance_map
            )
            task_logits = outputs['task_logits']
        
        return task_logits
    
    def visualize_concept_activation(
        self,
        image: torch.Tensor,
        concept_id: int,
        original_image: np.ndarray = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize concept activation map (CAM) overlaid on image
        
        Args:
            image: (1, 3, H, W) preprocessed input
            concept_id: which concept to visualize
            original_image: (H, W, 3) original image for overlay
            alpha: transparency for overlay
            
        Returns:
            visualization: (H, W, 3) image with CAM overlay
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get feature maps and CAMs
            f = self.model.feature_extractor(image)
            cam = self.model.concept_head(f)  # (1, K, H, W)
            
            # Get CAM for specific concept
            concept_cam = cam[0, concept_id].cpu().numpy()  # (H, W)
            
            # Normalize to [0, 1]
            concept_cam = (concept_cam - concept_cam.min()) / (concept_cam.max() - concept_cam.min() + 1e-8)
            
            # Resize to original image size if provided
            if original_image is not None:
                H, W = original_image.shape[:2]
                concept_cam = np.array(
                    Image.fromarray((concept_cam * 255).astype(np.uint8)).resize(
                        (W, H), Image.BILINEAR
                    )
                ) / 255.0
            else:
                # Use input image
                _, _, H, W = image.shape
                original_image = image[0].permute(1, 2, 0).cpu().numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                original_image = original_image * std + mean
                original_image = np.clip(original_image, 0, 1)
            
            # Create heatmap
            import matplotlib.cm as cm
            heatmap = cm.jet(concept_cam)[:, :, :3]  # (H, W, 3)
            
            # Overlay
            visualization = (1 - alpha) * original_image + alpha * heatmap
            visualization = np.clip(visualization, 0, 1)
        
        return visualization
    
    def visualize_similarity_maps(
        self,
        image: torch.Tensor,
        concept_id: int,
        prototype_id: int,
        original_image: np.ndarray = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize similarity map for a specific prototype
        
        Args:
            image: (1, 3, H, W) preprocessed input
            concept_id: which concept
            prototype_id: which prototype within concept
            original_image: (H, W, 3) original image
            alpha: transparency
            
        Returns:
            visualization: (H, W, 3) image with similarity map overlay
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get feature maps
            f = self.model.feature_extractor(image)  # (1, C, H, W)
            B, C, H, W = f.shape
            
            # Project features
            f_flat = f.permute(0, 2, 3, 1).reshape(B * H * W, C)
            f_proj = self.model.projector(f_flat)
            f_proj = f_proj.view(B, H, W, self.model.proj_dim)
            
            # Get prototype
            prototypes = self.model.prototypes()
            prototype = prototypes[concept_id, prototype_id]  # (proj_dim,)
            
            # Compute similarity map
            similarity_map = torch.matmul(
                f_proj[0], prototype
            ).cpu().numpy()  # (H, W)
            
            # Normalize
            similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-8)
            
            # Resize and overlay (similar to above)
            if original_image is not None:
                H_img, W_img = original_image.shape[:2]
                similarity_map = np.array(
                    Image.fromarray((similarity_map * 255).astype(np.uint8)).resize(
                        (W_img, H_img), Image.BILINEAR
                    )
                ) / 255.0
            else:
                original_image = image[0].permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                original_image = original_image * std + mean
                original_image = np.clip(original_image, 0, 1)
            
            # Create heatmap
            import matplotlib.cm as cm
            heatmap = cm.jet(similarity_map)[:, :, :3]
            
            # Overlay
            visualization = (1 - alpha) * original_image + alpha * heatmap
            visualization = np.clip(visualization, 0, 1)
        
        return visualization
    
    def interactive_diagnosis(
        self,
        image: torch.Tensor,
        concept_names: List[str] = None,
        class_names: List[str] = None
    ) -> Dict:
        """
        Complete interactive diagnosis workflow
        
        Args:
            image: (1, 3, H, W) input image
            concept_names: Optional list of concept names
            class_names: Optional list of class names
            
        Returns:
            Dict with predictions, top concepts, and feedback options
        """
        self.model.eval()
        
        with torch.no_grad():
            # Initial prediction
            outputs = self.model(image, stage='task')
            task_logits = outputs['task_logits']
            similarities = outputs['similarities']
            
            # Get predictions
            probs = F.softmax(task_logits, dim=1)[0]
            pred_class = probs.argmax().item()
            
            # Get top concepts
            concept_scores = similarities.max(dim=2)[0].squeeze(0)
            top_k = min(10, self.model.num_concepts)
            top_values, top_indices = torch.topk(concept_scores, top_k)
            
            top_concepts = []
            for idx, score in zip(top_indices.cpu(), top_values.cpu()):
                concept_info = {
                    'id': int(idx),
                    'score': float(score)
                }
                if concept_names:
                    concept_info['name'] = concept_names[int(idx)]
                top_concepts.append(concept_info)
            
            # Format predictions
            predictions = []
            for i, prob in enumerate(probs.cpu().numpy()):
                pred_info = {
                    'class_id': i,
                    'probability': float(prob)
                }
                if class_names:
                    pred_info['name'] = class_names[i]
                predictions.append(pred_info)
            
            predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
        
        return {
            'predicted_class': pred_class,
            'predictions': predictions,
            'top_concepts': top_concepts,
            'raw_logits': task_logits[0].cpu().numpy().tolist(),
            'raw_similarities': similarities[0].cpu().numpy().tolist()
        }


def convert_image_boxes_to_feature_boxes(
    boxes: List[Tuple[int, int, int, int]],
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int]
) -> List[Tuple[int, int, int, int]]:
    """
    Convert bounding boxes from image coordinates to feature map coordinates
    
    Args:
        boxes: List of (x1, y1, x2, y2) in image coordinates
        image_size: (H, W) image dimensions
        feature_size: (H, W) feature map dimensions
        
    Returns:
        List of boxes in feature map coordinates
    """
    H_img, W_img = image_size
    H_feat, W_feat = feature_size
    
    scale_h = H_feat / H_img
    scale_w = W_feat / W_img
    
    feature_boxes = []
    for x1, y1, x2, y2 in boxes:
        feat_x1 = int(x1 * scale_w)
        feat_y1 = int(y1 * scale_h)
        feat_x2 = int(x2 * scale_w)
        feat_y2 = int(y2 * scale_h)
        
        # Ensure within bounds
        feat_x1 = max(0, min(feat_x1, W_feat - 1))
        feat_x2 = max(0, min(feat_x2, W_feat))
        feat_y1 = max(0, min(feat_y1, H_feat - 1))
        feat_y2 = max(0, min(feat_y2, H_feat))
        
        feature_boxes.append((feat_x1, feat_y1, feat_x2, feat_y2))
    
    return feature_boxes


def visualize_boxes_on_image(
    image: np.ndarray,
    positive_boxes: List[Tuple[int, int, int, int]],
    negative_boxes: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Visualize bounding boxes on image
    
    Args:
        image: (H, W, 3) RGB image
        positive_boxes: List of (x1, y1, x2, y2) for positive regions
        negative_boxes: List of (x1, y1, x2, y2) for negative regions
        
    Returns:
        image with boxes drawn
    """
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # Draw positive boxes (green)
    for x1, y1, x2, y2 in positive_boxes:
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, 'Positive', color='green', fontsize=12, weight='bold')
    
    # Draw negative boxes (red)
    for x1, y1, x2, y2 in negative_boxes:
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, 'Negative', color='red', fontsize=12, weight='bold')
    
    ax.axis('off')
    
    # Convert to numpy array
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return vis_image
