"""
CSR: Concept-grounded Self-interpretable Medical Image Analysis
Learns concept-grounded patch prototypes and classifies by max cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. ConvNeXtV2 backbone will not be available.")


class FeatureExtractor(nn.Module):
    """
    Backbone feature extractor (F)
    Uses ResNet-50 by default, returns spatial feature maps
    Supports: resnet50, resnet34, convnextv2_base (requires timm)
    """
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
            # Remove final pooling and FC layer to get spatial features
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
            # Remove final pooling and FC layer to get spatial features
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            
        elif backbone == 'convnextv2_base' or backbone == 'convnextv2_base.fcmae_ft_in22k_in1k':
            if not TIMM_AVAILABLE:
                raise ImportError(
                    "timm is required for ConvNeXtV2 backbone. "
                    "Install it with: pip install timm"
                )
            
            # Use timm to load ConvNeXtV2
            backbone_name = 'convnextv2_base.fcmae_ft_in22k_in1k'
            self.features = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,      # remove classification head
                features_only=False,  # get feature maps
            )
            
            # ConvNeXtV2 base has 1024 output channels
            self.feature_dim = 1024
            
            # Wrap to ensure spatial output (timm models may have different interfaces)
            self._is_timm_model = True
            
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported backbones: resnet50, resnet34, convnextv2_base"
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            f: (B, C, H', W') feature maps
        """
        features = self.features(x)
        
        # Handle timm models that might return different formats
        if hasattr(self, '_is_timm_model') and self._is_timm_model:
            # timm models with num_classes=0 return feature maps directly
            # If it's a 2D tensor, we need to reshape to spatial format
            if len(features.shape) == 2:
                # (B, C) -> (B, C, 1, 1) for compatibility
                features = features.unsqueeze(-1).unsqueeze(-1)
            elif len(features.shape) == 3:
                # (B, N, C) -> (B, C, H, W) - handle tokens from transformers
                B, N, C = features.shape
                H = W = int(N ** 0.5)
                features = features.transpose(1, 2).reshape(B, C, H, W)
        
        return features


class ConceptHead(nn.Module):
    """
    Concept head (C)
    1x1 conv that produces K concept activation maps (CAMs)
    """
    def __init__(self, in_channels: int, num_concepts: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.conv = nn.Conv2d(in_channels, num_concepts, kernel_size=1)
        
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (B, C, H, W) feature maps
        Returns:
            cam: (B, K, H, W) concept activation maps
        """
        return self.conv(f)
    
    def get_concept_predictions(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Pool CAMs to get image-level concept predictions
        Args:
            cam: (B, K, H, W) concept activation maps
        Returns:
            concept_logits: (B, K) image-level concept logits
        """
        # Global average pooling
        concept_logits = F.adaptive_avg_pool2d(cam, 1).squeeze(-1).squeeze(-1)
        return concept_logits


class Projector(nn.Module):
    """
    Projector (P)
    MLP that maps features to prototype space with L2 normalization
    """
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.proj_dim = proj_dim
        
        # Simple MLP: linear -> BN -> ReLU -> linear
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, proj_dim * 2),
            nn.BatchNorm1d(proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim * 2, proj_dim)
        )
        
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: (B, D) or (B*H*W, D) feature vectors
        Returns:
            v': (B, proj_dim) or (B*H*W, proj_dim) L2-normalized projections
        """
        v_proj = self.mlp(v)
        v_proj = F.normalize(v_proj, p=2, dim=-1)  # L2 normalize
        return v_proj


class PrototypeLayer(nn.Module):
    """
    Learnable prototypes for each concept
    Maintains M prototypes per concept (K concepts total = K*M prototypes)
    """
    def __init__(self, num_concepts: int, num_prototypes_per_concept: int, 
                 proj_dim: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.M = num_prototypes_per_concept
        self.proj_dim = proj_dim
        
        # Initialize prototypes: shape (K, M, proj_dim)
        self.prototypes = nn.Parameter(
            torch.randn(num_concepts, num_prototypes_per_concept, proj_dim)
        )
        self.normalize_prototypes()
        
    def normalize_prototypes(self):
        """L2 normalize all prototypes"""
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=-1)
    
    def forward(self) -> torch.Tensor:
        """Returns normalized prototypes"""
        return F.normalize(self.prototypes, p=2, dim=-1)
    
    def initialize_from_vectors(self, vectors_per_concept: List[torch.Tensor]):
        """
        Initialize prototypes using k-means or sampling from concept vectors
        Args:
            vectors_per_concept: List of K tensors, each (N_k, proj_dim)
        """
        with torch.no_grad():
            for k, vectors in enumerate(vectors_per_concept):
                if len(vectors) >= self.M:
                    # Use k-means or random sampling
                    indices = torch.randperm(len(vectors))[:self.M]
                    self.prototypes.data[k] = vectors[indices]
                else:
                    # If fewer vectors than prototypes, repeat
                    indices = torch.randint(0, len(vectors), (self.M,))
                    self.prototypes.data[k] = vectors[indices]
            
            self.normalize_prototypes()


class TaskHead(nn.Module):
    """
    Task head (H)
    Linear classifier from similarity scores to disease classes
    """
    def __init__(self, num_concepts: int, num_prototypes_per_concept: int, 
                 num_classes: int, use_per_concept_max: bool = True):
        super().__init__()
        self.num_concepts = num_concepts
        self.M = num_prototypes_per_concept
        self.use_per_concept_max = use_per_concept_max
        
        # Input: either K*M similarities or K max similarities
        input_dim = num_concepts if use_per_concept_max else (num_concepts * num_prototypes_per_concept)
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, similarities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarities: (B, K, M) similarity scores or (B, K*M)
        Returns:
            logits: (B, num_classes) disease classification logits
        """
        if len(similarities.shape) == 3:  # (B, K, M)
            if self.use_per_concept_max:
                # Take max over prototypes per concept
                similarities = similarities.max(dim=2)[0]  # (B, K)
            else:
                # Flatten all similarities
                similarities = similarities.view(similarities.size(0), -1)  # (B, K*M)
        
        return self.fc(similarities)


class CSRModel(nn.Module):
    """
    Complete CSR Model
    Combines all components for end-to-end training and inference
    """
    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        num_prototypes_per_concept: int = 10,
        backbone: str = 'resnet50',
        proj_dim: int = 128,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.M = num_prototypes_per_concept
        self.proj_dim = proj_dim
        
        # Components
        self.feature_extractor = FeatureExtractor(backbone, pretrained)
        self.concept_head = ConceptHead(self.feature_extractor.feature_dim, num_concepts)
        self.projector = Projector(self.feature_extractor.feature_dim, proj_dim)
        self.prototypes = PrototypeLayer(num_concepts, num_prototypes_per_concept, proj_dim)
        self.task_head = TaskHead(num_concepts, num_prototypes_per_concept, num_classes)
        
    def forward_stage_a(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage A: Concept prediction
        Args:
            x: (B, 3, H, W) input images
        Returns:
            concept_logits: (B, K) concept predictions
            cam: (B, K, H, W) concept activation maps
        """
        f = self.feature_extractor(x)  # (B, C, H, W)
        cam = self.concept_head(f)      # (B, K, H, W)
        concept_logits = self.concept_head.get_concept_predictions(cam)
        return concept_logits, cam
    
    def compute_local_concept_vectors(
        self, 
        f: torch.Tensor, 
        cam: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute local concept vectors v_k (Eq.2)
        v_k = sum_{h,w} softmax(cam_k) * f(h,w)
        
        Args:
            f: (B, C, H, W) feature maps
            cam: (B, K, H, W) concept activation maps
        Returns:
            v: (B, K, C) local concept vectors
        """
        B, C, H, W = f.shape
        K = cam.shape[1]
        
        # Reshape for efficient computation
        f_flat = f.view(B, C, -1)  # (B, C, H*W)
        cam_flat = cam.view(B, K, -1)  # (B, K, H*W)
        
        # Softmax over spatial locations per concept
        weights = F.softmax(cam_flat, dim=2)  # (B, K, H*W)
        
        # Weighted sum: v_k = sum_hw weights_k(h,w) * f(h,w)
        v = torch.einsum('bkn,bcn->bkc', weights, f_flat)  # (B, K, C)
        
        return v
    
    def forward_stage_b(
        self, 
        x: torch.Tensor, 
        concept_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage B: Compute projected concept vectors for contrastive learning
        Args:
            x: (B, 3, H, W) input images
            concept_labels: (B, K) binary concept labels
        Returns:
            v_proj: (B, K, proj_dim) projected concept vectors
            concept_labels: (B, K) concept labels (passed through)
        """
        f = self.feature_extractor(x)
        cam = self.concept_head(f)
        v = self.compute_local_concept_vectors(f, cam)  # (B, K, C)
        
        # Project each concept vector
        B, K, C = v.shape
        v_flat = v.view(B * K, C)
        v_proj = self.projector(v_flat)  # (B*K, proj_dim)
        v_proj = v_proj.view(B, K, self.proj_dim)  # (B, K, proj_dim)
        
        return v_proj, concept_labels
    
    def compute_similarity_maps(
        self, 
        f: torch.Tensor,
        importance_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute similarity maps S_km(h,w) = cos(p_km, P(f(h,w)))
        Then compute s_km = max_{h,w} S_km(h,w)
        
        Args:
            f: (B, C, H, W) feature maps
            importance_map: (B, 1, H, W) optional spatial importance (for doctor interaction)
        Returns:
            similarities: (B, K, M) max similarities per prototype
        """
        B, C, H, W = f.shape
        
        # Project all spatial features
        f_flat = f.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)
        f_proj = self.projector(f_flat)  # (B*H*W, proj_dim)
        f_proj = f_proj.view(B, H, W, self.proj_dim)  # (B, H, W, proj_dim)
        
        # Get normalized prototypes
        prototypes = self.prototypes()  # (K, M, proj_dim)
        
        # Compute cosine similarities for all prototypes
        # f_proj: (B, H, W, proj_dim)
        # prototypes: (K, M, proj_dim)
        # Result: (B, K, M, H, W)
        
        f_proj_exp = f_proj.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W, proj_dim)
        prototypes_exp = prototypes.unsqueeze(0).unsqueeze(3).unsqueeze(4)  # (1, K, M, 1, 1, proj_dim)
        
        # Cosine similarity (already normalized)
        similarity_maps = (f_proj_exp * prototypes_exp).sum(dim=-1)  # (B, K, M, H, W)
        
        # Apply importance map if provided (for doctor-in-the-loop)
        if importance_map is not None:
            importance_map = importance_map.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
            similarity_maps = similarity_maps * importance_map
        
        # Max pooling over spatial dimensions
        similarities = similarity_maps.view(B, self.num_concepts, self.M, -1).max(dim=3)[0]  # (B, K, M)
        
        return similarities
    
    def forward_stage_c(
        self, 
        x: torch.Tensor,
        rejected_concepts: Optional[torch.Tensor] = None,
        importance_map: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage C: Task classification with optional doctor-in-the-loop
        Args:
            x: (B, 3, H, W) input images
            rejected_concepts: (B, K) binary mask, 1 = rejected concept
            importance_map: (B, 1, H, W) spatial importance map
        Returns:
            task_logits: (B, num_classes) disease classification logits
            similarities: (B, K, M) similarity scores (for interpretation)
        """
        f = self.feature_extractor(x)
        similarities = self.compute_similarity_maps(f, importance_map)  # (B, K, M)
        
        # Apply concept rejection if provided
        if rejected_concepts is not None:
            # Set similarities to 0 for rejected concepts
            mask = (1 - rejected_concepts).unsqueeze(2)  # (B, K, 1)
            similarities = similarities * mask
        
        task_logits = self.task_head(similarities)
        return task_logits, similarities
    
    def forward(
        self, 
        x: torch.Tensor, 
        stage: str = 'task',
        concept_labels: Optional[torch.Tensor] = None,
        rejected_concepts: Optional[torch.Tensor] = None,
        importance_map: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with different modes for different training stages
        
        Args:
            x: (B, 3, H, W) input images
            stage: 'concept', 'prototype', or 'task'
            concept_labels: (B, K) for stage='prototype'
            rejected_concepts: (B, K) for doctor-in-the-loop
            importance_map: (B, 1, H, W) for spatial feedback
            
        Returns:
            Dictionary with relevant outputs for the stage
        """
        if stage == 'concept':
            concept_logits, cam = self.forward_stage_a(x)
            return {
                'concept_logits': concept_logits,
                'cam': cam
            }
        
        elif stage == 'prototype':
            v_proj, concept_labels = self.forward_stage_b(x, concept_labels)
            return {
                'v_proj': v_proj,
                'concept_labels': concept_labels
            }
        
        elif stage == 'task':
            task_logits, similarities = self.forward_stage_c(
                x, rejected_concepts, importance_map
            )
            return {
                'task_logits': task_logits,
                'similarities': similarities
            }
        
        else:
            raise ValueError(f"Unknown stage: {stage}")


def build_importance_map(
    H: int, 
    W: int, 
    positive_boxes: List[Tuple[int, int, int, int]], 
    negative_boxes: List[Tuple[int, int, int, int]],
    alpha: float = 0.2,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Build spatial importance map A from doctor's bounding box feedback (Eq.12)
    
    Args:
        H, W: spatial dimensions
        positive_boxes: List of (x1, y1, x2, y2) for positive regions
        negative_boxes: List of (x1, y1, x2, y2) for negative regions
        alpha: parameter for negative region suppression (default 0.2)
        device: torch device
        
    Returns:
        A: (1, 1, H, W) importance map
    """
    A = torch.ones(1, 1, H, W, device=device)
    
    # Set positive regions to 1 (already default)
    # Set negative regions to alpha
    for x1, y1, x2, y2 in negative_boxes:
        A[:, :, y1:y2, x1:x2] = alpha
    
    return A
