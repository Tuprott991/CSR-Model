"""
Loss functions for CSR model
Implements multi-prototype contrastive loss (Eq.9 from paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPrototypeContrastiveLoss(nn.Module):
    """
    Multi-prototype contrastive loss (SoftTriple-like)
    Based on Equations 6-9 from the CSR paper
    
    For each concept vector v', computes soft assignment to prototypes
    within its concept, then contrasts with all other concepts.
    """
    def __init__(
        self, 
        lambda_temp: float = 20.0,
        gamma: float = 20.0,
        margin: float = 0.05
    ):
        """
        Args:
            lambda_temp: Temperature scaling for final contrastive loss (λ)
            gamma: Temperature for soft assignment within concept (γ)
            margin: Margin for positive concept (δ)
        """
        super().__init__()
        self.lambda_temp = lambda_temp
        self.gamma = gamma
        self.margin = margin
    
    def compute_soft_assignment(
        self, 
        v_proj: torch.Tensor, 
        prototypes: torch.Tensor, 
        concept_idx: int
    ) -> torch.Tensor:
        """
        Compute soft assignment q_m (Eq.6)
        q_m = softmax(γ * <p_km, v'>)
        
        Args:
            v_proj: (N, proj_dim) projected concept vectors
            prototypes: (M, proj_dim) prototypes for concept k
            concept_idx: which concept k
            
        Returns:
            q: (N, M) soft assignment weights
        """
        # Cosine similarity (already normalized)
        similarities = torch.matmul(v_proj, prototypes.t())  # (N, M)
        
        # Softmax with temperature
        q = F.softmax(self.gamma * similarities, dim=1)  # (N, M)
        
        return q
    
    def compute_weighted_similarity(
        self,
        v_proj: torch.Tensor,
        prototypes: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted similarity sim_k(v') (Eq.7-8)
        sim_k(v') = sum_m q_m * <p_km, v'>
        
        Args:
            v_proj: (N, proj_dim) projected concept vectors
            prototypes: (M, proj_dim) prototypes for concept k
            q: (N, M) soft assignment weights
            
        Returns:
            sim: (N,) weighted similarity scores
        """
        # Cosine similarity
        similarities = torch.matmul(v_proj, prototypes.t())  # (N, M)
        
        # Weighted sum
        sim = (q * similarities).sum(dim=1)  # (N,)
        
        return sim
    
    def forward(
        self,
        v_proj: torch.Tensor,
        concept_labels: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-prototype contrastive loss (Eq.9)
        
        Args:
            v_proj: (B, K, proj_dim) projected concept vectors
            concept_labels: (B, K) binary concept labels
            prototypes: (K, M, proj_dim) all prototypes
            
        Returns:
            loss: scalar contrastive loss
        """
        B, K, proj_dim = v_proj.shape
        M = prototypes.shape[1]
        
        total_loss = 0.0
        num_positives = 0
        
        # For each sample and each positive concept
        for b in range(B):
            # Get positive concepts for this sample
            positive_concepts = torch.where(concept_labels[b] == 1)[0]
            
            for pos_k in positive_concepts:
                v = v_proj[b, pos_k]  # (proj_dim,)
                
                # Compute soft assignment for positive concept
                q_pos = self.compute_soft_assignment(
                    v.unsqueeze(0), 
                    prototypes[pos_k], 
                    pos_k
                )  # (1, M)
                
                # Compute weighted similarity for positive concept with margin
                sim_pos = self.compute_weighted_similarity(
                    v.unsqueeze(0),
                    prototypes[pos_k],
                    q_pos
                ).squeeze(0)  # scalar
                
                sim_pos = sim_pos + self.margin
                
                # Compute similarities to all concepts
                sim_all = []
                for k in range(K):
                    # Soft assignment for concept k
                    q_k = self.compute_soft_assignment(
                        v.unsqueeze(0),
                        prototypes[k],
                        k
                    )  # (1, M)
                    
                    # Weighted similarity to concept k
                    sim_k = self.compute_weighted_similarity(
                        v.unsqueeze(0),
                        prototypes[k],
                        q_k
                    ).squeeze(0)  # scalar
                    
                    sim_all.append(sim_k)
                
                sim_all = torch.stack(sim_all)  # (K,)
                
                # Contrastive loss (Eq.9)
                # ℓ = -log(exp(λ * sim_pos) / sum_k exp(λ * sim_k))
                numerator = torch.exp(self.lambda_temp * sim_pos)
                denominator = torch.sum(torch.exp(self.lambda_temp * sim_all))
                
                loss = -torch.log(numerator / (denominator + 1e-8))
                
                total_loss += loss
                num_positives += 1
        
        # Average over all positive concept instances
        if num_positives > 0:
            total_loss = total_loss / num_positives
        
        return total_loss


class ConceptBCELoss(nn.Module):
    """
    BCE loss for multi-label concept classification
    Used in Stage A
    """
    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, K) concept logits
            labels: (B, K) binary concept labels
        """
        return self.criterion(logits, labels)


class TaskCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for disease classification
    Used in Stage C
    """
    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) task logits
            labels: (B,) class labels
        """
        return self.criterion(logits, labels)


class CSRCombinedLoss(nn.Module):
    """
    Combined loss for joint training (optional)
    Can be used for end-to-end fine-tuning
    """
    def __init__(
        self,
        concept_weight: float = 1.0,
        task_weight: float = 1.0
    ):
        super().__init__()
        self.concept_weight = concept_weight
        self.task_weight = task_weight
        self.concept_loss = ConceptBCELoss()
        self.task_loss = TaskCrossEntropyLoss()
    
    def forward(
        self,
        concept_logits: torch.Tensor,
        concept_labels: torch.Tensor,
        task_logits: torch.Tensor,
        task_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss
        """
        loss_concept = self.concept_loss(concept_logits, concept_labels)
        loss_task = self.task_loss(task_logits, task_labels)
        
        total_loss = (self.concept_weight * loss_concept + 
                      self.task_weight * loss_task)
        
        return total_loss
