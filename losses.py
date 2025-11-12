"""
Loss functions for CSR model
Implements multi-prototype contrastive loss (Eq.9 from paper)
Plus enhanced losses: Focal Loss, Class-Balanced Loss, Prototype Diversity Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class FocalBCELoss(nn.Module):
    """
    Focal Loss for binary classification - focuses on hard examples
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Addresses class imbalance and focuses training on hard-to-classify examples.
    Use this for concept prediction instead of standard BCE.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for positive class (0-1). 
                   Set higher (0.75) if positive class is rare.
            gamma: Focusing parameter. Higher gamma = more focus on hard examples.
                   gamma=0 reduces to standard BCE.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, K) raw logits for K concepts
            targets: (N, K) binary labels
            
        Returns:
            Focal loss value
        """
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # p_t: probability of correct class
        p_t = torch.where(targets == 1, probs, 1 - probs)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight: balance positive/negative samples
        alpha_weight = torch.where(
            targets == 1, 
            self.alpha, 
            1 - self.alpha
        )
        
        # Final focal loss
        loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedCrossEntropyLoss(nn.Module):
    """
    Class-balanced Cross Entropy Loss
    Automatically computes inverse frequency weights for each class.
    Use this for task (disease) prediction when classes are imbalanced.
    """
    def __init__(self, num_classes: int, beta: float = 0.9999, reduction: str = 'mean'):
        """
        Args:
            num_classes: Number of classes
            beta: Reweighting parameter. beta=0 is uniform, beta→1 is inverse freq.
                  Recommended: 0.999 for moderate imbalance, 0.9999 for severe.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.reduction = reduction
        self.class_weights = None
    
    def compute_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights based on effective number of samples
        Weight_k = (1 - beta) / (1 - beta^n_k)
        where n_k is number of samples in class k
        """
        # Count samples per class
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        
        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1.0)
        
        # Effective number: (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - torch.pow(self.beta, class_counts)
        weights = (1.0 - self.beta) / effective_num
        
        # Normalize so sum = num_classes
        weights = weights / weights.sum() * self.num_classes
        
        return weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, num_classes) class logits
            targets: (N,) class labels
            
        Returns:
            Class-balanced cross entropy loss
        """
        # Compute or update weights
        if self.class_weights is None or self.training:
            self.class_weights = self.compute_weights(targets).to(logits.device)
        
        # Standard cross entropy with class weights
        loss = F.cross_entropy(
            logits, 
            targets, 
            weight=self.class_weights,
            reduction=self.reduction
        )
        
        return loss


class PrototypeDiversityLoss(nn.Module):
    """
    Prototype Diversity Regularization
    Encourages prototypes within same concept to be diverse/different.
    
    Penalizes high similarity between prototypes of the same concept.
    This prevents mode collapse where all prototypes become identical.
    """
    def __init__(self, margin: float = 0.3, reduction: str = 'mean'):
        """
        Args:
            margin: Similarity margin. Penalize if similarity > margin.
                    Lower margin = more diversity required.
                    Recommended: 0.2-0.5
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prototypes: (K, M, D) - K concepts, M prototypes each, D dimensions
            
        Returns:
            Diversity loss (lower = more diverse)
        """
        K, M, D = prototypes.shape
        
        if M <= 1:
            # No diversity loss if only 1 prototype per concept
            return torch.tensor(0.0, device=prototypes.device)
        
        total_loss = 0.0
        
        for k in range(K):
            # Get prototypes for concept k: (M, D)
            proto_k = prototypes[k]  # (M, D)
            
            # Normalize prototypes
            proto_k_norm = F.normalize(proto_k, p=2, dim=1)
            
            # Compute pairwise cosine similarity
            sim_matrix = torch.mm(proto_k_norm, proto_k_norm.t())  # (M, M)
            
            # Remove diagonal (self-similarity = 1)
            mask = 1 - torch.eye(M, device=prototypes.device)
            similarity = sim_matrix * mask
            
            # Penalize high similarity (above margin)
            # Loss = max(0, similarity - margin)
            diversity_loss = F.relu(similarity - self.margin)
            
            total_loss += diversity_loss.sum()
        
        if self.reduction == 'mean':
            # Average over all prototype pairs
            return total_loss / (K * M * (M - 1))
        else:
            return total_loss


class PrototypeUsageBalancingLoss(nn.Module):
    """
    Prototype Usage Balancing Loss
    Encourages balanced usage of prototypes during training.
    
    Computes entropy of prototype usage distribution.
    High entropy = balanced usage, Low entropy = some prototypes dominate.
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        similarities: torch.Tensor, 
        concept_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            similarities: (B, K, M) similarity scores to prototypes
            concept_labels: (B, K) binary concept labels
            
        Returns:
            Usage balancing loss (lower = more balanced)
        """
        B, K, M = similarities.shape
        
        total_loss = 0.0
        num_concepts = 0
        
        for k in range(K):
            # Only consider samples where concept k is present
            concept_mask = concept_labels[:, k] > 0.5
            
            if concept_mask.sum() == 0:
                continue  # Skip if concept not present in batch
            
            # Get similarities for this concept
            sim_k = similarities[concept_mask, k, :]  # (N_k, M)
            
            # Which prototype is most similar for each sample
            proto_indices = sim_k.argmax(dim=1)  # (N_k,)
            
            # Count usage of each prototype
            usage_counts = torch.bincount(
                proto_indices, 
                minlength=M
            ).float()  # (M,)
            
            # Convert to probability distribution
            usage_dist = usage_counts / (usage_counts.sum() + 1e-8)
            
            # Compute entropy: H = -sum(p * log(p))
            # High entropy = balanced usage
            entropy = -(usage_dist * torch.log(usage_dist + 1e-8)).sum()
            
            # Maximum possible entropy (uniform distribution)
            max_entropy = np.log(M)
            
            # Loss = (max_entropy - entropy) / max_entropy
            # 0 = perfectly balanced, 1 = completely imbalanced
            loss = (max_entropy - entropy) / max_entropy
            
            total_loss += loss
            num_concepts += 1
        
        if num_concepts == 0:
            return torch.tensor(0.0, device=similarities.device)
        
        if self.reduction == 'mean':
            return total_loss / num_concepts
        else:
            return total_loss
