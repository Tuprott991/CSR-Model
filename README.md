# CSR: Concept-grounded Self-interpretable Medical Image Analysis

PyTorch implementation of CSR (Concept-grounded Self-interpretable Reasoning), a framework that learns concept-grounded patch prototypes and classifies medical images by maximum cosine similarity between prototypes and patch features. Prototypes are semantically mapped to human concepts and can be refined by doctors at train/test time.

## Overview

CSR learns interpretable representations by:
1. **Concept Head**: Predicts K medical concepts (findings) using concept activation maps (CAMs)
2. **Prototypes**: Learns M prototypes per concept in a projected space
3. **Task Head**: Classifies diseases using max similarity between prototypes and image patches
4. **Doctor-in-the-Loop**: Enables concept rejection and spatial feedback for refinement

## Key Features

- ✅ Three-stage training pipeline (Concept → Prototype → Task)
- ✅ Multi-prototype contrastive learning with SoftTriple loss
- ✅ Doctor-in-the-loop interactions (concept-level and spatial-level)
- ✅ Support for TBX11K, VinDr-CXR, and ISIC datasets
- ✅ Comprehensive evaluation metrics
- ✅ Visualization tools for CAMs and similarity maps

## Architecture

```
Image → Feature Extractor (F) → Concept Head (C) → Local Concept Vectors (v_k)
                                        ↓
                                   Projector (P)
                                        ↓
                            Prototypes {p_km} (K×M)
                                        ↓
                              Similarity Computation
                                        ↓
                                  Task Head (H) → Disease Classification
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.10
- torchvision >= 0.11
- scikit-learn >= 0.24
- Pillow >= 8.0
- numpy >= 1.19
- pandas >= 1.2
- tqdm >= 4.60
- matplotlib >= 3.3

## Dataset Preparation

### Expected Format

For each dataset, you need:
1. **Images**: Medical images (JPG/PNG)
2. **Annotations**: Concept labels (findings) and class labels (diseases)

### TBX11K Format

```json
{
  "image_001.png": {
    "findings": ["infiltration", "effusion"],
    "disease": "tuberculosis"
  },
  ...
}
```

### VinDr-CXR / ISIC Format

CSV file with columns:
- `image_id` or `image`: Image filename
- Multiple binary columns for concepts (0/1)
- `disease` or `diagnosis`: Disease class label

### Directory Structure

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── train_annotations.json (or .csv)
├── val_annotations.json
└── test_annotations.json
```

## Training

### Three-Stage Training Pipeline

#### Stage A: Train Concept Model

Trains the feature extractor (F) and concept head (C) for multi-label concept prediction.

```bash
python train.py \
    --dataset tbx11k \
    --data_root ./data/images \
    --train_file ./data/train_annotations.json \
    --val_file ./data/val_annotations.json \
    --test_file ./data/test_annotations.json \
    --backbone resnet50 \
    --num_prototypes 10 \
    --proj_dim 128 \
    --batch_size 32 \
    --stage_a_epochs 30 \
    --stage_b_epochs 20 \
    --stage_c_epochs 30 \
    --save_dir checkpoints/tbx11k \
    --device cuda
```

#### Stage B: Learn Prototypes

Trains the projector (P) and prototypes using multi-prototype contrastive loss. Backbone is frozen.

#### Stage C: Train Task Head

Trains the task head (H) for disease classification using prototype similarities. Optionally fine-tunes the entire model.

### Training from Specific Stage

```bash
# Start from Stage B (load Stage A checkpoint)
python train.py \
    --start_stage b \
    --load_checkpoint checkpoints/stage_a_best.pth \
    [other args...]

# Start from Stage C (load Stage B checkpoint)
python train.py \
    --start_stage c \
    --load_checkpoint checkpoints/stage_b_best.pth \
    [other args...]
```

### Hyperparameters

Key hyperparameters (from paper):

- **Stage A**:
  - Learning rate: 1e-4 (concept head), 1e-5 (backbone)
  - Epochs: 20-40
  - Loss: BCE for multi-label concepts

- **Stage B**:
  - Learning rate: 1e-3
  - Epochs: 10-30
  - λ (lambda_temp): 10-50
  - γ (gamma): 10-50
  - δ (margin): 0.01-0.1

- **Stage C**:
  - Learning rate: 1e-4 (task head), 1e-5 (backbone)
  - Epochs: 20-40
  - Loss: Cross-entropy for disease classification

## Evaluation

### Comprehensive Evaluation

```bash
python evaluate.py \
    --dataset tbx11k \
    --data_root ./data/images \
    --train_file ./data/train_annotations.json \
    --val_file ./data/val_annotations.json \
    --test_file ./data/test_annotations.json \
    --checkpoint checkpoints/stage_c_best.pth \
    --backbone resnet50 \
    --num_prototypes 10 \
    --output_dir results/
```

### Evaluation Metrics

- **Task Performance**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Concept Performance**: Per-concept and overall metrics
- **Prototype Analysis**: Usage frequency per prototype
- **Robustness**: Performance with concept rejection

### Output

Results are saved as JSON in `results/evaluation_report.json`.

## Doctor-in-the-Loop Interactions

### Example Usage

```python
from model import CSRModel
from doctor_interaction import DoctorInteraction
import torch

# Load model
model = CSRModel(num_concepts=15, num_classes=3, num_prototypes_per_concept=10)
model.load_state_dict(torch.load('checkpoints/stage_c_best.pth')['model_state_dict'])

# Create interaction interface
doctor = DoctorInteraction(model, device='cuda')

# Load and preprocess image
image = load_and_preprocess_image('patient_xray.jpg')  # (1, 3, 224, 224)

# Get interactive diagnosis
result = doctor.interactive_diagnosis(
    image,
    concept_names=['infiltration', 'effusion', ...],
    class_names=['tuberculosis', 'pneumonia', 'normal']
)

print(f"Predicted class: {result['predicted_class']}")
print(f"Top concepts: {result['top_concepts']}")
```

### Concept-Level Rejection

```python
# Reject specific concepts (e.g., concept IDs 2, 5, 7)
rejected_logits = doctor.reject_concepts(image, rejected_concept_ids=[2, 5, 7])
```

### Spatial-Level Feedback

```python
# Define bounding boxes (x1, y1, x2, y2) in feature map coordinates
positive_boxes = [(10, 10, 50, 50)]  # Important regions
negative_boxes = [(100, 100, 150, 150)]  # Unimportant regions

# Inference with spatial feedback
modified_logits = doctor.inference_with_spatial_feedback(
    image,
    positive_boxes=positive_boxes,
    negative_boxes=negative_boxes,
    alpha=0.2  # Suppression factor
)
```

### Visualization

```python
# Visualize concept activation map
cam_viz = doctor.visualize_concept_activation(
    image,
    concept_id=3,
    original_image=original_img_array
)

# Visualize prototype similarity map
sim_viz = doctor.visualize_similarity_maps(
    image,
    concept_id=3,
    prototype_id=0,
    original_image=original_img_array
)
```

## Model Components

### Feature Extractor (F)
- **Backbone**: ResNet-50/34 or ConvNeXtV2-Base (pretrained on ImageNet)
- **Output**: Spatial feature maps (B, C, H, W)
- **Supported backbones**:
  - `resnet50` (2048 channels) - Default
  - `resnet34` (512 channels) - Lighter weight
  - `convnextv2_base` (1024 channels) - State-of-the-art CNN (requires timm)

### Concept Head (C)
- **Architecture**: 1×1 convolution
- **Output**: K concept activation maps (CAMs)
- **Loss**: Binary cross-entropy (multi-label)

### Projector (P)
- **Architecture**: 2-layer MLP with BN and ReLU
- **Output**: L2-normalized embeddings (proj_dim)

### Prototypes
- **Number**: K concepts × M prototypes per concept
- **Representation**: L2-normalized vectors in projected space
- **Initialization**: K-means clustering on concept vectors
- **Learning**: Multi-prototype contrastive loss

### Task Head (H)
- **Input**: Similarity scores (B, K, M) or max-pooled (B, K)
- **Architecture**: Linear classifier
- **Output**: Disease classification logits

## Files Structure

```
CSR-Model/
├── model.py                  # CSR model architecture
├── dataloader.py            # Dataset loaders (TBX11K, VinDr-CXR, ISIC)
├── train.py                 # Three-stage training script
├── losses.py                # Loss functions (contrastive, BCE, CE)
├── evaluate.py              # Evaluation script
├── doctor_interaction.py    # Doctor-in-the-loop interactions
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Citation

If you use this implementation, please cite the original CSR paper:

```bibtex
@inproceedings{csr2024,
  title={Interactive Medical Image Analysis with Concept-grounded Self-interpretable Reasoning},
  author={[Authors]},
  booktitle={[Conference/Journal]},
  year={2024}
}
```

## Key Equations (from Paper)

### Local Concept Vector (Eq. 2)
```
v_k = Σ_{h,w} softmax(cam_k(h,w)) · f(h,w)
```

### Soft Assignment (Eq. 6)
```
q_m = softmax(γ · ⟨p_km, v'⟩)
```

### Weighted Similarity (Eq. 7-8)
```
sim_k(v') = Σ_m q_m · ⟨p_km, v'⟩
```

### Multi-Prototype Contrastive Loss (Eq. 9)
```
ℓ = -log(exp(λ(sim_pos + δ)) / Σ_k exp(λ · sim_k))
```

### Similarity Map (Eq. 10)
```
S_km(h,w) = cos(p_km, P(f(h,w)))
s_km = max_{h,w} S_km(h,w)
```

### Importance Map (Eq. 12)
```
A(h,w) = 1.0 for positive regions
A(h,w) = α for negative regions
```

## Backbone Options

The model supports three backbone architectures:

| Backbone | Feature Channels | Parameters | Speed | Accuracy | Requirements |
|----------|-----------------|------------|-------|----------|--------------|
| `resnet34` | 512 | ~21M | Fast | Good | torchvision |
| `resnet50` | 2048 | ~23M | Medium | Better | torchvision (default) |
| `convnextv2_base` | 1024 | ~89M | Slower | Best | timm |

**Using ConvNeXtV2**:
```bash
# Install timm first
pip install timm

# Train with ConvNeXtV2
python train.py --backbone convnextv2_base [other args...]
```

## Tips and Best Practices

1. **Data Preprocessing**: Resize images to 224×224 or 256×256, normalize with ImageNet stats
2. **Backbone Selection**: ResNet-34 for speed, ResNet-50 for balance, ConvNeXtV2 for accuracy
3. **Prototype Initialization**: Use K-means on training concept vectors for better initialization
4. **Learning Rates**: Use lower LR for backbone (1e-5) vs. heads (1e-4)
5. **Stage B**: Freeze backbone to focus on learning good prototype representations
6. **Stage C**: Fine-tune entire model for best performance
7. **Imbalanced Data**: Use class/concept weights in loss functions
8. **Doctor Feedback**: Start with α=0.2 for spatial suppression

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller backbone (ResNet-34)
2. **Poor Concept Accuracy**: Increase Stage A epochs or use concept weights
3. **Prototypes Not Separating**: Adjust λ, γ hyperparameters in Stage B
4. **timm Not Found**: Install timm for ConvNeXtV2: `pip install timm`
4. **Low Task Accuracy**: Ensure Stage A and B converged before Stage C

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
