# 🔮 CSR Model Inference Guide

Complete guide for using `infer.py` with the CSR model.

## 🚀 Quick Start

### 1. Single Image Inference (Basic)

```bash
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 \
  --num_classes 3 \
  --backbone convnextv2_base \
  --num_prototypes 10 \
  --proj_dim 128 \
  --mode single \
  --image path/to/image.png
```

**Output:**
```
Predicted Class: active_tuberculosis
Confidence: 0.9234

All Class Probabilities:
  active_tuberculosis: 0.9234
  latent_tuberculosis: 0.0521
  normal: 0.0245

Top Predicted Concepts:
  has_tb_lesion: 0.8912
  has_active_tb: 0.7654
  lesion_size_large: 0.6543
```

---

### 2. Single Image with Concept Names (Better Labels)

```bash
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 \
  --num_classes 3 \
  --class_names active_tuberculosis latent_tuberculosis normal \
  --concept_names has_tb_lesion has_active_tb has_latent_tb lesion_count_low lesion_count_medium lesion_size_large lesion_size_medium \
  --mode single \
  --image patient001.png \
  --visualize \
  --output results/patient001.json
```

**Creates:**
- `results/patient001.json` - Detailed prediction results
- `visualizations/patient001_concepts.png` - Concept activation heatmaps

---

### 3. Concept Rejection (Doctor-in-the-Loop)

```bash
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 \
  --num_classes 3 \
  --class_names active_tuberculosis latent_tuberculosis normal \
  --concept_names has_tb_lesion has_active_tb has_latent_tb lesion_count_low lesion_count_medium lesion_size_large lesion_size_medium \
  --mode single \
  --image patient001.png \
  --reject_concepts has_active_tb lesion_size_large
```

**Output shows before/after:**
```
--- Standard Prediction ---
Predicted Class: active_tuberculosis
Confidence: 0.9234

--- Prediction with Rejected Concepts ---
Rejecting: has_active_tb, lesion_size_large

New Predicted Class: latent_tuberculosis
New Confidence: 0.6543

Change in probabilities:
  active_tuberculosis: 0.9234 ↓ 0.2145 (-0.7089)
  latent_tuberculosis: 0.0521 ↑ 0.6543 (+0.6022)
  normal: 0.0245 ↑ 0.1312 (+0.1067)
```

---

### 4. Batch Inference on Directory

```bash
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 \
  --num_classes 3 \
  --class_names active_tuberculosis latent_tuberculosis normal \
  --concept_names has_tb_lesion has_active_tb has_latent_tb lesion_count_low lesion_count_medium lesion_size_large lesion_size_medium \
  --mode batch \
  --image_dir test_images/ \
  --output batch_results.json \
  --visualize \
  --viz_dir batch_visualizations/
```

**Processes all images in directory and creates:**
- `batch_results.json` - All predictions
- `batch_visualizations/` - Concept visualizations for each image

---

### 5. Detailed Visualization Mode

```bash
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 \
  --num_classes 3 \
  --concept_names has_tb_lesion has_active_tb has_latent_tb lesion_count_low lesion_count_medium lesion_size_large lesion_size_medium \
  --mode visualize \
  --image patient001.png \
  --top_k 5 \
  --viz_dir detailed_viz/
```

**Creates detailed visualizations:**
- `detailed_viz/patient001_concepts.png` - Top 5 concept activation maps
- `detailed_viz/patient001_prototypes_c0.png` - All 10 prototypes for top concept

---

### 6. Interactive Mode (Real-time Exploration)

```bash
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 \
  --num_classes 3 \
  --class_names active_tuberculosis latent_tuberculosis normal \
  --concept_names has_tb_lesion has_active_tb has_latent_tb lesion_count_low lesion_count_medium lesion_size_large lesion_size_medium \
  --mode interactive \
  --image patient001.png
```

**Interactive menu:**
```
Interactive Mode - Options:
  1. View top concepts
  2. Reject concepts
  3. View current prediction
  4. Visualize concepts
  5. Exit

Enter choice (1-5): 2

Available concepts:
  0. has_tb_lesion
  1. has_active_tb
  2. has_latent_tb
  ...

Enter concept numbers to reject (comma-separated): 1,5

New Prediction: latent_tuberculosis
New Confidence: 0.6543
```

---

## 📋 All Command-Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--checkpoint` | str | Path to trained model checkpoint |
| `--num_concepts` | int | Number of concepts (K) in model |
| `--num_classes` | int | Number of disease classes |

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--backbone` | str | `resnet50` | Backbone architecture |
| `--num_prototypes` | int | `10` | Prototypes per concept (M) |
| `--proj_dim` | int | `128` | Projection dimension |
| `--device` | str | `cuda` | Device: `cuda` or `cpu` |

### Labels (Optional but Recommended)

| Argument | Type | Description |
|----------|------|-------------|
| `--class_names` | str+ | Disease class names (space-separated) |
| `--concept_names` | str+ | Concept names (space-separated) |

### Inference Mode

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--mode` | str | `single` | `single`, `batch`, `interactive`, `visualize` | Inference mode |

### Input/Output

| Argument | Type | Description |
|----------|------|-------------|
| `--image` | str | Path to single image (for `single`, `interactive`, `visualize` modes) |
| `--image_dir` | str | Directory of images (for `batch` mode) |
| `--output` | str | Output JSON file for results |
| `--visualize` | flag | Save concept visualizations |
| `--viz_dir` | str | Directory for visualizations (default: `visualizations/`) |

### Interactive Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--reject_concepts` | str+ | None | Concept names to reject (space-separated) |
| `--top_k` | int | `5` | Number of top concepts to show/visualize |

---

## 💡 Use Cases

### Use Case 1: Radiologist Review
**Scenario:** Doctor wants to verify model's reasoning

```bash
# Get prediction with concept explanations
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 --num_classes 3 \
  --class_names active_tb latent_tb normal \
  --concept_names has_lesion active latent count_low count_med size_large size_med \
  --mode single \
  --image xray_suspicious.png \
  --visualize \
  --output doctor_review/case_001.json
```

Doctor reviews:
1. Prediction: active_tb (92%)
2. Concepts: has_lesion ✓, active ✓, size_large ✓
3. Heatmaps show lesions correctly located
4. **Decision:** Confirms model prediction

---

### Use Case 2: Correcting Misdiagnosis
**Scenario:** Model says "active TB" but doctor sees it's actually "latent TB"

```bash
# Test by rejecting "has_active_tb" concept
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 --num_classes 3 \
  --class_names active_tb latent_tb normal \
  --concept_names has_lesion active latent count_low count_med size_large size_med \
  --mode single \
  --image misdiagnosed_case.png \
  --reject_concepts active
```

**Result:**
- Before: active_tb (85%)
- After rejecting "active": latent_tb (72%)
- **Insight:** Model relied too much on "active" concept, needs more training on distinguishing active vs latent

---

### Use Case 3: Batch Screening
**Scenario:** Screen 500 patients, prioritize high-risk cases

```bash
# Batch process all cases
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 --num_classes 3 \
  --class_names active_tb latent_tb normal \
  --mode batch \
  --image_dir patient_xrays/ \
  --output screening_results.json

# Then filter high-risk cases in Python:
import json
with open('screening_results.json') as f:
    results = json.load(f)

high_risk = [
    r for r in results 
    if r['predicted_class'] == 'active_tb' and r['confidence'] > 0.8
]

print(f"High-risk cases: {len(high_risk)}")
```

---

### Use Case 4: Model Debugging
**Scenario:** Understand why model fails on edge cases

```bash
# Interactive exploration
python infer.py \
  --checkpoint checkpoints/tbx11k/stage_c_best.pth \
  --num_concepts 7 --num_classes 3 \
  --concept_names has_lesion active latent count_low count_med size_large size_med \
  --mode interactive \
  --image edge_case_017.png

# In interactive mode:
# 1. View top concepts → See which concepts activated
# 2. Reject concepts one by one → See prediction change
# 3. Visualize → Check if heatmaps make sense
# 4. Identify: Model confused because "count_med" and "size_large" both high
```

---

## 🎨 Visualization Examples

### Concept Activation Maps
Shows WHERE in the image each concept is detected:

```
Original Image    | has_tb_lesion   | has_active_tb   | lesion_size_large
------------------|-----------------|-----------------|------------------
[chest X-ray]     | [red hotspot]   | [red hotspot]  | [red hotspot]
                  | at lesion       | at lesion      | at lesion
```

### Prototype Similarity Maps
Shows which prototype each concept matched:

```
Prototype 0 | Prototype 1 | Prototype 2 | ... | Prototype 9
------------|-------------|-------------|-----|------------
[low sim]   | [high sim]  | [med sim]   | ... | [low sim]
            | ← Matched!  |             |     |
```

---

## 🔧 Troubleshooting

### Error: `num_concepts` mismatch

**Problem:**
```
RuntimeError: size mismatch for concept_head.conv.weight
```

**Solution:**
Check checkpoint and use correct `--num_concepts`:
```bash
import torch
ckpt = torch.load('checkpoints/stage_c_best.pth')
prototypes_shape = ckpt['model_state_dict']['prototypes.prototypes'].shape
print(f"num_concepts: {prototypes_shape[0]}")
print(f"num_prototypes: {prototypes_shape[1]}")
```

---

### Error: Concept names don't match

**Problem:**
```
Warning: Unknown concept 'has_active_tb', skipping
```

**Solution:**
Use exact concept names from training. Check annotations:
```bash
cat TBX11K/csr_annotations/train_annotations.json | jq '.concepts'
```

---

### Memory Error on GPU

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead
python infer.py ... --device cpu

# Or reduce batch size (not applicable for single image)
```

---

## 📚 Python API Usage

You can also use `infer.py` as a module:

```python
from infer import CSRInference

# Initialize
inferencer = CSRInference(
    checkpoint_path='checkpoints/tbx11k/stage_c_best.pth',
    num_concepts=7,
    num_classes=3,
    class_names=['active_tb', 'latent_tb', 'normal'],
    concept_names=['has_lesion', 'active', 'latent', ...]
)

# Load image
image, original = inferencer.load_image('patient001.png')

# Standard prediction
result = inferencer.predict(image)
print(result['predicted_class'])

# With concept rejection
result = inferencer.predict_with_rejection(
    image, 
    rejected_concepts=['active']
)

# Get top concepts
top_concepts = inferencer.get_top_concepts(image, top_k=5)

# Visualize
inferencer.visualize_concepts(image, original, save_path='viz.png')
```

---

## ✅ Best Practices

1. **Always use class and concept names** for readable output
2. **Visualize concepts** for important decisions
3. **Use interactive mode** for model debugging
4. **Save results to JSON** for audit trail
5. **Batch process** for large-scale screening
6. **Test concept rejection** to understand model reasoning

---

## 🎯 Next Steps

After running inference:
1. Review HTML evaluation report for overall model performance
2. Use `find_optimal_threshold.py` if concept recall is low
3. Use `analyze_concept_probs.py` to understand probability distributions
4. Retrain with enhanced losses if needed

---

**Questions? Check the code - `infer.py` has extensive docstrings!**
