# TBX11K Dataset Preparation for CSR

This guide explains how to convert TBX11K detection annotations to CSR-compatible format.

## TBX11K Dataset Structure

```
TBX11K/
├── imgs/                    # Images (512x512)
├── annotations/
│   ├── xml/                 # Bounding box annotations
│   └── json/                # COCO-style annotations
├── lists/
│   ├── TBX11K_train.txt    # 6600 images
│   ├── TBX11K_val.txt      # 1800 images
│   ├── TBX11K_trainval.txt
│   ├── all_train.txt
│   ├── all_val.txt
│   └── all_test.txt        # 2800 images + extra datasets
└── imgs/extra/             # Additional TB datasets (DA, DB, Montgomery, Shenzhen)
```

## Problem: Detection vs Classification

**TBX11K provides:**
- Bounding box annotations for TB lesions
- Categories: Active TB, Latent TB, Uncertain TB
- Format: Object detection (XML/JSON)

**CSR requires:**
- Multi-label concept annotations (findings)
- Single-label disease classification
- Format: JSON with concept labels + disease labels

## Solution: Concept Extraction

The converter (`convert_tbx11k_to_csr.py`) extracts interpretable concepts from detection annotations:

### Derived Concepts (10 concepts)

1. **Lesion Presence**:
   - `has_tb_lesion`: Any TB bounding box present
   - `has_active_tb`: Active TB lesion present
   - `has_latent_tb`: Latent TB lesion present
   - `has_uncertain_tb`: Uncertain TB lesion present

2. **Lesion Count** (based on number of bboxes):
   - `lesion_count_low`: 1-2 lesions
   - `lesion_count_medium`: 3-5 lesions
   - `lesion_count_high`: >5 lesions

3. **Lesion Size** (based on max bbox area):
   - `lesion_size_small`: max area < 5000 pixels
   - `lesion_size_medium`: max area 5000-15000 pixels
   - `lesion_size_large`: max area > 15000 pixels

### Disease Classes (4 classes)

- `normal`: No TB lesions
- `active_tuberculosis`: Has active TB lesions
- `latent_tuberculosis`: Has latent TB lesions
- `uncertain_tuberculosis`: Has uncertain TB lesions

## Usage

### Step 1: Convert Annotations

```bash
# Convert all splits (train/val/test)
python convert_tbx11k_to_csr.py \
    --data_root /path/to/TBX11K \
    --output_dir csr_annotations \
    --format both
```

This creates:
```
csr_annotations/
├── train_annotations.json  # CSR-compatible JSON
├── train_annotations.csv   # CSR-compatible CSV
├── val_annotations.json
├── val_annotations.csv
├── test_annotations.json
└── test_annotations.csv
```

### Step 2: Train CSR Model

```bash
python train.py \
    --dataset tbx11k \
    --data_root /path/to/TBX11K/imgs \
    --train_file csr_annotations/train_annotations.json \
    --val_file csr_annotations/val_annotations.json \
    --test_file csr_annotations/test_annotations.json \
    --backbone resnet50 \
    --num_prototypes 10 \
    --proj_dim 128 \
    --batch_size 32 \
    --stage_a_epochs 30 \
    --stage_b_epochs 20 \
    --stage_c_epochs 30 \
    --save_dir checkpoints/tbx11k
```

## Output Format Examples

### JSON Format

```json
{
  "image_001.png": {
    "findings": [
      "has_tb_lesion",
      "has_active_tb",
      "lesion_count_low",
      "lesion_size_medium"
    ],
    "disease": "active_tuberculosis"
  },
  "image_002.png": {
    "findings": [],
    "disease": "normal"
  }
}
```

### CSV Format

```csv
image_path,has_tb_lesion,has_active_tb,has_latent_tb,...,disease
image_001.png,1,1,0,0,1,0,0,0,1,0,active_tuberculosis
image_002.png,0,0,0,0,0,0,0,0,0,0,normal
```

## Training Pipeline

With the converted annotations, CSR will train in 3 stages:

### Stage A: Concept Model
- Predicts 10 binary concepts from X-ray images
- Learns which visual features correspond to each concept

### Stage B: Prototype Learning
- Learns 10 prototypes per concept (100 total prototypes)
- Each prototype represents a typical instance of a concept

### Stage C: Task Classification
- Classifies disease (normal vs TB types) using prototype similarities
- Interpretable: can show which concepts/prototypes contribute to diagnosis

## Interpretability Benefits

After training, you can:

1. **See which concepts are detected** in each X-ray
2. **Visualize concept activation maps** (where each concept appears)
3. **Understand disease predictions** based on concept combinations
4. **Enable doctor feedback**:
   - Reject incorrect concepts
   - Mark important/unimportant regions

## Alternative: Manual Concept Annotation

For better interpretability, consider manually annotating additional clinical concepts:

- Radiological findings: infiltration, consolidation, cavity, nodule, etc.
- Location: upper lobe, middle lobe, lower lobe
- Bilateral vs unilateral
- With complications: pleural effusion, lymphadenopathy

This would require expert radiologists to annotate but provides richer clinical interpretability.

## Notes

1. **Test Set**: TBX11K's test set includes additional datasets (DA, DB, Montgomery, Shenzhen). The converter handles this automatically.

2. **Class Imbalance**: Normal cases might be underrepresented. Consider using class weights in training.

3. **Concept Quality**: The auto-derived concepts are heuristic. For clinical deployment, expert-annotated concepts are recommended.

4. **Performance**: The 3-stage training might take several hours depending on GPU. Use smaller backbone (ResNet-34) for faster training.

## Troubleshooting

**Issue**: Missing XML files
- Some images might not have annotations (normal cases)
- Converter handles this by assigning 'normal' disease label

**Issue**: Memory errors
- Reduce batch size: `--batch_size 16`
- Use smaller backbone: `--backbone resnet34`

**Issue**: Poor concept accuracy
- Increase Stage A epochs: `--stage_a_epochs 50`
- Use concept weights for class imbalance
