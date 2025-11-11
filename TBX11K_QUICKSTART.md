# Quick Start: TBX11K Dataset with CSR

This guide helps you quickly get started with TBX11K dataset and CSR model.

## Your Dataset Structure

```
TBX11K/
├── imgs/
│   ├── health/      # Healthy X-rays
│   ├── sick/        # Sick but non-TB X-rays
│   ├── tb/          # TB X-rays
│   ├── test/        # Test X-rays
│   └── extra/       # Additional datasets (DA, DB, Montgomery, Shenzhen)
├── annotations/
│   ├── xml/         # Bounding box annotations (XML format)
│   └── json/        # COCO-style annotations
├── lists/
│   ├── TBX11K_train.txt      # 6600 training images
│   ├── TBX11K_val.txt        # 1800 validation images
│   ├── TBX11K_trainval.txt   # 8400 train+val images
│   ├── all_train.txt         # TBX11K + extra datasets (training)
│   ├── all_val.txt           # TBX11K + extra datasets (validation)
│   ├── all_trainval.txt      # All train+val
│   └── all_test.txt          # 2800 test images + extra datasets
└── code/
    └── make_json_anno.py     # Original COCO annotation converter
```

## Step-by-Step Guide

### Step 1: Convert TBX11K to CSR Format

#### Option A: Use the helper script (Recommended)

```powershell
python run_tbx11k_conversion.py
```

This will:
- Check your dataset structure
- Ask which splits to use (TBX11K only or with extra datasets)
- Convert annotations to CSR format
- Show you the training command

#### Option B: Manual conversion

```powershell
# Using TBX11K splits only
python convert_tbx11k_to_csr.py `
    --data_root TBX11K `
    --output_dir TBX11K/csr_annotations `
    --format both

# OR using all datasets (includes DA, DB, Montgomery, Shenzhen)
python convert_tbx11k_to_csr.py `
    --data_root TBX11K `
    --output_dir TBX11K/csr_annotations `
    --format both `
    --use_all_splits
```

**Output:**
```
TBX11K/csr_annotations/
├── train_annotations.json   # CSR-compatible format
├── train_annotations.csv
├── val_annotations.json
├── val_annotations.csv
├── test_annotations.json
└── test_annotations.csv
```

### Step 2: Train CSR Model

```powershell
python train.py `
    --dataset tbx11k `
    --data_root TBX11K/imgs `
    --train_file TBX11K/csr_annotations/train_annotations.json `
    --val_file TBX11K/csr_annotations/val_annotations.json `
    --test_file TBX11K/csr_annotations/test_annotations.json `
    --backbone resnet50 `
    --num_prototypes 10 `
    --batch_size 32 `
    --stage_a_epochs 30 `
    --stage_b_epochs 20 `
    --stage_c_epochs 30 `
    --save_dir checkpoints/tbx11k `
    --device cuda
```

**Training will run in 3 stages automatically:**
1. **Stage A** (30 epochs): Learn concepts from X-rays
2. **Stage B** (20 epochs): Learn prototypes for each concept
3. **Stage C** (30 epochs): Train disease classifier

**Note:** Full training may take several hours depending on your GPU.

### Step 3: Evaluate Model

```powershell
python evaluate.py `
    --dataset tbx11k `
    --data_root TBX11K/imgs `
    --train_file TBX11K/csr_annotations/train_annotations.json `
    --val_file TBX11K/csr_annotations/val_annotations.json `
    --test_file TBX11K/csr_annotations/test_annotations.json `
    --checkpoint checkpoints/tbx11k/stage_c_best.pth `
    --backbone resnet50 `
    --output_dir results/tbx11k
```

Results will be saved in `results/tbx11k/evaluation_report.json`

## What the Converter Does

Since TBX11K is a **detection dataset** (bounding boxes), the converter extracts **interpretable concepts**:

### 10 Concepts Extracted:

1. **Lesion Presence:**
   - `has_tb_lesion`: Any TB bounding box present
   - `has_active_tb`: Active TB lesion
   - `has_latent_tb`: Latent TB lesion
   - `has_uncertain_tb`: Uncertain TB lesion

2. **Lesion Count:**
   - `lesion_count_low`: 1-2 lesions
   - `lesion_count_medium`: 3-5 lesions
   - `lesion_count_high`: >5 lesions

3. **Lesion Size:**
   - `lesion_size_small`: Small lesions (< 5000 px²)
   - `lesion_size_medium`: Medium lesions (5000-15000 px²)
   - `lesion_size_large`: Large lesions (> 15000 px²)

### 4 Disease Classes:

- `normal`: No TB lesions
- `active_tuberculosis`: Has active TB
- `latent_tuberculosis`: Has latent TB
- `uncertain_tuberculosis`: Has uncertain TB

## Expected Performance

Training on TBX11K with default settings:
- **Dataset**: 6600 train, 1800 val images
- **Time**: ~3-4 hours on RTX 3090
- **Memory**: ~8GB GPU memory with batch_size=32

## Tips for Better Results

1. **Use all datasets** for more training data:
   ```powershell
   python convert_tbx11k_to_csr.py --use_all_splits
   ```

2. **Adjust batch size** based on your GPU:
   - RTX 3090/4090: `--batch_size 64`
   - RTX 3080: `--batch_size 32`
   - RTX 3070: `--batch_size 16`

3. **Try ConvNeXtV2** for better accuracy:
   ```powershell
   pip install timm
   python train.py --backbone convnextv2_base [other args...]
   ```

4. **Use class weights** for imbalanced data:
   - Most TBX11K images have TB lesions
   - Normal cases are fewer

## Troubleshooting

### Issue: "File not found" errors
**Solution:** Check your `--data_root` path points to TBX11K folder

### Issue: CUDA out of memory
**Solution:** Reduce `--batch_size` (try 16 or 8)

### Issue: Poor accuracy
**Solutions:**
- Train longer: increase epoch counts
- Use all datasets: `--use_all_splits`
- Try ConvNeXtV2 backbone

### Issue: Slow training
**Solution:** Use lighter backbone: `--backbone resnet34`

## Next Steps

After training, you can:

1. **Visualize predictions:**
   ```python
   python demo.py  # Shows interactive examples
   ```

2. **Use doctor-in-the-loop:**
   - Reject incorrect concepts
   - Mark important regions
   - Refine predictions

3. **Deploy the model:**
   - Load checkpoint
   - Run inference on new X-rays
   - Get interpretable diagnoses

## Questions?

- Check `TBX11K_PREPARATION.md` for detailed explanations
- See `README.md` for complete CSR documentation
- Run `python test_installation.py` to verify setup
