# Resume Training Guide

This guide explains how to resume CSR training from any stage.

## Quick Reference

### Resume from where you stopped

```bash
# If you stopped after Stage A
python train.py --start_stage b --dataset tbx11k --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json

# If you stopped after Stage B  
python train.py --start_stage c --dataset tbx11k --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json
```

### Explicitly specify checkpoint to resume from

```bash
# Resume from specific checkpoint
python train.py --resume checkpoints/stage_a_best.pth --start_stage b \
    --dataset tbx11k --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json
```

## Training Stages Explained

CSR training consists of 3 sequential stages:

1. **Stage A (Concept Model)**: Trains feature extractor + concept head with BCE loss
   - Output: `stage_a_best.pth`, `stage_a_final.pth`

2. **Stage B (Prototype Learning)**: Trains projector + prototypes with contrastive loss
   - Requires: Stage A checkpoint
   - Output: `stage_b_best.pth`, `stage_b_final.pth`

3. **Stage C (Task Head)**: Trains task classifier for disease prediction
   - Requires: Stage B checkpoint
   - Output: `stage_c_best.pth`, `stage_c_final.pth`

## Detailed Examples

### Example 1: Train Stage A only, then continue later

**Session 1: Train Stage A**
```bash
python train.py \
    --dataset tbx11k \
    --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json \
    --stage_a_epochs 30 \
    --save_dir checkpoints/tbx11k

# Training will complete Stage A and save:
#   - checkpoints/tbx11k/stage_a_best.pth
#   - checkpoints/tbx11k/stage_a_final.pth
# Then press Ctrl+C to stop before Stage B
```

**Session 2: Continue with Stage B → C**
```bash
python train.py \
    --dataset tbx11k \
    --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json \
    --start_stage b \
    --save_dir checkpoints/tbx11k
    
# Will auto-load checkpoints/tbx11k/stage_a_best.pth
# Then train Stage B → Stage C
```

### Example 2: Training interrupted during Stage B

```bash
# Original command that was interrupted
python train.py \
    --dataset tbx11k \
    --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json \
    --save_dir checkpoints/tbx11k

# Training stopped during Stage B
# Available checkpoints:
#   ✓ checkpoints/tbx11k/stage_a_best.pth
#   ✓ checkpoints/tbx11k/stage_a_final.pth
#   ✗ checkpoints/tbx11k/stage_b_best.pth (incomplete/missing)

# Restart from Stage B using Stage A checkpoint
python train.py \
    --dataset tbx11k \
    --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json \
    --resume checkpoints/tbx11k/stage_a_final.pth \
    --start_stage b \
    --save_dir checkpoints/tbx11k
```

### Example 3: Experiment with different Stage C settings

```bash
# First, train Stage A and B once
python train.py \
    --dataset tbx11k \
    --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json \
    --stage_a_epochs 30 \
    --stage_b_epochs 20 \
    --save_dir checkpoints/tbx11k
# Stop after Stage B completes

# Experiment 1: Train Stage C with early stopping
python train.py \
    --dataset tbx11k \
    --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json \
    --start_stage c \
    --stage_c_epochs 50 \
    --early_stopping \
    --patience 10 \
    --save_dir checkpoints/exp1

# Experiment 2: Train Stage C with different epochs
python train.py \
    --dataset tbx11k \
    --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json \
    --resume checkpoints/tbx11k/stage_b_best.pth \
    --start_stage c \
    --stage_c_epochs 100 \
    --save_dir checkpoints/exp2
```

### Example 4: Using Early Stopping

```bash
# Enable early stopping for all stages
python train.py \
    --dataset tbx11k \
    --data_root TBX11K/imgs \
    --train_file TBX11K/csr_annotations/train_annotations.json \
    --val_file TBX11K/csr_annotations/val_annotations.json \
    --test_file TBX11K/csr_annotations/test_annotations.json \
    --early_stopping \
    --patience 10 \
    --min_delta 0.001 \
    --save_dir checkpoints/tbx11k

# Training will stop early if validation metric doesn't improve for 10 epochs
# Stage A & B: monitors validation loss
# Stage C: monitors validation accuracy
```

## Available Checkpoints

After each stage, the following checkpoints are saved:

| Checkpoint | Description | When to Use |
|------------|-------------|-------------|
| `stage_a_best.pth` | Best validation loss during Stage A | ✅ **Recommended** for resuming Stage B |
| `stage_a_final.pth` | Final model after Stage A | Use if you want the latest weights |
| `stage_b_best.pth` | Best validation loss during Stage B | ✅ **Recommended** for resuming Stage C |
| `stage_b_final.pth` | Final model after Stage B | Use if you want the latest weights |
| `stage_c_best.pth` | Best validation accuracy during Stage C | ✅ **Use this** for inference/evaluation |
| `stage_c_final.pth` | Final model after Stage C | Alternative final model |

## Command-Line Options

### Resume Options
- `--start_stage {a,b,c}`: Which stage to start from (default: `a`)
- `--resume PATH`: Explicit checkpoint path to load (optional)

### Early Stopping Options
- `--early_stopping`: Enable early stopping (disabled by default)
- `--patience N`: Number of epochs to wait before stopping (default: 10)
- `--min_delta FLOAT`: Minimum improvement threshold (default: 0.001)

### Stage-Specific Epochs
- `--stage_a_epochs N`: Epochs for Stage A (default: 30)
- `--stage_b_epochs N`: Epochs for Stage B (default: 20)
- `--stage_c_epochs N`: Epochs for Stage C (default: 30)

## Auto-Resume Feature

If you don't specify `--resume`, the script will automatically look for checkpoints:

- **Starting Stage B**: Looks for `stage_a_best.pth` in `--save_dir`
- **Starting Stage C**: Looks for `stage_b_best.pth` in `--save_dir`

If the checkpoint is not found, you'll get a helpful error message.

## Troubleshooting

### Error: "Checkpoint not found"
**Solution**: Make sure the checkpoint path is correct
```bash
# Check what checkpoints exist
ls checkpoints/tbx11k/*.pth

# Use the correct path
python train.py --resume checkpoints/tbx11k/stage_a_best.pth --start_stage b ...
```

### Error: "Starting from stage B/C requires checkpoint"
**Solution**: Either train previous stages first OR provide checkpoint
```bash
# Option 1: Train from Stage A
python train.py --start_stage a ...

# Option 2: Provide explicit checkpoint
python train.py --resume checkpoints/stage_a_best.pth --start_stage b ...
```

### Want to start completely fresh?
```bash
# Delete old checkpoints
rm checkpoints/tbx11k/*.pth

# Train from scratch
python train.py --start_stage a --dataset tbx11k ...
```

### Check checkpoint details
```bash
# Use Python to inspect a checkpoint
python -c "import torch; ckpt = torch.load('checkpoints/stage_a_best.pth'); print(ckpt.keys())"
```

## Best Practices

1. **Always use `_best.pth` checkpoints** for resuming - they have the best validation performance
2. **Keep checkpoints organized** - use different `--save_dir` for different experiments
3. **Enable early stopping** for faster experimentation - saves time if model stops improving
4. **Monitor training** - check that validation metrics are improving
5. **Backup important checkpoints** before re-training

## Quick Cheat Sheet

```bash
# Full training from scratch
python train.py --start_stage a --dataset tbx11k --data_root TBX11K/imgs \
    --train_file train.json --val_file val.json --test_file test.json

# Resume after Stage A
python train.py --start_stage b --dataset tbx11k --data_root TBX11K/imgs \
    --train_file train.json --val_file val.json --test_file test.json

# Resume after Stage B
python train.py --start_stage c --dataset tbx11k --data_root TBX11K/imgs \
    --train_file train.json --val_file val.json --test_file test.json

# With early stopping
python train.py --start_stage a --early_stopping --patience 10 \
    --dataset tbx11k --data_root TBX11K/imgs \
    --train_file train.json --val_file val.json --test_file test.json

# Explicit resume
python train.py --resume checkpoints/stage_b_best.pth --start_stage c \
    --dataset tbx11k --data_root TBX11K/imgs \
    --train_file train.json --val_file val.json --test_file test.json
```

## Common Workflows

### Workflow 1: Incremental Training
```bash
# Day 1: Train Stage A
python train.py --start_stage a --stage_a_epochs 30 ...

# Day 2: Train Stage B
python train.py --start_stage b --stage_b_epochs 20 ...

# Day 3: Train Stage C
python train.py --start_stage c --stage_c_epochs 30 ...
```

### Workflow 2: Hyperparameter Search for Stage C
```bash
# Train A and B once
python train.py --start_stage a --save_dir checkpoints/base ...

# Try different Stage C configurations
for lr in 1e-3 1e-4 1e-5; do
    python train.py --resume checkpoints/base/stage_b_best.pth \
        --start_stage c --save_dir checkpoints/lr_${lr} ...
done
```

### Workflow 3: Quick Iteration with Early Stopping
```bash
# Enable early stopping for fast experimentation
python train.py --start_stage a --early_stopping --patience 5 \
    --stage_a_epochs 100 --stage_b_epochs 100 --stage_c_epochs 100 ...
# Will stop early if not improving - no need to wait for all 100 epochs!
```
