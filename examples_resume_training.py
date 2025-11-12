"""
Example: Resume Training Demo

This script demonstrates different ways to resume CSR training.
"""

# Example 1: Train all stages from scratch
print("="*60)
print("Example 1: Train all stages from scratch")
print("="*60)
example1 = """
python train.py \\
    --dataset tbx11k \\
    --data_root TBX11K/imgs \\
    --train_file TBX11K/csr_annotations/train_annotations.json \\
    --val_file TBX11K/csr_annotations/val_annotations.json \\
    --test_file TBX11K/csr_annotations/test_annotations.json \\
    --backbone resnet50 \\
    --stage_a_epochs 30 \\
    --stage_b_epochs 20 \\
    --stage_c_epochs 30 \\
    --save_dir checkpoints/tbx11k

# This will run: Stage A → Stage B → Stage C
# Checkpoints saved:
#   - checkpoints/tbx11k/stage_a_best.pth
#   - checkpoints/tbx11k/stage_a_final.pth
#   - checkpoints/tbx11k/stage_b_best.pth
#   - checkpoints/tbx11k/stage_b_final.pth
#   - checkpoints/tbx11k/stage_c_best.pth
#   - checkpoints/tbx11k/stage_c_final.pth
"""
print(example1)

# Example 2: Stop after Stage A, resume later
print("\n" + "="*60)
print("Example 2: Resume from Stage A checkpoint")
print("="*60)
example2_session1 = """
# Session 1: Train only Stage A
python train.py \\
    --dataset tbx11k \\
    --data_root TBX11K/imgs \\
    --train_file TBX11K/csr_annotations/train_annotations.json \\
    --val_file TBX11K/csr_annotations/val_annotations.json \\
    --test_file TBX11K/csr_annotations/test_annotations.json \\
    --stage_a_epochs 30 \\
    --save_dir checkpoints/tbx11k

# Press Ctrl+C after Stage A completes (before Stage B starts)
"""
print("Session 1:")
print(example2_session1)

example2_session2 = """
# Session 2: Resume from Stage B (auto-loads stage_a_best.pth)
python train.py \\
    --dataset tbx11k \\
    --data_root TBX11K/imgs \\
    --train_file TBX11K/csr_annotations/train_annotations.json \\
    --val_file TBX11K/csr_annotations/val_annotations.json \\
    --test_file TBX11K/csr_annotations/test_annotations.json \\
    --start_stage b \\
    --save_dir checkpoints/tbx11k

# Will automatically load: checkpoints/tbx11k/stage_a_best.pth
# Then run: Stage B → Stage C
"""
print("Session 2:")
print(example2_session2)

# Example 3: Explicit checkpoint resume
print("\n" + "="*60)
print("Example 3: Explicit checkpoint resume")
print("="*60)
example3 = """
# Resume from specific checkpoint (not necessarily from save_dir)
python train.py \\
    --dataset tbx11k \\
    --data_root TBX11K/imgs \\
    --train_file TBX11K/csr_annotations/train_annotations.json \\
    --val_file TBX11K/csr_annotations/val_annotations.json \\
    --test_file TBX11K/csr_annotations/test_annotations.json \\
    --resume checkpoints/experiment1/stage_b_best.pth \\
    --start_stage c \\
    --save_dir checkpoints/experiment2

# Will load the specified checkpoint
# Then run: Stage C only
# New checkpoints saved to checkpoints/experiment2/
"""
print(example3)

# Example 4: With early stopping
print("\n" + "="*60)
print("Example 4: Resume with early stopping enabled")
print("="*60)
example4 = """
python train.py \\
    --dataset tbx11k \\
    --data_root TBX11K/imgs \\
    --train_file TBX11K/csr_annotations/train_annotations.json \\
    --val_file TBX11K/csr_annotations/val_annotations.json \\
    --test_file TBX11K/csr_annotations/test_annotations.json \\
    --start_stage b \\
    --early_stopping \\
    --patience 10 \\
    --min_delta 0.001 \\
    --save_dir checkpoints/tbx11k

# Will auto-load stage_a_best.pth
# Early stopping monitors:
#   - Stage B: validation loss (stops if no improvement for 10 epochs)
#   - Stage C: validation accuracy (stops if no improvement for 10 epochs)
"""
print(example4)

# Example 5: Different backbones
print("\n" + "="*60)
print("Example 5: Resume with different backbone")
print("="*60)
example5 = """
# Train with ResNet50
python train.py \\
    --dataset tbx11k \\
    --data_root TBX11K/imgs \\
    --train_file TBX11K/csr_annotations/train_annotations.json \\
    --val_file TBX11K/csr_annotations/val_annotations.json \\
    --test_file TBX11K/csr_annotations/test_annotations.json \\
    --backbone resnet50 \\
    --save_dir checkpoints/resnet50

# Then resume and train with ConvNeXtV2
python train.py \\
    --dataset tbx11k \\
    --data_root TBX11K/imgs \\
    --train_file TBX11K/csr_annotations/train_annotations.json \\
    --val_file TBX11K/csr_annotations/val_annotations.json \\
    --test_file TBX11K/csr_annotations/test_annotations.json \\
    --backbone convnextv2_base \\
    --start_stage b \\
    --save_dir checkpoints/convnextv2

# Note: You can't directly resume from a different backbone's checkpoint
# The architecture must match. This example shows training different backbones
# from scratch with the same dataset.
"""
print(example5)

print("\n" + "="*60)
print("Summary of Resume Options")
print("="*60)
summary = """
1. Auto-resume (no --resume flag):
   --start_stage b  → auto-loads stage_a_best.pth
   --start_stage c  → auto-loads stage_b_best.pth

2. Explicit resume (with --resume flag):
   --resume PATH --start_stage b  → loads specified checkpoint, runs B→C
   --resume PATH --start_stage c  → loads specified checkpoint, runs C

3. With early stopping:
   Add --early_stopping --patience 10 --min_delta 0.001

4. Checkpoint naming convention:
   stage_a_best.pth   - Best validation loss (Stage A)
   stage_a_final.pth  - Final epoch (Stage A)
   stage_b_best.pth   - Best validation loss (Stage B)
   stage_b_final.pth  - Final epoch (Stage B)
   stage_c_best.pth   - Best validation accuracy (Stage C)
   stage_c_final.pth  - Final epoch (Stage C)

Recommendation: Use *_best.pth checkpoints for resuming!
"""
print(summary)

print("\n" + "="*60)
print("For more details, see RESUME_TRAINING.md")
print("="*60)
