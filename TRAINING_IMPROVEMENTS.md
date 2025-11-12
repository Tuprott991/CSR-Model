# 🚀 Training Improvements Guide

## Tổng quan các cải tiến đã thêm

Dựa trên phân tích evaluation report, các cải tiến sau đã được tích hợp vào code:

### 1. **Class Imbalance Solutions** ⚖️

#### A. Weighted Random Sampling
```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --use_weighted_sampling
```

**Khi nào dùng:** Dataset có class imbalance nghiêm trọng (Normal >> Active TB >> Latent TB)

#### B. Class Weights
```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --use_class_weights
```

**Khi nào dùng:** Khi muốn model chú ý hơn vào minority classes

#### C. Class-Balanced Loss (Recommended) ⭐
```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --use_class_balanced_loss
```

**Khi nào dùng:** Best practice cho class imbalance, sử dụng effective number of samples

---

### 2. **Concept Detection Improvements** 🧩

#### A. Focal BCE Loss (Stage A)
```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --use_focal_loss \
  --focal_alpha 0.75 \
  --focal_gamma 2.0
```

**Khi nào dùng:** 
- Concept recall thấp (< 70%)
- Nhiều concepts bị bỏ sót (has_latent_tb recall = 11.6%)

**Parameters:**
- `--focal_alpha`: 0.75 (weight cho positive samples)
- `--focal_gamma`: 2.0 (focus vào hard examples)

#### B. Increase Concept Loss Weight
```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --concept_loss_weight 2.0
```

**Khi nào dùng:** Concept performance kém hơn task performance

---

### 3. **Prototype Diversity** 🎯

```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --use_prototype_diversity \
  --diversity_weight 0.1
```

**Khi nào dùng:**
- Nhiều prototypes không được sử dụng (usage < 1%)
- Prototype collapse (1-2 prototypes chiếm > 90% usage)

**Parameters:**
- `--diversity_weight`: 0.05-0.2 (bắt đầu với 0.1)

---

### 4. **Label Smoothing** 🎨

```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --label_smoothing 0.1
```

**Khi nào dùng:**
- Calibration kém (ECE > 0.15)
- Model quá tự tin với predictions sai

**Parameters:**
- `--label_smoothing`: 0.1 (recommended), 0.05-0.2

---

## 🎯 Recommended Training Configurations

### Configuration 1: Fix Class Imbalance (Priority 1) 🔥

```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --use_class_balanced_loss \
  --use_weighted_sampling \
  --label_smoothing 0.1 \
  --stage_a_epochs 30 \
  --stage_b_epochs 20 \
  --stage_c_epochs 30 \
  --early_stopping \
  --patience 10
```

**Expected improvements:**
- Active TB recall: 17.7% → >70%
- Latent TB precision: 20.5% → >60%
- Overall accuracy maintained or improved

---

### Configuration 2: Fix Concept Detection (Priority 2) 🧩

```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --use_focal_loss \
  --focal_alpha 0.75 \
  --focal_gamma 2.0 \
  --concept_loss_weight 2.0 \
  --stage_a_epochs 50 \
  --stage_b_epochs 20 \
  --stage_c_epochs 30
```

**Expected improvements:**
- Concept recall: 86.9% → >90%
- lesion_count_medium, lesion_size_medium: 0% → detected
- has_latent_tb recall: 11.6% → >50%

---

### Configuration 3: All Improvements (Recommended) ⭐

```bash
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --use_focal_loss \
  --focal_alpha 0.75 \
  --focal_gamma 2.0 \
  --concept_loss_weight 2.0 \
  --use_class_balanced_loss \
  --use_weighted_sampling \
  --label_smoothing 0.1 \
  --use_prototype_diversity \
  --diversity_weight 0.1 \
  --stage_a_epochs 50 \
  --stage_b_epochs 30 \
  --stage_c_epochs 40 \
  --early_stopping \
  --patience 15 \
  --batch_size 32
```

**Expected improvements:**
- Active TB recall: 17.7% → >75%
- Latent TB precision: 20.5% → >70%
- Concept recall: 86.9% → >92%
- Calibration ECE: 0.159 → <0.10
- All concepts detected (no 0% recall)

---

## 📊 Monitoring Training

### Key Metrics to Watch

**Stage A (Concept Training):**
```
✅ Concept BCE Loss decreasing
✅ Per-concept recall > 60% (especially has_latent_tb)
⚠️ Watch for concepts with 0% prediction
```

**Stage B (Prototype Training):**
```
✅ Contrastive loss decreasing
✅ Prototype usage balanced (no prototype > 80%)
✅ Diversity loss stable (if enabled)
```

**Stage C (Task Training):**
```
✅ Task accuracy > 90%
✅ Per-class recall > 50% (especially Active TB, Latent TB)
✅ Validation loss not increasing (no overfitting)
```

---

## 🔄 Resume Training

If training interrupted, resume from checkpoint:

```bash
# Resume from specific checkpoint
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --resume checkpoints/stage_a_best.pth \
  --start_stage b

# Or auto-resume from best checkpoint
python train.py \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --start_stage b  # Will auto-load stage_a_best.pth
```

---

## 🧪 After Training

### 1. Find Optimal Thresholds
```bash
python find_optimal_threshold.py \
  --checkpoint checkpoints/stage_c_best.pth \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv
```

### 2. Re-evaluate
```bash
python evaluate.py \
  --checkpoint checkpoints/stage_c_best.pth \
  --dataset tbx11k \
  --data_root ./data/TBX11K \
  --train_file data/TBX11K/train_labels.csv \
  --val_file data/TBX11K/val_labels.csv \
  --test_file data/TBX11K/test_labels.csv \
  --output_dir reports
```

### 3. Compare Results
Check `reports/evaluation_report.html` for:
- ✅ Active TB recall improved
- ✅ Latent TB precision improved
- ✅ All concepts detected
- ✅ Better calibration (ECE < 0.10)

---

## 📝 Troubleshooting

### Problem: Out of Memory
```bash
# Reduce batch size
--batch_size 16

# Or use gradient accumulation (requires code modification)
```

### Problem: Training too slow
```bash
# Reduce workers
--num_workers 2

# Use smaller backbone
--backbone resnet34
```

### Problem: Still poor concept recall
```bash
# Increase Stage A epochs significantly
--stage_a_epochs 80

# Higher concept loss weight
--concept_loss_weight 3.0

# More aggressive focal loss
--focal_gamma 3.0
```

### Problem: Overfitting
```bash
# Enable early stopping
--early_stopping --patience 10

# Add label smoothing
--label_smoothing 0.15

# More aggressive data augmentation (modify dataloader)
```

---

## 🎓 Best Practices

1. **Always start with Configuration 3** (all improvements enabled)
2. **Monitor training logs** for concept-level metrics
3. **Run threshold optimization** after training
4. **Compare before/after reports** to verify improvements
5. **Use early stopping** to prevent overfitting
6. **Save checkpoints frequently** for each stage
7. **Test on held-out test set** before deployment

---

## 📞 Quick Reference

| Issue | Solution | Flag |
|-------|----------|------|
| Low Active TB recall | Class-balanced loss | `--use_class_balanced_loss` |
| Low concept recall | Focal loss + weight | `--use_focal_loss --concept_loss_weight 2.0` |
| Concepts never predicted | Focal loss Stage A | `--use_focal_loss --stage_a_epochs 50` |
| Prototype collapse | Diversity loss | `--use_prototype_diversity` |
| Poor calibration | Label smoothing | `--label_smoothing 0.1` |
| Class imbalance | Weighted sampling | `--use_weighted_sampling` |

---

**Happy Training! 🚀**
