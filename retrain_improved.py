"""
Example: Retrain CSR model with improvements based on evaluation report

This script demonstrates how to retrain the model with:
1. Class-balanced loss for imbalanced classes
2. Focal loss for better concept detection
3. Prototype diversity loss
4. Label smoothing for better calibration
5. Weighted sampling for class balance
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        sys.exit(1)
    else:
        print(f"\n✅ Completed: {description}")
    
    return result.returncode


def main():
    # Configuration
    DATASET = "tbx11k"
    DATA_ROOT = "./data/TBX11K"
    TRAIN_FILE = "data/TBX11K/train_labels.csv"
    VAL_FILE = "data/TBX11K/val_labels.csv"
    TEST_FILE = "data/TBX11K/test_labels.csv"
    SAVE_DIR = "checkpoints/tbx11k_improved"
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   CSR Model Improved Training                        ║
║                                                                      ║
║  Based on evaluation report analysis, this training will fix:       ║
║  - ❌ Active TB recall: 17.7% → Target: >75%                        ║
║  - ❌ Latent TB precision: 20.5% → Target: >70%                     ║
║  - ❌ Concept recall: 86.9% → Target: >92%                          ║
║  - ❌ Calibration ECE: 0.159 → Target: <0.10                        ║
║  - ❌ Concepts never predicted → All detected                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Ensure checkpoint directory exists
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    # ==================================================================
    # OPTION 1: Full training with all improvements (RECOMMENDED)
    # ==================================================================
    
    print("\n📋 Training Configuration: ALL IMPROVEMENTS")
    print("  ✓ Focal BCE Loss (Stage A) - α=0.75, γ=2.0")
    print("  ✓ Concept Loss Weight - 2.0x")
    print("  ✓ Class-Balanced Loss (Stage C)")
    print("  ✓ Weighted Random Sampling")
    print("  ✓ Prototype Diversity Loss - weight=0.1")
    print("  ✓ Label Smoothing - 0.1")
    print("  ✓ Early Stopping - patience=15")
    print("  ✓ Extended training - Stage A: 50, B: 30, C: 40 epochs")
    
    train_cmd = [
        "python", "train.py",
        "--dataset", DATASET,
        "--data_root", DATA_ROOT,
        "--train_file", TRAIN_FILE,
        "--val_file", VAL_FILE,
        "--test_file", TEST_FILE,
        "--save_dir", SAVE_DIR,
        
        # Class imbalance solutions
        "--use_class_balanced_loss",
        "--use_weighted_sampling",
        
        # Concept detection improvements
        "--use_focal_loss",
        "--focal_alpha", "0.75",
        "--focal_gamma", "2.0",
        "--concept_loss_weight", "2.0",
        
        # Prototype improvements
        "--use_prototype_diversity",
        "--diversity_weight", "0.1",
        
        # Calibration
        "--label_smoothing", "0.1",
        
        # Training schedule
        "--stage_a_epochs", "50",
        "--stage_b_epochs", "30",
        "--stage_c_epochs", "40",
        
        # Early stopping
        "--early_stopping",
        "--patience", "15",
        "--min_delta", "0.001",
        
        # Model config
        "--backbone", "resnet50",
        "--num_prototypes", "10",
        "--proj_dim", "128",
        "--input_size", "224",
        
        # Hardware
        "--batch_size", "32",
        "--num_workers", "4",
        "--device", "cuda"
    ]
    
    run_command(train_cmd, "Training CSR Model with All Improvements")
    
    # ==================================================================
    # Step 2: Find optimal thresholds
    # ==================================================================
    
    print("\n\n📊 Finding optimal thresholds for concept detection...")
    
    threshold_cmd = [
        "python", "find_optimal_threshold.py",
        "--checkpoint", f"{SAVE_DIR}/stage_c_best.pth",
        "--dataset", DATASET,
        "--data_root", DATA_ROOT,
        "--train_file", TRAIN_FILE,
        "--val_file", VAL_FILE,
        "--test_file", TEST_FILE,
        "--output", f"{SAVE_DIR}/optimal_thresholds.json"
    ]
    
    run_command(threshold_cmd, "Finding Optimal Thresholds")
    
    # ==================================================================
    # Step 3: Evaluate with new checkpoint
    # ==================================================================
    
    print("\n\n📈 Evaluating improved model...")
    
    eval_cmd = [
        "python", "evaluate.py",
        "--checkpoint", f"{SAVE_DIR}/stage_c_best.pth",
        "--dataset", DATASET,
        "--data_root", DATA_ROOT,
        "--train_file", TRAIN_FILE,
        "--val_file", VAL_FILE,
        "--test_file", TEST_FILE,
        "--output_dir", f"reports/{SAVE_DIR.replace('checkpoints/', '')}"
    ]
    
    run_command(eval_cmd, "Evaluating Improved Model")
    
    # ==================================================================
    # Success message
    # ==================================================================
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    ✅ TRAINING COMPLETED!                            ║
╚══════════════════════════════════════════════════════════════════════╝

📁 Checkpoints saved in: {SAVE_DIR}/
   - stage_a_best.pth
   - stage_b_best.pth  
   - stage_c_best.pth (← use this for inference)

📊 Evaluation report: reports/{SAVE_DIR.replace('checkpoints/', '')}/evaluation_report.html
   Open this file in your browser to see detailed metrics

🎯 Optimal thresholds: {SAVE_DIR}/optimal_thresholds.json
   Use these thresholds in your inference pipeline

Next steps:
1. Open evaluation_report.html to compare with previous results
2. Check if Active TB recall improved (target: >75%)
3. Check if Latent TB precision improved (target: >70%)
4. Check if all concepts are now detected
5. Verify calibration improved (ECE < 0.10)

If results are still not satisfactory:
- Increase Stage A epochs further (--stage_a_epochs 80)
- Use higher concept loss weight (--concept_loss_weight 3.0)
- Try more aggressive focal loss (--focal_gamma 3.0)
- Check if data has labeling errors

Happy training! 🚀
    """)


if __name__ == "__main__":
    main()
