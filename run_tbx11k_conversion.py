"""
Example script to convert TBX11K dataset to CSR format

Before running:
1. Make sure TBX11K dataset is downloaded and extracted
2. Check the directory structure matches:
   TBX11K/
   ├── imgs/
   │   ├── health/
   │   ├── sick/
   │   ├── tb/
   │   ├── test/
   │   └── extra/
   ├── annotations/xml/
   └── lists/
       ├── TBX11K_train.txt
       ├── TBX11K_val.txt
       └── all_test.txt
"""

import subprocess
import sys
from pathlib import Path


def check_dataset_structure(data_root: str):
    """Check if TBX11K dataset structure is correct"""
    data_root = Path(data_root)
    
    required_dirs = [
        data_root / 'imgs',
        data_root / 'annotations' / 'xml',
        data_root / 'lists'
    ]
    
    required_files = [
        data_root / 'lists' / 'TBX11K_train.txt',
        data_root / 'lists' / 'TBX11K_val.txt',
    ]
    
    print("Checking TBX11K dataset structure...")
    all_good = True
    
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  ✓ Found: {dir_path}")
        else:
            print(f"  ✗ Missing: {dir_path}")
            all_good = False
    
    for file_path in required_files:
        if file_path.exists():
            print(f"  ✓ Found: {file_path}")
        else:
            print(f"  ✗ Missing: {file_path}")
            all_good = False
    
    return all_good


def main():
    # Change this to your TBX11K dataset path
    DATA_ROOT = "TBX11K"  # or "D:/datasets/TBX11K" or "/path/to/TBX11K"
    
    print("="*60)
    print("TBX11K to CSR Format Converter")
    print("="*60)
    
    # Check dataset structure
    if not check_dataset_structure(DATA_ROOT):
        print("\n❌ Dataset structure is incomplete!")
        print("Please check the paths above and fix the structure.")
        sys.exit(1)
    
    print("\n✓ Dataset structure looks good!")
    
    # Ask user which splits to use
    print("\nWhich splits do you want to use?")
    print("  1. TBX11K only (6600 train, 1800 val)")
    print("  2. All datasets (includes DA, DB, Montgomery, Shenzhen)")
    
    choice = input("Enter choice (1 or 2) [default: 1]: ").strip() or "1"
    
    use_all = choice == "2"
    
    # Run conversion
    print(f"\n{'='*60}")
    print("Running conversion...")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        "convert_tbx11k_to_csr.py",
        "--data_root", DATA_ROOT,
        "--output_dir", f"{DATA_ROOT}/csr_annotations",
        "--format", "both"
    ]
    
    if use_all:
        cmd.append("--use_all_splits")
    
    print("Command:", " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        
        print(f"\n{'='*60}")
        print("✓ Conversion completed successfully!")
        print(f"{'='*60}\n")
        
        print("Next steps:")
        print("1. Verify the generated annotations:")
        print(f"   - Check: {DATA_ROOT}/csr_annotations/")
        print("\n2. Train the CSR model:")
        print(f"   python train.py \\")
        print(f"       --dataset tbx11k \\")
        print(f"       --data_root {DATA_ROOT}/imgs \\")
        print(f"       --train_file {DATA_ROOT}/csr_annotations/train_annotations.json \\")
        print(f"       --val_file {DATA_ROOT}/csr_annotations/val_annotations.json \\")
        print(f"       --test_file {DATA_ROOT}/csr_annotations/test_annotations.json \\")
        print(f"       --backbone resnet50 \\")
        print(f"       --batch_size 32 \\")
        print(f"       --save_dir checkpoints/tbx11k")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Conversion failed with error code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
