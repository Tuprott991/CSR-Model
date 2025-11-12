"""
Create proper train/val/test splits for TBX11K with annotations
Uses only images that have XML annotations
"""

import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def main():
    data_root = Path('TBX11K')
    anno_dir = data_root / 'annotations' / 'xml'
    output_dir = data_root / 'lists'
    
    # Get all images with XML annotations
    xml_files = list(anno_dir.glob('*.xml'))
    image_paths = []
    
    for xml_file in xml_files:
        img_name = xml_file.stem + '.png'
        
        # Check which subfolder the image is in
        for subfolder in ['health', 'sick', 'tb']:
            img_path = data_root / 'imgs' / subfolder / img_name
            if img_path.exists():
                # Use relative path from imgs/
                image_paths.append(f"{subfolder}/{img_name}")
                break
    
    print(f"Found {len(image_paths)} images with annotations")
    
    # Shuffle
    random.seed(42)
    random.shuffle(image_paths)
    
    # Split: 70% train, 15% val, 15% test
    train_val, test = train_test_split(image_paths, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, random_state=42)  # 0.176 of 0.85 = 0.15 of total
    
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    
    # Save splits
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / 'TBX11K_train_new.txt', 'w') as f:
        f.write('\n'.join(train))
    
    with open(output_dir / 'TBX11K_val_new.txt', 'w') as f:
        f.write('\n'.join(val))
    
    with open(output_dir / 'TBX11K_test_new.txt', 'w') as f:
        f.write('\n'.join(test))
    
    print(f"\nSaved to {output_dir}")
    print("\nRun conversion with:")
    print("python convert_tbx11k_to_csr.py --data_root TBX11K --output_dir TBX11K/csr_annotations_new")
    print("\nThen update train.py to use:")
    print("  --train_file TBX11K/csr_annotations_new/train_annotations.json")
    print("  --val_file TBX11K/csr_annotations_new/val_annotations.json")
    print("  --test_file TBX11K/csr_annotations_new/test_annotations.json")

if __name__ == '__main__':
    main()
