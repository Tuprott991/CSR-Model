"""
Convert TBX11K detection annotations to CSR-compatible format

TBX11K has bounding box annotations for TB detection.
CSR needs concept labels (findings) and disease labels (classes).

Strategy:
1. Convert bounding box presence to concept labels (binary: has_tb_box or not)
2. Convert detection categories to disease classes
3. Extract additional visual concepts if needed (optional)
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import pandas as pd


def load_xml_annotation(xml_path: str) -> Tuple[List, List]:
    """
    Load bounding box annotations from XML file
    Returns: (bboxes, labels)
    """
    if not os.path.exists(xml_path):
        return None, None
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    labels = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        bboxes.append(bbox)
        labels.append(name)
    
    return bboxes, labels


def category_to_disease(category: str) -> str:
    """
    Map TBX11K categories to disease labels
    """
    mapping = {
        'ActiveTuberculosis': 'active_tuberculosis',
        'ObsoletePulmonaryTuberculosis': 'latent_tuberculosis',
        'PulmonaryTuberculosis': 'uncertain_tuberculosis',
        'Healthy': 'normal',
        'Sick': 'sick_non_tb'
    }
    return mapping.get(category, 'unknown')


def extract_concepts_from_annotation(bboxes, labels, img_name: str) -> Dict[str, int]:
    """
    Extract binary concept labels from bounding box annotations
    
    Concepts derived from TB detection:
    - has_tb_lesion: whether there's any TB bounding box
    - has_active_tb: whether there's active TB
    - has_latent_tb: whether there's latent TB
    - has_uncertain_tb: whether there's uncertain TB
    - lesion_count_low: 1-2 lesions
    - lesion_count_medium: 3-5 lesions
    - lesion_count_high: >5 lesions
    - lesion_size_small: max bbox area < 5000 pixels
    - lesion_size_medium: max bbox area 5000-15000
    - lesion_size_large: max bbox area > 15000
    """
    concepts = {
        'has_tb_lesion': 0,
        'has_active_tb': 0,
        'has_latent_tb': 0,
        'has_uncertain_tb': 0,
        'lesion_count_low': 0,
        'lesion_count_medium': 0,
        'lesion_count_high': 0,
        'lesion_size_small': 0,
        'lesion_size_medium': 0,
        'lesion_size_large': 0
    }
    
    if bboxes is None or len(bboxes) == 0:
        return concepts
    
    # Has TB lesion
    concepts['has_tb_lesion'] = 1
    
    # Check lesion types
    for label in labels:
        if 'Active' in label:
            concepts['has_active_tb'] = 1
        elif 'Obsolete' in label:
            concepts['has_latent_tb'] = 1
        elif 'Pulmonary' in label:
            concepts['has_uncertain_tb'] = 1
    
    # Lesion count
    num_lesions = len(bboxes)
    if 1 <= num_lesions <= 2:
        concepts['lesion_count_low'] = 1
    elif 3 <= num_lesions <= 5:
        concepts['lesion_count_medium'] = 1
    elif num_lesions > 5:
        concepts['lesion_count_high'] = 1
    
    # Lesion size (use max bbox area)
    max_area = 0
    for bbox in bboxes:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        max_area = max(max_area, area)
    
    if max_area > 0:
        if max_area < 5000:
            concepts['lesion_size_small'] = 1
        elif max_area < 15000:
            concepts['lesion_size_medium'] = 1
        else:
            concepts['lesion_size_large'] = 1
    
    return concepts


def convert_tbx11k_to_csr_format(
    image_list_path: str,
    anno_dir: str,
    output_path: str,
    split_name: str = 'train'
):
    """
    Convert TBX11K annotations to CSR format
    
    Args:
        image_list_path: Path to TBX11K image list (e.g., TBX11K_train.txt)
        anno_dir: Directory containing XML annotations
        output_path: Output JSON file path
        split_name: Split name (train/val/test)
    """
    print(f"Converting {image_list_path} to CSR format...")
    
    # Read image list
    with open(image_list_path, 'r') as f:
        image_files = f.read().splitlines()
    
    # Define concept columns
    concept_columns = [
        'has_tb_lesion',
        'has_active_tb',
        'has_latent_tb',
        'has_uncertain_tb',
        'lesion_count_low',
        'lesion_count_medium',
        'lesion_count_high',
        'lesion_size_small',
        'lesion_size_medium',
        'lesion_size_large'
    ]
    
    annotations = {}
    
    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        # Keep the relative path (e.g., 'sick/s4387.png')
        img_path = img_path.strip()
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(anno_dir, img_name.replace('.png', '.xml'))
        
        # Load annotations
        bboxes, labels = load_xml_annotation(xml_path)
        
        # Extract concepts
        concepts = extract_concepts_from_annotation(bboxes, labels, img_name)
        
        # Determine disease class
        if bboxes is None or len(bboxes) == 0:
            # No TB bounding boxes - could be healthy or sick non-TB
            # We'll need additional info to distinguish, default to normal
            disease = 'normal'
        else:
            # Has TB lesions - determine which type
            if concepts['has_active_tb']:
                disease = 'active_tuberculosis'
            elif concepts['has_latent_tb']:
                disease = 'latent_tuberculosis'
            elif concepts['has_uncertain_tb']:
                disease = 'uncertain_tuberculosis'
            else:
                disease = 'tuberculosis'  # Generic TB
        
        # Create annotation entry - use full relative path as key
        annotations[img_path] = {
            'findings': [k for k, v in concepts.items() if v == 1],
            'disease': disease
        }
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Saved {len(annotations)} annotations to {output_path}")
    
    # Print statistics
    print("\nDataset Statistics:")
    diseases = [anno['disease'] for anno in annotations.values()]
    from collections import Counter
    disease_counts = Counter(diseases)
    for disease, count in disease_counts.items():
        print(f"  {disease}: {count}")
    
    concept_counts = Counter()
    for anno in annotations.values():
        for finding in anno['findings']:
            concept_counts[finding] += 1
    
    print("\nConcept Frequencies:")
    for concept, count in concept_counts.most_common():
        print(f"  {concept}: {count}")


def create_csv_format(
    image_list_path: str,
    anno_dir: str,
    output_csv_path: str,
    split_name: str = 'train'
):
    """
    Create CSV format for CSR (alternative to JSON)
    """
    print(f"Creating CSV format for {split_name}...")
    
    # Read image list
    with open(image_list_path, 'r') as f:
        image_files = f.read().splitlines()
    
    # Define columns
    concept_columns = [
        'has_tb_lesion', 'has_active_tb', 'has_latent_tb', 'has_uncertain_tb',
        'lesion_count_low', 'lesion_count_medium', 'lesion_count_high',
        'lesion_size_small', 'lesion_size_medium', 'lesion_size_large'
    ]
    
    data = []
    
    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        # Keep the relative path (e.g., 'sick/s4387.png')
        img_path = img_path.strip()
        img_name = os.path.basename(img_path)
        xml_path = os.path.join(anno_dir, img_name.replace('.png', '.xml'))
        
        # Load annotations
        bboxes, labels = load_xml_annotation(xml_path)
        
        # Extract concepts
        concepts = extract_concepts_from_annotation(bboxes, labels, img_name)
        
        # Determine disease
        if bboxes is None or len(bboxes) == 0:
            disease = 'normal'
        else:
            if concepts['has_active_tb']:
                disease = 'active_tuberculosis'
            elif concepts['has_latent_tb']:
                disease = 'latent_tuberculosis'
            elif concepts['has_uncertain_tb']:
                disease = 'uncertain_tuberculosis'
            else:
                disease = 'tuberculosis'
        
        # Create row - use full relative path
        row = {'image_path': img_path}
        for concept in concept_columns:
            row[concept] = concepts[concept]
        row['disease'] = disease
        
        data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    
    print(f"Saved CSV to {output_csv_path}")
    print(f"Shape: {df.shape}")


def main():
    parser = argparse.ArgumentParser(description='Convert TBX11K to CSR format')
    parser.add_argument('--data_root', type=str, default='TBX11K',
                        help='Root directory of TBX11K dataset')
    parser.add_argument('--output_dir', type=str, default='TBX11K/csr_annotations',
                        help='Output directory for CSR annotations')
    parser.add_argument('--format', type=str, default='both', 
                        choices=['json', 'csv', 'both'],
                        help='Output format')
    parser.add_argument('--use_all_splits', action='store_true',
                        help='Use all_train/all_val/all_test splits (includes extra datasets)')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define splits based on user preference
    if args.use_all_splits:
        print("Using 'all_*' splits (includes extra datasets)")
        splits = {
            'train': 'lists/all_train.txt',
            'val': 'lists/all_val.txt',
            'test': 'lists/all_test.txt'
        }
    else:
        print("Using 'TBX11K_*' splits (TBX11K dataset only)")
        splits = {
            'train': 'lists/TBX11K_train.txt',
            'val': 'lists/TBX11K_val.txt',
            'test': 'lists/all_test.txt'  # Test set includes extra datasets
        }
    
    anno_dir = data_root / 'annotations' / 'xml'
    
    print(f"\nDataset structure:")
    print(f"  Data root: {data_root}")
    print(f"  Images: {data_root / 'imgs'}")
    print(f"  Annotations: {anno_dir}")
    print(f"  Lists: {data_root / 'lists'}")
    print(f"  Output: {output_dir}\n")
    
    # Convert each split
    for split_name, list_file in splits.items():
        list_path = data_root / list_file
        
        if not list_path.exists():
            print(f"Warning: {list_path} not found, skipping...")
            continue
        
        # JSON format
        if args.format in ['json', 'both']:
            output_json = output_dir / f'{split_name}_annotations.json'
            convert_tbx11k_to_csr_format(
                str(list_path),
                str(anno_dir),
                str(output_json),
                split_name
            )
        
        # CSV format
        if args.format in ['csv', 'both']:
            output_csv = output_dir / f'{split_name}_annotations.csv'
            create_csv_format(
                str(list_path),
                str(anno_dir),
                str(output_csv),
                split_name
            )
        
        print(f"\n{'='*60}\n")
    
    print("Conversion completed!")
    print(f"\n{'='*60}")
    print("To use with CSR training:")
    print(f"{'='*60}\n")
    print("python train.py \\")
    print(f"    --dataset tbx11k \\")
    print(f"    --data_root {data_root / 'imgs'} \\")
    print(f"    --train_file {output_dir / 'train_annotations.json'} \\")
    print(f"    --val_file {output_dir / 'val_annotations.json'} \\")
    print(f"    --test_file {output_dir / 'test_annotations.json'} \\")
    print(f"    --backbone resnet50 \\")
    print(f"    --num_prototypes 10 \\")
    print(f"    --batch_size 32 \\")
    print(f"    --save_dir checkpoints/tbx11k\n")
    
    print("Note: Images are organized in subfolders:")
    print(f"  - {data_root / 'imgs' / 'health'} (healthy cases)")
    print(f"  - {data_root / 'imgs' / 'sick'} (sick but non-TB)")
    print(f"  - {data_root / 'imgs' / 'tb'} (TB cases)")
    print(f"  - {data_root / 'imgs' / 'test'} (test set)")
    print(f"  - {data_root / 'imgs' / 'extra'} (additional datasets)")


if __name__ == '__main__':
    main()
