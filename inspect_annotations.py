"""
Inspect converted CSR annotations to verify the conversion
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd


def inspect_json_annotations(json_path: str):
    """Inspect JSON annotations"""
    print(f"\nInspecting: {json_path}")
    print("="*60)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total images: {len(data)}")
    
    # Count diseases
    diseases = [anno['disease'] for anno in data.values()]
    disease_counts = Counter(diseases)
    
    print("\nDisease Distribution:")
    for disease, count in disease_counts.most_common():
        pct = 100 * count / len(data)
        print(f"  {disease:30s}: {count:5d} ({pct:5.1f}%)")
    
    # Count concepts
    all_concepts = []
    for anno in data.values():
        all_concepts.extend(anno['findings'])
    
    concept_counts = Counter(all_concepts)
    
    print("\nConcept Frequencies:")
    for concept, count in concept_counts.most_common():
        pct = 100 * count / len(data)
        print(f"  {concept:30s}: {count:5d} ({pct:5.1f}%)")
    
    # Show examples
    print("\nExample Annotations (first 5):")
    for i, (img_name, anno) in enumerate(list(data.items())[:5]):
        print(f"\n{i+1}. {img_name}")
        print(f"   Disease: {anno['disease']}")
        print(f"   Concepts: {', '.join(anno['findings']) if anno['findings'] else 'None'}")


def inspect_csv_annotations(csv_path: str):
    """Inspect CSV annotations"""
    print(f"\nInspecting: {csv_path}")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    
    print(f"Total images: {len(df)}")
    print(f"Columns: {df.shape[1]}")
    
    # Disease distribution
    print("\nDisease Distribution:")
    disease_counts = df['disease'].value_counts()
    for disease, count in disease_counts.items():
        pct = 100 * count / len(df)
        print(f"  {disease:30s}: {count:5d} ({pct:5.1f}%)")
    
    # Concept frequencies
    concept_cols = [col for col in df.columns if col not in ['image_path', 'disease']]
    
    print("\nConcept Frequencies:")
    for col in concept_cols:
        count = df[col].sum()
        pct = 100 * count / len(df)
        print(f"  {col:30s}: {int(count):5d} ({pct:5.1f}%)")
    
    # Show examples
    print("\nExample Annotations (first 5 rows):")
    print(df.head().to_string())


def compare_train_val_test(anno_dir: str):
    """Compare statistics across train/val/test splits"""
    anno_dir = Path(anno_dir)
    
    print("\n" + "="*60)
    print("Comparing Train/Val/Test Splits")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    
    # Load all splits
    split_data = {}
    for split in splits:
        json_path = anno_dir / f"{split}_annotations.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                split_data[split] = json.load(f)
    
    if not split_data:
        print("No annotation files found!")
        return
    
    # Compare sizes
    print("\nDataset Sizes:")
    for split, data in split_data.items():
        print(f"  {split:10s}: {len(data):5d} images")
    
    # Compare disease distributions
    print("\nDisease Distribution per Split:")
    print(f"{'Disease':<30s} | {'Train':>8s} | {'Val':>8s} | {'Test':>8s}")
    print("-" * 65)
    
    all_diseases = set()
    for data in split_data.values():
        all_diseases.update([anno['disease'] for anno in data.values()])
    
    for disease in sorted(all_diseases):
        row = [disease]
        for split in splits:
            if split in split_data:
                count = sum(1 for anno in split_data[split].values() 
                           if anno['disease'] == disease)
                pct = 100 * count / len(split_data[split])
                row.append(f"{count:4d} ({pct:4.1f}%)")
            else:
                row.append("-")
        print(f"{row[0]:<30s} | {' | '.join(row[1:])}")
    
    # Compare concept frequencies
    print("\nTop 5 Most Common Concepts per Split:")
    for split, data in split_data.items():
        all_concepts = []
        for anno in data.values():
            all_concepts.extend(anno['findings'])
        
        concept_counts = Counter(all_concepts)
        print(f"\n{split.upper()}:")
        for concept, count in concept_counts.most_common(5):
            pct = 100 * count / len(data)
            print(f"  {concept:30s}: {count:5d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Inspect CSR annotations')
    parser.add_argument('--anno_dir', type=str, default='TBX11K/csr_annotations',
                        help='Directory containing CSR annotations')
    parser.add_argument('--format', type=str, default='json',
                        choices=['json', 'csv', 'both'],
                        help='Annotation format to inspect')
    
    args = parser.parse_args()
    
    anno_dir = Path(args.anno_dir)
    
    if not anno_dir.exists():
        print(f"Error: Annotation directory not found: {anno_dir}")
        print("Have you run the conversion script?")
        return
    
    print("="*60)
    print("CSR Annotation Inspector")
    print("="*60)
    
    # Inspect each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        if args.format in ['json', 'both']:
            json_path = anno_dir / f"{split}_annotations.json"
            if json_path.exists():
                inspect_json_annotations(str(json_path))
            else:
                print(f"\nWarning: {json_path} not found")
        
        if args.format in ['csv', 'both']:
            csv_path = anno_dir / f"{split}_annotations.csv"
            if csv_path.exists():
                inspect_csv_annotations(str(csv_path))
            else:
                print(f"\nWarning: {csv_path} not found")
        
        print("\n" + "-"*60)
    
    # Compare splits
    compare_train_val_test(str(anno_dir))
    
    print("\n" + "="*60)
    print("Inspection complete!")
    print("="*60)


if __name__ == '__main__':
    main()
