"""
Copy training images to organized directory structure

Organizes positive and negative samples from the validation dataset
into the model training directory structure.
"""

import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def copy_training_images(csv_file: Path, 
                        source_base: Path,
                        target_positive: Path, 
                        target_negative: Path,
                        max_negatives_per_shipment: int = 2):
    """
    Copy images from validated dataset to training directories
    
    Args:
        csv_file: Path to training_dataset.csv
        source_base: Base directory containing validated shipments
        target_positive: Directory for positive samples
        target_negative: Directory for negative samples  
        max_negatives_per_shipment: Max negative images per shipment
    """
    
    print(f"Reading training dataset from {csv_file}...")
    
    positive_samples = []
    negative_samples = []
    
    # Read CSV and separate positive/negative samples
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['has_number'] == 'True':
                positive_samples.append(row)
            else:
                negative_samples.append(row)
    
    print(f"Found {len(positive_samples)} positive samples")
    print(f"Found {len(negative_samples)} negative samples")
    
    # Copy positive samples
    print("\nCopying positive samples...")
    positive_copied = 0
    positive_errors = 0
    
    for sample in tqdm(positive_samples, desc="Positive images"):
        source_path = source_base / "shipment_validation" / sample['image_path']
        
        # Create new filename with shipment prefix
        shipment_uid = sample['shipment_uid']
        original_name = sample['image_name']
        new_filename = f"{shipment_uid}_{original_name}"
        target_path = target_positive / new_filename
        
        try:
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                positive_copied += 1
            else:
                print(f"Warning: Source file not found: {source_path}")
                positive_errors += 1
        except Exception as e:
            print(f"Error copying {source_path}: {e}")
            positive_errors += 1
    
    # Sample negative images (limit per shipment to avoid imbalance)
    print("\nSampling and copying negative samples...")
    
    # Group negative samples by shipment
    negatives_by_shipment = defaultdict(list)
    for sample in negative_samples:
        negatives_by_shipment[sample['shipment_uid']].append(sample)
    
    # Sample from each shipment
    selected_negatives = []
    for shipment_uid, samples in negatives_by_shipment.items():
        # Take up to max_negatives_per_shipment random samples
        selected = random.sample(samples, min(len(samples), max_negatives_per_shipment))
        selected_negatives.extend(selected)
    
    print(f"Selected {len(selected_negatives)} negative samples from {len(negatives_by_shipment)} shipments")
    
    negative_copied = 0
    negative_errors = 0
    
    for sample in tqdm(selected_negatives, desc="Negative images"):
        source_path = source_base / "shipment_validation" / sample['image_path']
        
        # Create new filename with shipment prefix
        shipment_uid = sample['shipment_uid']
        original_name = sample['image_name']
        new_filename = f"{shipment_uid}_{original_name}"
        target_path = target_negative / new_filename
        
        try:
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                negative_copied += 1
            else:
                print(f"Warning: Source file not found: {source_path}")
                negative_errors += 1
        except Exception as e:
            print(f"Error copying {source_path}: {e}")
            negative_errors += 1
    
    # Summary
    print(f"\nCopy completed!")
    print(f"Positive images:")
    print(f"  Copied: {positive_copied}")
    print(f"  Errors: {positive_errors}")
    print(f"Negative images:")
    print(f"  Copied: {negative_copied}")
    print(f"  Errors: {negative_errors}")
    print(f"Total images copied: {positive_copied + negative_copied}")


def main():
    """Main entry point"""
    # Set random seed for reproducible negative sampling
    random.seed(42)
    
    # Define paths
    csv_file = Path("training_dataset.csv")
    source_base = Path(".")  # Base directory containing shipment_validation
    target_positive = Path("model_training/data/positive")
    target_negative = Path("model_training/data/negative")
    
    # Validate inputs
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return
    
    if not target_positive.exists() or not target_negative.exists():
        print(f"Error: Target directories not found. Run Step 1.2 first.")
        return
    
    # Copy images
    copy_training_images(
        csv_file=csv_file,
        source_base=source_base,
        target_positive=target_positive,
        target_negative=target_negative,
        max_negatives_per_shipment=2
    )
    
    # Verify results
    positive_count = len(list(target_positive.glob("*.jpg")))
    negative_count = len(list(target_negative.glob("*.jpg")))
    
    print(f"\nFinal verification:")
    print(f"Positive directory: {positive_count} images")
    print(f"Negative directory: {negative_count} images")


if __name__ == "__main__":
    main()