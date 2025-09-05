"""
Parse validation results to create training dataset

Extracts positive and negative samples from shipment validation results
for building number detection model training.
"""

import csv
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


def parse_validation_file(validation_file: Path) -> List[Dict]:
    """
    Parse the validation_results.txt file to extract shipment data
    
    Returns:
        List of dictionaries with shipment information
    """
    shipments = []
    
    with open(validation_file, 'r') as f:
        content = f.read()
    
    # Split into sections - each shipment starts with UUID
    sections = re.split(r'\n([a-f0-9-]{36}): (VALID|INVALID)\n', content)
    
    # Process sections (skip the header)
    for i in range(1, len(sections), 3):
        if i + 2 >= len(sections):
            break
            
        shipment_uid = sections[i]
        status = sections[i + 1] 
        details = sections[i + 2]
        
        # Extract details from the section
        address_match = re.search(r'Address number: (\d+)', details)
        valid_image_match = re.search(r'Valid image: (.+)', details)
        images_count_match = re.search(r'Images: (\d+)', details)
        address_text_match = re.search(r'Address: (.+)', details)
        
        if not address_match or not images_count_match:
            continue
            
        address_number = address_match.group(1)
        total_images = int(images_count_match.group(1))
        valid_image = valid_image_match.group(1).strip() if valid_image_match else None
        address_text = address_text_match.group(1).strip() if address_text_match else ""
        
        # Handle "None" as string
        if valid_image == "None":
            valid_image = None
            
        shipment_data = {
            'shipment_uid': shipment_uid,
            'status': status,
            'address_number': address_number,
            'address_text': address_text,
            'valid_image': valid_image,
            'total_images': total_images
        }
        
        shipments.append(shipment_data)
    
    return shipments


def get_shipment_images(shipment_uid: str, valid_shipments_dir: Path, invalid_shipments_dir: Path) -> List[str]:
    """
    Get all image files for a given shipment
    
    Args:
        shipment_uid: The shipment identifier
        valid_shipments_dir: Path to valid shipments directory
        invalid_shipments_dir: Path to invalid shipments directory
        
    Returns:
        List of image filenames in the shipment
    """
    # Check valid shipments first
    shipment_dir = valid_shipments_dir / shipment_uid
    if not shipment_dir.exists():
        # Check invalid shipments
        shipment_dir = invalid_shipments_dir / shipment_uid
    
    if not shipment_dir.exists():
        print(f"Warning: Shipment directory not found: {shipment_uid}")
        return []
    
    # Get all jpg files
    image_files = []
    for img_file in shipment_dir.glob("*.jpg"):
        image_files.append(img_file.name)
    
    return sorted(image_files)


def create_training_dataset(validation_file: Path, 
                          valid_shipments_dir: Path,
                          invalid_shipments_dir: Path,
                          output_file: Path) -> None:
    """
    Create training dataset CSV from validation results
    
    Args:
        validation_file: Path to validation_results.txt
        valid_shipments_dir: Path to valid_shipments directory
        invalid_shipments_dir: Path to invalid_shipments directory  
        output_file: Path to output CSV file
    """
    print(f"Parsing validation results from {validation_file}...")
    shipments = parse_validation_file(validation_file)
    
    print(f"Found {len(shipments)} shipments")
    
    # Create training samples
    training_samples = []
    positive_count = 0
    negative_count = 0
    
    for shipment in shipments:
        shipment_uid = shipment['shipment_uid']
        status = shipment['status']
        address_number = shipment['address_number']
        valid_image = shipment['valid_image']
        
        # Get all images for this shipment
        all_images = get_shipment_images(shipment_uid, valid_shipments_dir, invalid_shipments_dir)
        
        if not all_images:
            continue
            
        if status == "VALID" and valid_image:
            sample = {
                'shipment_uid': shipment_uid,
                'image_name': valid_image,
                'image_path': f"{'valid_shipments' if status == 'VALID' else 'invalid_shipments'}/{shipment_uid}/{valid_image}",
                'has_number': True,
                'number_value': address_number,
                'address_text': shipment['address_text']
            }
            training_samples.append(sample)
            positive_count += 1
            
        elif status == "INVALID":
            # All images are negative samples
            for img in all_images:
                sample = {
                    'shipment_uid': shipment_uid,
                    'image_name': img,
                    'image_path': f"invalid_shipments/{shipment_uid}/{img}",
                    'has_number': False,
                    'number_value': '',
                    'address_text': shipment['address_text']
                }
                training_samples.append(sample)
                negative_count += 1
    
    # Write to CSV
    print(f"Writing {len(training_samples)} samples to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['shipment_uid', 'image_name', 'image_path', 'has_number', 'number_value', 'address_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for sample in training_samples:
            writer.writerow(sample)
    
    print(f"\nDataset created successfully!")
    print(f"  Positive samples: {positive_count}")
    print(f"  Negative samples: {negative_count}")
    print(f"  Total samples: {len(training_samples)}")
    
    # Show some statistics
    valid_shipments = sum(1 for s in shipments if s['status'] == 'VALID')
    invalid_shipments = sum(1 for s in shipments if s['status'] == 'INVALID')
    
    print(f"\nShipment statistics:")
    print(f"  Valid shipments: {valid_shipments}")
    print(f"  Invalid shipments: {invalid_shipments}")
    
    # Show number distribution
    number_counts = defaultdict(int)
    for sample in training_samples:
        if sample['has_number']:
            number_counts[sample['number_value']] += 1
    
    if number_counts:
        print(f"\nMost common building numbers:")
        sorted_numbers = sorted(number_counts.items(), key=lambda x: int(x[0]))
        for number, count in sorted_numbers[:20]:
            print(f"  {number}: {count} samples")
        
        if len(sorted_numbers) > 20:
            print(f"  ... and {len(sorted_numbers) - 20} more numbers")


def main():
    """Main entry point"""
    # Define paths
    base_dir = Path("shipment_validation")
    validation_file = base_dir / "validation_results.txt"
    valid_shipments_dir = base_dir / "valid_shipments"
    invalid_shipments_dir = base_dir / "invalid_shipments"
    output_file = Path("training_dataset.csv")
    
    # Validate input files exist
    if not validation_file.exists():
        print(f"Error: Validation file not found: {validation_file}")
        return
    
    if not valid_shipments_dir.exists():
        print(f"Error: Valid shipments directory not found: {valid_shipments_dir}")
        return
        
    if not invalid_shipments_dir.exists():
        print(f"Error: Invalid shipments directory not found: {invalid_shipments_dir}")
        return
    
    # Create dataset
    create_training_dataset(
        validation_file=validation_file,
        valid_shipments_dir=valid_shipments_dir,
        invalid_shipments_dir=invalid_shipments_dir,
        output_file=output_file
    )


if __name__ == "__main__":
    main()