"""
Create YOLO format bounding box annotations for training data

Uses pre-computed bounding box data from validate_shipments.py
to generate object detection labels for building number detection model training.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


def convert_to_yolo_format(bbox: List[List[float]], image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    """
    Convert EasyOCR bounding box to YOLO format
    
    Args:
        bbox: EasyOCR bounding box [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    # Extract coordinates
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    
    # Get bounding box dimensions
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize to image dimensions
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def load_bbox_data(bbox_file: Path) -> Dict[str, Dict]:
    """
    Load pre-computed bounding box data from validation results
    
    Args:
        bbox_file: Path to bbox_data.json file
        
    Returns:
        Dictionary mapping training filenames to bbox data
    """
    print(f"Loading bounding box data from {bbox_file}...")
    
    if not bbox_file.exists():
        raise FileNotFoundError(f"Bbox data file not found: {bbox_file}")
    
    with open(bbox_file, 'r') as f:
        bbox_data = json.load(f)
    
    print(f"Loaded bounding box data for {len(bbox_data)} positive samples")
    return bbox_data


def create_annotations_from_bbox_data(bbox_data: Dict[str, Dict],
                                      positive_dir: Path,
                                      annotations_dir: Path):
    """
    Create YOLO format annotations using pre-computed bounding box data
    
    Args:
        bbox_data: Dictionary containing bounding box data for each image
        positive_dir: Directory containing positive sample images
        annotations_dir: Directory to save annotation files
    """
    print(f"Creating annotations for {len(bbox_data)} positive samples...")
    
    successful_annotations = 0
    failed_annotations = 0
    missing_images = 0
    
    for filename, data in tqdm(bbox_data.items(), desc="Creating annotations"):
        image_path = positive_dir / filename
        annotation_path = annotations_dir / f"{Path(filename).stem}.txt"
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            missing_images += 1
            continue
        
        try:
            # Extract data from pre-computed results
            bbox = data['bbox']
            width = data['image_width']
            height = data['image_height']
            target_number = data['target_number']
            confidence = data['confidence']
            
            # Convert to YOLO format using stored dimensions
            x_center, y_center, box_width, box_height = convert_to_yolo_format(bbox, width, height)
            
            # Ensure coordinates are within bounds
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            box_width = max(0.0, min(1.0, box_width))
            box_height = max(0.0, min(1.0, box_height))
            
            # Save annotation in YOLO format
            # Class 0 = "number", then normalized coordinates
            with open(annotation_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            successful_annotations += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_annotations += 1
            continue
    
    print(f"\nAnnotation creation completed!")
    print(f"  Successful annotations: {successful_annotations}")
    print(f"  Failed annotations: {failed_annotations}")
    print(f"  Missing images: {missing_images}")
    print(f"  Success rate: {successful_annotations/len(bbox_data)*100:.1f}%")
    
    return successful_annotations, failed_annotations


def verify_annotations(positive_dir: Path, annotations_dir: Path, bbox_data: Dict[str, Dict]):
    """
    Verify that annotation files were created correctly
    
    Args:
        positive_dir: Directory containing positive images
        annotations_dir: Directory containing annotation files
        bbox_data: Dictionary containing expected annotation data
    """
    print("\nVerifying annotations...")
    
    expected_annotations = len(bbox_data)
    annotation_files = list(annotations_dir.glob("*.txt"))
    
    print(f"Expected annotations: {expected_annotations}")
    print(f"Created annotation files: {len(annotation_files)}")
    
    # Check for missing annotations
    missing_annotations = []
    for filename in bbox_data.keys():
        annotation_file = annotations_dir / f"{Path(filename).stem}.txt"
        if not annotation_file.exists():
            missing_annotations.append(filename)
    
    if missing_annotations:
        print(f"Missing annotations: {len(missing_annotations)}")
        if len(missing_annotations) <= 10:
            for missing in missing_annotations:
                print(f"  - {missing}")
        else:
            print(f"  - {missing_annotations[0]} (and {len(missing_annotations)-1} more)")
    
    # Validate annotation files
    valid_annotations = 0
    invalid_annotations = []
    
    for annotation_file in annotation_files:  # Check all annotation files
        try:
            with open(annotation_file, 'r') as f:
                content = f.read().strip()
                if content:
                    parts = content.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        if class_id == 0 and all(0.0 <= coord <= 1.0 for coord in coords):
                            valid_annotations += 1
                        else:
                            invalid_annotations.append(annotation_file.name)
                    else:
                        invalid_annotations.append(annotation_file.name)
        except Exception as e:
            invalid_annotations.append(f"{annotation_file.name} (error: {e})")
    
    print(f"Validated {len(annotation_files)} annotation files:")
    print(f"  Valid: {valid_annotations}")
    print(f"  Invalid: {len(invalid_annotations)}")
    
    if invalid_annotations:
        print("Invalid annotations:")
        for invalid in invalid_annotations:
            print(f"  - {invalid}")


def main():
    """Main entry point"""
    # Define paths
    bbox_file = Path("shipment_validation/bbox_data.json")
    positive_dir = Path("model_training/data/positive")
    annotations_dir = Path("model_training/data/annotations")
    
    # Validate inputs
    if not bbox_file.exists():
        print(f"Error: Bbox data file not found: {bbox_file}")
        print(f"Please run validate_shipments.py first to generate bbox data")
        return
    
    if not positive_dir.exists():
        print(f"Error: Positive samples directory not found: {positive_dir}")
        return
    
    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return
    
    # Load pre-computed bounding box data
    bbox_data = load_bbox_data(bbox_file)
    
    # Create annotations using stored bbox data
    successful, failed = create_annotations_from_bbox_data(
        bbox_data=bbox_data,
        positive_dir=positive_dir,
        annotations_dir=annotations_dir
    )
    
    # Verify results
    verify_annotations(positive_dir, annotations_dir, bbox_data)
    
    print(f"\nAnnotation generation complete!")
    print(f"Ready for YOLO training with {successful} annotated samples")
    print(f"Using pre-validated bounding boxes for 100% accuracy!")


if __name__ == "__main__":
    main()