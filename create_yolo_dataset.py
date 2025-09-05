"""
Create YOLO dataset configuration for building number detection

Splits positive images into train/val sets and creates YOLO dataset configuration.
Includes negative images for hard negative mining.
"""

import random
from pathlib import Path
from typing import List, Tuple
import shutil

def get_image_annotation_pairs(positive_dir: Path, annotations_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Get pairs of images and their corresponding annotation files
    
    Args:
        positive_dir: Directory containing positive images
        annotations_dir: Directory containing annotation files
        
    Returns:
        List of (image_path, annotation_path) tuples
    """
    pairs = []
    for image_path in positive_dir.glob("*.jpg"):
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        if annotation_path.exists():
            pairs.append((image_path, annotation_path))
        else:
            print(f"Warning: No annotation found for {image_path.name}")
    
    return pairs

def create_train_val_split(pairs: List[Tuple[Path, Path]], train_ratio: float = 0.8) -> Tuple[List, List]:
    """
    Split image/annotation pairs into train and validation sets
    
    Args:
        pairs: List of (image_path, annotation_path) tuples
        train_ratio: Ratio of data to use for training (default: 0.8)
        
    Returns:
        (train_pairs, val_pairs) tuple
    """
    # Shuffle pairs for random split
    random.shuffle(pairs)
    
    # Calculate split point
    split_point = int(len(pairs) * train_ratio)
    
    train_pairs = pairs[:split_point]
    val_pairs = pairs[split_point:]
    
    return train_pairs, val_pairs

def copy_dataset_files(train_pairs: List[Tuple[Path, Path]], 
                      val_pairs: List[Tuple[Path, Path]],
                      dataset_dir: Path):
    """
    Copy images and annotations to YOLO dataset structure
    
    Args:
        train_pairs: Training image/annotation pairs
        val_pairs: Validation image/annotation pairs
        dataset_dir: Base dataset directory
    """
    # Create directory structure
    train_images_dir = dataset_dir / "images" / "train"
    train_labels_dir = dataset_dir / "labels" / "train"
    val_images_dir = dataset_dir / "images" / "val"
    val_labels_dir = dataset_dir / "labels" / "val"
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying {len(train_pairs)} training samples...")
    for image_path, annotation_path in train_pairs:
        # Copy image
        shutil.copy2(image_path, train_images_dir / image_path.name)
        # Copy annotation
        shutil.copy2(annotation_path, train_labels_dir / annotation_path.name)
    
    print(f"Copying {len(val_pairs)} validation samples...")
    for image_path, annotation_path in val_pairs:
        # Copy image
        shutil.copy2(image_path, val_images_dir / image_path.name)
        # Copy annotation
        shutil.copy2(annotation_path, val_labels_dir / annotation_path.name)

def add_negative_samples(negative_dir: Path, dataset_dir: Path, num_negatives: int = 200):
    """
    Add negative samples (images without annotations) for hard negative mining
    
    Args:
        negative_dir: Directory containing negative images
        dataset_dir: Base dataset directory
        num_negatives: Number of negative images to include
    """
    print(f"Adding {num_negatives} negative samples for hard negative mining...")
    
    # Get list of negative images
    negative_images = list(negative_dir.glob("*.jpg"))
    
    if len(negative_images) < num_negatives:
        print(f"Warning: Only {len(negative_images)} negative images available, using all")
        num_negatives = len(negative_images)
    
    # Sample random negative images
    selected_negatives = random.sample(negative_images, num_negatives)
    
    # Split negatives between train and val (80/20)
    train_negatives_count = int(num_negatives * 0.8)
    train_negatives = selected_negatives[:train_negatives_count]
    val_negatives = selected_negatives[train_negatives_count:]
    
    # Copy negative images (no annotations needed)
    train_images_dir = dataset_dir / "images" / "train"
    val_images_dir = dataset_dir / "images" / "val"
    
    for neg_image in train_negatives:
        shutil.copy2(neg_image, train_images_dir / neg_image.name)
    
    for neg_image in val_negatives:
        shutil.copy2(neg_image, val_images_dir / neg_image.name)
    
    print(f"Added {len(train_negatives)} negative training images")
    print(f"Added {len(val_negatives)} negative validation images")

def create_dataset_yaml(dataset_dir: Path):
    """
    Create YOLO dataset configuration file
    
    Args:
        dataset_dir: Base dataset directory
    """
    yaml_content = f"""# Building Number Detection Dataset
path: {dataset_dir.absolute()}
train: images/train
val: images/val

# Number of classes
nc: 1

# Class names
names:
  0: number
"""
    
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset configuration: {yaml_path}")

def verify_dataset(dataset_dir: Path):
    """
    Verify the dataset was created correctly
    
    Args:
        dataset_dir: Base dataset directory
    """
    print("\nVerifying dataset structure...")
    
    # Count files in each directory
    train_images = len(list((dataset_dir / "images" / "train").glob("*.jpg")))
    train_labels = len(list((dataset_dir / "labels" / "train").glob("*.txt")))
    val_images = len(list((dataset_dir / "images" / "val").glob("*.jpg")))
    val_labels = len(list((dataset_dir / "labels" / "val").glob("*.txt")))
    
    print(f"Training images: {train_images}")
    print(f"Training labels: {train_labels}")
    print(f"Validation images: {val_images}")
    print(f"Validation labels: {val_labels}")
    
    # Check for dataset.yaml
    yaml_path = dataset_dir / "dataset.yaml"
    if yaml_path.exists():
        print(f"✓ Dataset configuration found: {yaml_path}")
    else:
        print("✗ Dataset configuration missing!")
    
    # Verify some images have corresponding labels
    train_with_labels = 0
    val_with_labels = 0
    
    for img_path in (dataset_dir / "images" / "train").glob("*.jpg"):
        label_path = dataset_dir / "labels" / "train" / f"{img_path.stem}.txt"
        if label_path.exists():
            train_with_labels += 1
    
    for img_path in (dataset_dir / "images" / "val").glob("*.jpg"):
        label_path = dataset_dir / "labels" / "val" / f"{img_path.stem}.txt"
        if label_path.exists():
            val_with_labels += 1
    
    print(f"Training images with labels: {train_with_labels}")
    print(f"Validation images with labels: {val_with_labels}")
    
    negative_train = train_images - train_with_labels
    negative_val = val_images - val_with_labels
    
    print(f"Negative training images: {negative_train}")
    print(f"Negative validation images: {negative_val}")

def main():
    """Main entry point"""
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Define paths
    positive_dir = Path("model_training/data/positive")
    negative_dir = Path("model_training/data/negative") 
    annotations_dir = Path("model_training/data/annotations")
    dataset_dir = Path("model_training/data/yolo_dataset")
    
    # Validate inputs
    if not positive_dir.exists():
        print(f"Error: Positive images directory not found: {positive_dir}")
        return
    
    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return
    
    if not negative_dir.exists():
        print(f"Warning: Negative images directory not found: {negative_dir}")
        print("Proceeding without negative samples...")
    
    # Get image/annotation pairs
    pairs = get_image_annotation_pairs(positive_dir, annotations_dir)
    print(f"Found {len(pairs)} valid image/annotation pairs")
    
    if len(pairs) == 0:
        print("Error: No valid image/annotation pairs found!")
        return
    
    # Create train/val split
    train_pairs, val_pairs = create_train_val_split(pairs, train_ratio=0.8)
    print(f"Train split: {len(train_pairs)} samples")
    print(f"Validation split: {len(val_pairs)} samples")
    
    # Create dataset directory structure and copy files
    print(f"Creating YOLO dataset in {dataset_dir}...")
    copy_dataset_files(train_pairs, val_pairs, dataset_dir)
    
    # Add negative samples if available
    if negative_dir.exists():
        add_negative_samples(negative_dir, dataset_dir, num_negatives=200)
    
    # Create dataset configuration
    create_dataset_yaml(dataset_dir)
    
    # Verify results
    verify_dataset(dataset_dir)
    
    print(f"\n✅ YOLO dataset created successfully!")
    print(f"Dataset location: {dataset_dir}")
    print(f"Ready for YOLOv8 training!")

if __name__ == "__main__":
    main()