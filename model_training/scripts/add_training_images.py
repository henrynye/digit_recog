"""
Add New Training Images to YOLO Dataset

This script processes a flat directory of new images and incorporates them
into the existing YOLO training dataset. It replicates all the preprocessing
steps that were applied to the original dataset.

Usage:
    python add_training_images.py -i /path/to/new/images

Pipeline:
1. Detect building numbers in new images using EasyOCR
2. Classify images as positive (likely contains numbers) or negative
3. Create YOLO format bounding box annotations for positive samples
4. Add images to existing YOLO dataset with proper train/val split
5. Create empty label files for negative samples
6. Clean up cache files for fresh training
"""

import random
import shutil
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import sys

# Import existing modules
from building_number_detector import BuildingNumberDetector
from organize_by_digits import ImageOrganizer


class TrainingDatasetExpander:
    """Expands existing YOLO training dataset with new images"""
    
    def __init__(self, new_images_dir: str, dataset_dir: str = "../data/dataset"):
        """
        Initialize the dataset expander
        
        Args:
            new_images_dir: Directory containing new images to process
            dataset_dir: Existing YOLO dataset directory
        """
        self.new_images_dir = Path(new_images_dir)
        self.dataset_dir = Path(dataset_dir)
        
        # Working directories
        self.temp_dir = Path("temp_processing")
        self.results_file = self.temp_dir / "detection_results.txt"
        self.organized_dir = self.temp_dir / "organized_images"
        self.positive_dir = self.temp_dir / "positive"
        self.negative_dir = self.temp_dir / "negative"
        self.annotations_dir = self.temp_dir / "annotations"
        self.bbox_data_file = self.temp_dir / "bbox_data.json"
        
        # Initialize detector
        self.detector = None
        
        # Statistics
        self.stats = {
            'total_new_images': 0,
            'images_with_numbers': 0,
            'images_without_numbers': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'successful_annotations': 0,
            'added_to_train': 0,
            'added_to_val': 0
        }
    
    def setup_working_directories(self):
        """Create temporary working directories"""
        print("Setting up working directories...")
        
        directories = [
            self.temp_dir,
            self.organized_dir,
            self.positive_dir,
            self.negative_dir,
            self.annotations_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def cleanup_working_directories(self, keep_results: bool = False):
        """Clean up temporary working directories"""
        if not keep_results:
            print("Cleaning up temporary files...")
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        else:
            print(f"Working files preserved in: {self.temp_dir}")
    
    def detect_numbers_in_images(self):
        """Step 1: Detect building numbers in all new images"""
        print("Step 1: Detecting building numbers in new images...")
        
        if not self.new_images_dir.exists():
            raise FileNotFoundError(f"New images directory not found: {self.new_images_dir}")
        
        # Initialize detector
        print("Initializing EasyOCR...")
        self.detector = BuildingNumberDetector(languages=['en'], verbose=False)
        
        # Get all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(self.new_images_dir.glob(ext))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.new_images_dir}")
        
        self.stats['total_new_images'] = len(image_files)
        print(f"Found {len(image_files)} images to process")
        
        # Process images and save results
        results = {}
        with open(self.results_file, 'w') as f:
            f.write("Building Number Detection Results\n")
            f.write("=" * 40 + "\n\n")
            
            for image_path in tqdm(image_files, desc="Detecting numbers"):
                try:
                    detections = self.detector.detect_numbers(str(image_path), min_confidence=0.5)
                    results[image_path.name] = detections
                    
                    # Write to results file in format expected by organize_by_digits.py
                    f.write(f"{image_path.name}:\n")
                    if detections:
                        digits_list = [det['digits'] for det in detections if det['digits']]
                        if digits_list:
                            f.write(f"    Digits: {', '.join(digits_list)}\n")
                            self.stats['images_with_numbers'] += 1
                        else:
                            f.write("    No detections found\n")
                            self.stats['images_without_numbers'] += 1
                    else:
                        f.write("    No detections found\n")
                        self.stats['images_without_numbers'] += 1
                    f.write("\n")
                    
                except Exception as e:
                    print(f"Error processing {image_path.name}: {e}")
                    f.write(f"{image_path.name}:\n")
                    f.write("    No detections found\n\n")
                    self.stats['images_without_numbers'] += 1
        
        print(f"Detection complete: {self.stats['images_with_numbers']} with numbers, "
              f"{self.stats['images_without_numbers']} without")
    
    def organize_images_by_digits(self):
        """Step 2: Organize images by detected digits"""
        print("Step 2: Organizing images by detected digits...")
        
        # Use the existing ImageOrganizer
        organizer = ImageOrganizer(
            results_file=str(self.results_file),
            source_dir=str(self.new_images_dir),
            output_dir=str(self.organized_dir)
        )
        organizer.organize()
        
        # Count organized results
        digit_dirs = [d for d in self.organized_dir.iterdir() if d.is_dir() and d.name != 'no_numbers']
        no_numbers_dir = self.organized_dir / 'no_numbers'
        
        print(f"Images organized into {len(digit_dirs)} digit directories")
        if no_numbers_dir.exists():
            no_number_count = len(list(no_numbers_dir.glob("*.jpg")))
            print(f"Found {no_number_count} images with no detections")
    
    def classify_positive_negative_samples(self, 
                                         min_digit_length: int = 1,
                                         max_digit_length: int = 6,
                                         exclude_single_digits: bool = True):
        """Step 3: Classify images as positive or negative samples"""
        print("Step 3: Classifying positive and negative samples...")
        
        positive_images = []
        negative_images = []
        
        # Process digit directories for positive samples
        for digit_dir in self.organized_dir.iterdir():
            if not digit_dir.is_dir() or digit_dir.name == 'no_numbers':
                continue
            
            digit_value = digit_dir.name
            
            # Apply filtering criteria
            if not digit_value.isdigit():
                continue
            
            digit_length = len(digit_value)
            if digit_length < min_digit_length or digit_length > max_digit_length:
                continue
            
            if exclude_single_digits and digit_length == 1:
                continue
            
            # Add images from this directory as positive samples
            for image_file in digit_dir.glob("*.jpg"):
                positive_images.append(image_file)
        
        # Process no_numbers directory for negative samples
        no_numbers_dir = self.organized_dir / 'no_numbers'
        if no_numbers_dir.exists():
            negative_images = list(no_numbers_dir.glob("*.jpg"))
        
        # Also add single digit detections as negative samples if excluded
        if exclude_single_digits:
            for digit_dir in self.organized_dir.iterdir():
                if digit_dir.is_dir() and digit_dir.name.isdigit() and len(digit_dir.name) == 1:
                    negative_images.extend(list(digit_dir.glob("*.jpg")))
        
        # Copy to positive/negative directories
        print(f"Copying {len(positive_images)} positive samples...")
        for image_path in tqdm(positive_images, desc="Copying positive"):
            shutil.copy2(image_path, self.positive_dir / image_path.name)
        
        print(f"Copying {len(negative_images)} negative samples...")
        for image_path in tqdm(negative_images, desc="Copying negative"):
            shutil.copy2(image_path, self.negative_dir / image_path.name)
        
        self.stats['positive_samples'] = len(positive_images)
        self.stats['negative_samples'] = len(negative_images)
        
        print(f"Classification complete: {self.stats['positive_samples']} positive, "
              f"{self.stats['negative_samples']} negative")
    
    def create_bbox_annotations(self):
        """Step 4: Create YOLO format bounding box annotations for positive samples"""
        print("Step 4: Creating bounding box annotations for positive samples...")
        
        bbox_data = {}
        successful = 0
        failed = 0
        
        positive_images = list(self.positive_dir.glob("*.jpg"))
        
        for image_path in tqdm(positive_images, desc="Creating annotations"):
            try:
                # Detect numbers with high confidence to get bounding boxes
                detections = self.detector.detect_numbers(str(image_path), min_confidence=0.5)
                
                if detections:
                    # Use the highest confidence detection
                    best_detection = detections[0]
                    
                    # Get image dimensions
                    import cv2
                    image = cv2.imread(str(image_path))
                    height, width = image.shape[:2]
                    
                    # Store bbox data
                    bbox_data[image_path.name] = {
                        'bbox': best_detection['bbox'],
                        'image_width': width,
                        'image_height': height,
                        'target_number': best_detection['digits'],
                        'confidence': best_detection['confidence']
                    }
                    
                    # Convert to YOLO format
                    x_center, y_center, box_width, box_height = self.convert_to_yolo_format(
                        best_detection['bbox'], width, height
                    )
                    
                    # Create annotation file
                    annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
                    with open(annotation_path, 'w') as f:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                    
                    successful += 1
                else:
                    # No detections - this shouldn't happen for positive samples
                    print(f"Warning: No detections in positive sample {image_path.name}")
                    failed += 1
                    
            except Exception as e:
                print(f"Error creating annotation for {image_path.name}: {e}")
                failed += 1
        
        # Save bbox data for reference
        with open(self.bbox_data_file, 'w') as f:
            json.dump(bbox_data, f, indent=2)
        
        self.stats['successful_annotations'] = successful
        print(f"Annotation creation complete: {successful} successful, {failed} failed")
    
    @staticmethod
    def convert_to_yolo_format(bbox: List[List[float]], image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """Convert EasyOCR bounding box to YOLO format"""
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
        x_center_norm = max(0.0, min(1.0, x_center / image_width))
        y_center_norm = max(0.0, min(1.0, y_center / image_height))
        width_norm = max(0.0, min(1.0, width / image_width))
        height_norm = max(0.0, min(1.0, height / image_height))
        
        return x_center_norm, y_center_norm, width_norm, height_norm
    
    def add_to_existing_dataset(self, train_ratio: float = 0.8):
        """Step 5: Add new images to existing YOLO dataset"""
        print("Step 5: Adding new images to existing YOLO dataset...")
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"YOLO dataset directory not found: {self.dataset_dir}")
        
        # Create dataset directories if they don't exist
        train_images_dir = self.dataset_dir / "images" / "train"
        train_labels_dir = self.dataset_dir / "labels" / "train"
        val_images_dir = self.dataset_dir / "images" / "val"
        val_labels_dir = self.dataset_dir / "labels" / "val"
        
        for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Get positive and negative samples
        positive_images = list(self.positive_dir.glob("*.jpg"))
        negative_images = list(self.negative_dir.glob("*.jpg"))
        
        # Combine and shuffle for random split
        all_images = positive_images + negative_images
        random.shuffle(all_images)
        
        # Split into train/val
        split_point = int(len(all_images) * train_ratio)
        train_images = all_images[:split_point]
        val_images = all_images[split_point:]
        
        # Copy training images
        train_added = 0
        for image_path in tqdm(train_images, desc="Adding to train set"):
            # Copy image
            target_image_path = train_images_dir / image_path.name
            shutil.copy2(image_path, target_image_path)
            
            # Copy or create label
            annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
            target_label_path = train_labels_dir / f"{image_path.stem}.txt"
            
            if annotation_path.exists():
                # Has annotation - copy it
                shutil.copy2(annotation_path, target_label_path)
            else:
                # No annotation - create empty file for negative sample
                target_label_path.touch()
            
            train_added += 1
        
        # Copy validation images
        val_added = 0
        for image_path in tqdm(val_images, desc="Adding to val set"):
            # Copy image
            target_image_path = val_images_dir / image_path.name
            shutil.copy2(image_path, target_image_path)
            
            # Copy or create label
            annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
            target_label_path = val_labels_dir / f"{image_path.stem}.txt"
            
            if annotation_path.exists():
                # Has annotation - copy it
                shutil.copy2(annotation_path, target_label_path)
            else:
                # No annotation - create empty file for negative sample
                target_label_path.touch()
            
            val_added += 1
        
        self.stats['added_to_train'] = train_added
        self.stats['added_to_val'] = val_added
        
        print(f"Successfully added {train_added} images to training set")
        print(f"Successfully added {val_added} images to validation set")
    
    def clean_cache_files(self):
        """Step 6: Clean up cache files for fresh training"""
        print("Step 6: Cleaning cache files...")
        
        cache_files = list(self.dataset_dir.glob("**/*.cache"))
        removed_count = 0
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Warning: Could not remove cache file {cache_file}: {e}")
        
        print(f"Removed {removed_count} cache files")
    
    def verify_dataset_integrity(self):
        """Verify the updated dataset has proper structure"""
        print("Verifying dataset integrity...")
        
        train_images = len(list((self.dataset_dir / "images" / "train").glob("*.jpg")))
        train_labels = len(list((self.dataset_dir / "labels" / "train").glob("*.txt")))
        val_images = len(list((self.dataset_dir / "images" / "val").glob("*.jpg")))
        val_labels = len(list((self.dataset_dir / "labels" / "val").glob("*.txt")))
        
        print(f"Final dataset structure:")
        print(f"  Training: {train_images} images, {train_labels} labels")
        print(f"  Validation: {val_images} images, {val_labels} labels")
        
        # Check for image-label mismatches
        if train_images != train_labels:
            print(f"Warning: Training image-label mismatch ({train_images} != {train_labels})")
        if val_images != val_labels:
            print(f"Warning: Validation image-label mismatch ({val_images} != {val_labels})")
        
        return train_images == train_labels and val_images == val_labels
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("DATASET EXPANSION SUMMARY")
        print("="*60)
        print(f"Total new images processed: {self.stats['total_new_images']}")
        print(f"Images with number detections: {self.stats['images_with_numbers']}")
        print(f"Images without number detections: {self.stats['images_without_numbers']}")
        print(f"Positive samples (for training): {self.stats['positive_samples']}")
        print(f"Negative samples (background): {self.stats['negative_samples']}")
        print(f"Successful annotations created: {self.stats['successful_annotations']}")
        print(f"Added to training set: {self.stats['added_to_train']}")
        print(f"Added to validation set: {self.stats['added_to_val']}")
        
        total_added = self.stats['added_to_train'] + self.stats['added_to_val']
        if total_added > 0:
            positive_rate = (self.stats['successful_annotations'] / total_added) * 100
            print(f"Positive sample rate: {positive_rate:.1f}%")
        
        print("\n✅ Dataset expansion completed successfully!")
        print("The YOLO dataset is ready for training with the expanded data.")
    
    def process_new_images(self,
                          train_ratio: float = 0.8,
                          min_digit_length: int = 1,
                          max_digit_length: int = 6,
                          exclude_single_digits: bool = True,
                          keep_working_files: bool = False):
        """
        Complete pipeline to process new images and add to dataset
        
        Args:
            train_ratio: Ratio of data for training (default: 0.8)
            min_digit_length: Minimum length of detected digits to consider positive
            max_digit_length: Maximum length of detected digits to consider positive
            exclude_single_digits: Whether to exclude single digit detections
            keep_working_files: Whether to keep temporary working files
        """
        try:
            # Set random seed for reproducible splits
            random.seed(42)
            
            print(f"Processing new images from: {self.new_images_dir}")
            print(f"Target dataset: {self.dataset_dir}")
            print(f"Train/validation split: {train_ratio:.1%}/{1-train_ratio:.1%}")
            print("")
            
            # Execute pipeline
            self.setup_working_directories()
            self.detect_numbers_in_images()
            self.organize_images_by_digits()
            self.classify_positive_negative_samples(
                min_digit_length=min_digit_length,
                max_digit_length=max_digit_length,
                exclude_single_digits=exclude_single_digits
            )
            self.create_bbox_annotations()
            self.add_to_existing_dataset(train_ratio=train_ratio)
            self.clean_cache_files()
            
            # Verify and summarize
            if self.verify_dataset_integrity():
                print("✅ Dataset integrity verified")
            else:
                print("⚠️ Dataset integrity issues detected")
            
            self.print_summary()
            
        except Exception as e:
            print(f"Error during processing: {e}")
            raise
        
        finally:
            # Cleanup
            self.cleanup_working_directories(keep_results=keep_working_files)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Add new images to existing YOLO training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - add all images from directory
  python add_training_images.py -i /path/to/new/images
  
  # Custom train/val split and filtering
  python add_training_images.py -i new_photos/ --train-ratio 0.85 --min-length 2 --max-length 4
  
  # Include single digit detections and keep working files
  python add_training_images.py -i images/ --include-single-digits --keep-files
  
  # Specify custom dataset location
  python add_training_images.py -i photos/ --dataset-dir custom_yolo_dataset/
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input-dir',
        required=True,
        help='Directory containing new images to add to training dataset'
    )
    
    # Optional arguments
    parser.add_argument(
        '--dataset-dir',
        default='../data/dataset',
        help='Path to existing YOLO dataset directory (default: ../data/dataset)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of data for training vs validation (default: 0.8)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=1,
        help='Minimum digit length for positive samples (default: 1)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=6,
        help='Maximum digit length for positive samples (default: 6)'
    )
    parser.add_argument(
        '--include-single-digits',
        action='store_true',
        help='Include single digit detections as positive samples (default: exclude)'
    )
    parser.add_argument(
        '--keep-files',
        action='store_true',
        help='Keep temporary working files for debugging (default: clean up)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.train_ratio < 1.0):
        print("Error: Train ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    # Run the expander
    expander = TrainingDatasetExpander(
        new_images_dir=args.input_dir,
        dataset_dir=args.dataset_dir
    )
    
    try:
        expander.process_new_images(
            train_ratio=args.train_ratio,
            min_digit_length=args.min_length,
            max_digit_length=args.max_length,
            exclude_single_digits=not args.include_single_digits,
            keep_working_files=args.keep_files
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        expander.cleanup_working_directories(keep_results=args.keep_files)
        sys.exit(1)
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        expander.cleanup_working_directories(keep_results=True)
        sys.exit(1)


if __name__ == "__main__":
    main()