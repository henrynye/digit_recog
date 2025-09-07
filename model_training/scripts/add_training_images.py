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
import pandas as pd
import cv2
import numpy as np

# Import existing modules
# Add parent directory to path (digit_recog root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from building_number_detector import BuildingNumberDetector


class TrainingDatasetExpander:
    """Expands existing YOLO training dataset with new images"""

    def __init__(
        self,
        new_images_dir: str,
        dataset_dir: str = "model_training/data",
        dataframe_path: str = "adding_images_df.pkl",
    ):
        """
        Initialize the dataset expander

        Args:
            new_images_dir: Directory containing new images to process
            dataset_dir: Existing YOLO dataset directory
        """
        self.new_images_dir = Path(new_images_dir)
        self.dataset_dir = Path(dataset_dir)

        self.dataframe_path = Path(dataframe_path)
        self.df = None

        # Working directories
        self.temp_dir = Path("temp_processing")
        self.results_file = self.temp_dir / "validation_results.txt"
        self.positive_dir = self.temp_dir / "positive"
        self.negative_dir = self.temp_dir / "negative"
        self.annotations_dir = self.temp_dir / "annotations"
        self.bbox_data_file = self.temp_dir / "bbox_data.json"

        # Initialize detector
        self.detector = None

        # Statistics
        self.stats = {
            "total_new_images": 0,
            "images_with_numbers": 0,
            "images_without_numbers": 0,
            "positive_samples": 0,
            "negative_samples": 0,
            "successful_annotations": 0,
            "added_to_train": 0,
            "added_to_val": 0,
        }

    def load_dataframe(self):
        """Load and process the DataFrame"""
        print(f"Loading DataFrame from {self.dataframe_path}...")

        if not self.dataframe_path.exists():
            raise FileNotFoundError(f"DataFrame file not found: {self.dataframe_path}")

        try:
            self.df = pd.read_pickle(self.dataframe_path)
            print(f"Loaded DataFrame with {len(self.df)} rows")
        except Exception as e:
            raise Exception(f"Could not load DataFrame: {e}")

    def setup_working_directories(self):
        """Create temporary working directories"""
        print("Setting up working directories...")

        directories = [
            self.temp_dir,
            self.positive_dir,
            self.negative_dir,
            self.annotations_dir,
        ]

        for directory in directories:
            shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def extract_address_number(
        self, address_line_1: str, address_line_2: str = None
    ) -> Optional[str]:
        """
        Extract the first number from address lines

        Args:
            address_line_1: Primary address line
            address_line_2: Secondary address line (optional)

        Returns:
            First number found in the address, or None if no number found
        """
        # Combine address lines, handling None values
        address_text = ""
        if address_line_1 and str(address_line_1).strip() != "nan":
            address_text = str(address_line_1).strip()

        if address_line_2 and str(address_line_2).strip() not in [
            "nan",
            "",
            ".",
            "None",
        ]:
            address_text += " " + str(address_line_2).strip()

        if not address_text:
            return None

        # Find the first number in the address
        match = re.search(r"\d+", address_text)
        return match.group() if match else None

    def get_address_info(self):
        """Process DataFrame to extract shipment data"""
        print("Processing shipments from DataFrame...")

        self.df = pd.read_pickle(self.dataframe_path)
        # Group by shipment_uid
        shipment_groups = self.df.groupby("shipment_uid")

        for shipment_uid, group in tqdm(shipment_groups, desc="Processing shipments"):
            # Get address information (should be same for all rows in group)
            first_row = group.iloc[0]
            address_number = self.extract_address_number(
                first_row["address_line_1"], first_row["address_line_2"]
            )

            # Set address_number for all records in the group in self.df
            self.df.loc[group.index, "address_number"] = address_number
            self.df.loc[group.index, "address_line_1"] = first_row["address_line_1"]
            self.df.loc[group.index, "address_line_2"] = first_row["address_line_2"]

        self.df.to_pickle(self.dataframe_path)

    def convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        return obj

    def detect_numbers_in_images(self):
        """Step 1: Detect building numbers in all new images"""
        print("Step 1: Detecting building numbers in new images...")
        self.df = pd.read_pickle(self.dataframe_path)
        if not self.new_images_dir.exists():
            raise FileNotFoundError(
                f"New images directory not found: {self.new_images_dir}"
            )

        # Initialize detector
        print("Initializing EasyOCR...")
        self.detector = BuildingNumberDetector(languages=["en"], verbose=False)

        # Get all image files
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        image_files = []
        for ext in extensions:
            image_files.extend(self.new_images_dir.glob(ext))

        if not image_files:
            raise ValueError(f"No image files found in {self.new_images_dir}")

        self.stats["total_new_images"] = len(image_files)
        print(f"Found {len(image_files)} images to process")

        image_filenames = [img.name for img in image_files]
        # Remove all rows in self.df where asset_name isn't in image_filenames
        self.df = self.df[self.df["asset_name"].isin(image_filenames)].copy()
        self.df.to_pickle(self.dataframe_path)

        print(f"Filtered DataFrame to {len(self.df)} rows")
        validation_results = pd.Series(dtype="object")
        # Process images and save results
        for idx, row in tqdm(
            self.df.iterrows(), desc="Detecting numbers", total=len(self.df)
        ):
            added = False
            try:
                image_path = self.new_images_dir / row["asset_name"]
                # detections = self.detector.detect_numbers(str(image_path), min_confidence=0.5)
                result = self.detector.check_for_expected_number(
                    str(image_path),
                    row["address_number"],
                    fuzzy_threshold=0.7,  # Slightly lower threshold for more flexibility
                    template_threshold=0.6,
                )
                if result["found"] and result["confidence"] > 0.5:
                    # Load image to get dimensions for YOLO normalization
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        height, width = image.shape[:2]
                        image_info = {
                            "image_name": row["asset_name"],
                            "bbox": result["bbox"],  # EasyOCR bbox format
                            "image_width": width,
                            "image_height": height,
                            "target_number": row["address_number"],
                            "confidence": result["confidence"],
                        }
                        validation_results[row["asset_name"]] = image_info
                        added = True

            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
            if not added:
                validation_results[row["asset_name"]] = None
            if idx == 0:
                bbox_data_converted = self.convert_numpy_types(
                    validation_results.to_list()
                )
                with open(self.bbox_data_file, "w") as f:
                    json.dump(bbox_data_converted, f, indent=2)

        self.df["image_info"] = self.df["asset_name"].map(validation_results)
        bbox_data_converted = self.convert_numpy_types(validation_results.to_list())
        with open(self.bbox_data_file, "w") as f:
            json.dump(bbox_data_converted, f, indent=2)
        self.df.to_pickle(self.dataframe_path)

    def copy_positive_negative_images(self):
        """Step 3: Copy images into positive or negative directories"""
        print("Step 3: Copying images into positive and negative directories...")
        self.df = pd.read_pickle(self.dataframe_path)
        source_dir = self.new_images_dir

        mask = self.df["image_info"].notna()
        positive_images = self.df.loc[mask, "asset_name"]
        negative_images = self.df.loc[~mask, "asset_name"]

        # Copy to positive/negative directories
        print(f"Copying {len(positive_images)} positive samples...")
        for image_path in tqdm(positive_images, desc="Copying positive"):
            shutil.copy2(source_dir / image_path, self.positive_dir)

        print(f"Copying {len(negative_images)} negative samples...")
        for image_path in tqdm(negative_images, desc="Copying negative"):
            shutil.copy2(source_dir / image_path, self.negative_dir)

        self.df.to_pickle(self.dataframe_path)

    def save_results_summary(self):
        """Save detailed results to a file"""
        self.df = pd.read_pickle(self.dataframe_path)
        print(f"Saving results summary to {self.results_file}...")

        with open(self.results_file, "w") as f:
            f.write("Validation Results\n")
            f.write("=" * 50 + "\n\n")

            # Overall statistics
            valid_count = len(self.df[self.df["image_info"].notna()])
            invalid_count = len(self.df) - valid_count

            f.write(f"Total images processed: {len(self.df)}\n")
            f.write(f"Valid images: {valid_count}\n")
            f.write(f"Invalid images: {invalid_count}\n")
            f.write(f"Success rate: {valid_count / len(self.df) * 100:.1f}%\n\n")
            # Detailed results
            f.write("Detailed Results:\n")
            f.write("-" * 30 + "\n")

            for asset_name, image_info in self.df[
                ["asset_name", "image_info"]
            ].itertuples(index=False, name=None):
                image_data = self.df[self.df["asset_name"] == asset_name].iloc[0]
                status = "VALID" if image_info is not None else "INVALID"
                address_num = image_data["address_number"]
                address_line_1 = image_data["address_line_1"]

                if image_info is not None:
                    confidence = image_info["confidence"]
                    f.write(f"{asset_name}: {status}\n")
                    f.write(f"  Address number: {address_num}\n")
                    f.write(f"  Address: {address_line_1}\n")
                    f.write(f"  Confidence: {confidence:.3f}\n")
                else:
                    f.write(f"{asset_name}: {status}\n")
                    f.write(f"  Address number: {address_num}\n")
                    f.write(f"  Address: {address_line_1}\n")

        print(f"Results summary saved to {self.results_file}")
        self.df.to_pickle(self.dataframe_path)

    def create_bbox_annotations(self):
        """Step 4: Create YOLO format bounding box annotations for positive samples"""
        print("Step 4: Creating bounding box annotations for positive samples...")

        self.df = pd.read_pickle(self.dataframe_path)
        filtered_df = self.df.loc[self.df["image_info"].notna()]

        for _, row in tqdm(filtered_df.iterrows(), desc="Creating annotations"):
            image_path = Path(row["asset_name"])
            try:  # Convert to YOLO format
                x_center, y_center, box_width, box_height = self.convert_to_yolo_format(
                    row["image_info"]["bbox"],
                    row["image_info"]["image_width"],
                    row["image_info"]["image_height"],
                )

                # Create annotation file
                annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
                with open(annotation_path, "w") as f:
                    f.write(
                        f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
                    )

            except Exception as e:
                print(f"Error creating annotation for {image_path.name}: {e}")

    @staticmethod
    def convert_to_yolo_format(
        bbox: List[List[float]], image_width: int, image_height: int
    ) -> Tuple[float, float, float, float]:
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
            raise FileNotFoundError(
                f"YOLO dataset directory not found: {self.dataset_dir}"
            )

        # Create dataset directories if they don't exist
        train_images_dir = self.dataset_dir / "images" / "train"
        train_labels_dir = self.dataset_dir / "labels" / "train"
        val_images_dir = self.dataset_dir / "images" / "val"
        val_labels_dir = self.dataset_dir / "labels" / "val"

        for directory in [
            train_images_dir,
            train_labels_dir,
            val_images_dir,
            val_labels_dir,
        ]:
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

        self.stats["added_to_train"] = train_added
        self.stats["added_to_val"] = val_added

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
            print(
                f"Warning: Training image-label mismatch ({train_images} != {train_labels})"
            )
        if val_images != val_labels:
            print(
                f"Warning: Validation image-label mismatch ({val_images} != {val_labels})"
            )

        return train_images == train_labels and val_images == val_labels

    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("DATASET EXPANSION SUMMARY")
        print("=" * 60)
        print(f"Total new images processed: {self.stats['total_new_images']}")
        print(f"Images with number detections: {self.stats['images_with_numbers']}")
        print(
            f"Images without number detections: {self.stats['images_without_numbers']}"
        )
        print(f"Positive samples (for training): {self.stats['positive_samples']}")
        print(f"Negative samples (background): {self.stats['negative_samples']}")
        print(f"Successful annotations created: {self.stats['successful_annotations']}")
        print(f"Added to training set: {self.stats['added_to_train']}")
        print(f"Added to validation set: {self.stats['added_to_val']}")

        total_added = self.stats["added_to_train"] + self.stats["added_to_val"]
        if total_added > 0:
            positive_rate = (self.stats["successful_annotations"] / total_added) * 100
            print(f"Positive sample rate: {positive_rate:.1f}%")

        print("\n✅ Dataset expansion completed successfully!")
        print("The YOLO dataset is ready for training with the expanded data.")

    def process_new_images(self, train_ratio: float = 0.8):
        """
        Complete pipeline to process new images and add to dataset

        Args:
            train_ratio: Ratio of data for training (default: 0.8)
        """
        try:
            # Set random seed for reproducible splits
            random.seed(42)

            print(f"Processing new images from: {self.new_images_dir}")
            print(f"Target dataset: {self.dataset_dir}")
            print(f"Train/validation split: {train_ratio:.1%}/{1 - train_ratio:.1%}")
            print("")

            # # Execute pipeline
            # self.load_dataframe()
            # self.setup_working_directories()
            # self.get_address_info()
            # self.detect_numbers_in_images()
            # self.copy_positive_negative_images()
            self.save_results_summary()
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
        """,
    )

    # Required arguments
    parser.add_argument(
        "-i",
        "--input-dir",
        default="property_identifier",
        help="Directory containing new images to add to training dataset",
    )

    # Optional arguments
    parser.add_argument(
        "--dataset-dir",
        default="model_training/data",
        help="Path to existing YOLO dataset directory (default: model_training/data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training vs validation (default: 0.8)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum digit length for positive samples (default: 1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=6,
        help="Maximum digit length for positive samples (default: 6)",
    )
    parser.add_argument(
        "--include-single-digits",
        action="store_true",
        help="Include single digit detections as positive samples (default: exclude)",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep temporary working files for debugging (default: clean up)",
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
        new_images_dir=args.input_dir, dataset_dir=args.dataset_dir
    )

    try:
        expander.process_new_images(train_ratio=args.train_ratio)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
