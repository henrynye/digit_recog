"""
Organize Images by Detected Digits

Reads OCR detection results and organizes images into directories
based on the numbers detected in them.
"""

import os
import shutil
import re
import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set
from tqdm import tqdm


class ImageOrganizer:
    """Organizes images based on OCR detection results"""
    
    def __init__(self, results_file: str, source_dir: str, output_dir: str = "predicted_digits"):
        """
        Initialize the organizer
        
        Args:
            results_file: Path to the OCR results file
            source_dir: Directory containing source images
            output_dir: Output directory for organized images
        """
        self.results_file = Path(results_file)
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Storage for results
        self.image_detections = {}  # {filename: [list of detected numbers]}
        self.all_detected_numbers = set()
        self.errors = []
        
    def parse_results_file(self):
        """Parse the OCR results file and extract detections"""
        print(f"Parsing results from {self.results_file}...")
        
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        try:
            with open(self.results_file, 'r') as f:
                content = f.read()
        except Exception as e:
            raise Exception(f"Could not read results file: {e}")
        
        # Split content into lines
        lines = content.split('\n')
        current_filename = None
        current_digits = []
        
        # Patterns for parsing the actual file format
        filename_pattern = re.compile(r'^([^:]+\.(jpg|jpeg|png|bmp|tiff)):$', re.IGNORECASE)
        digits_pattern = re.compile(r'^\s+Digits:\s*(.+)$')
        no_detection_pattern = re.compile(r'^\s+No detections found$')
        
        for line in lines:
            # Don't strip the line yet - we need to preserve indentation
            
            # Check for filename (no leading whitespace)
            if not line.startswith(' ') and line.strip():
                filename_match = filename_pattern.match(line.strip())
                if filename_match:
                    # Save previous filename results
                    if current_filename is not None:
                        self.image_detections[current_filename] = current_digits.copy()
                        if current_digits:
                            self.all_detected_numbers.update(current_digits)
                    
                    # Start new filename
                    current_filename = filename_match.group(1)
                    current_digits = []
                    continue
            
            # Check for "No detections found" (with leading whitespace)
            if no_detection_pattern.match(line):
                # No digits for this image
                current_digits = []
                continue
            
            # Check for digits line (with leading whitespace)
            digits_match = digits_pattern.match(line)
            if digits_match:
                digits_text = digits_match.group(1).strip()
                # Only extract pure numbers - filter out any non-numeric characters
                if digits_text and digits_text.lower() != 'none':
                    # Extract only digits from the digits field
                    pure_digits = re.sub(r'[^0-9]', '', digits_text)
                    if pure_digits:  # Only add if there are actual digits
                        current_digits.append(pure_digits)
        
        # Don't forget the last filename
        if current_filename is not None:
            self.image_detections[current_filename] = current_digits.copy()
            if current_digits:
                self.all_detected_numbers.update(current_digits)
        
        print(f"Parsed {len(self.image_detections)} images")
        print(f"Found {len(self.all_detected_numbers)} unique detected numbers: {sorted(self.all_detected_numbers, key=int)}")
        
        # Count images with no detections
        no_detections_count = sum(1 for digits in self.image_detections.values() if not digits)
        if no_detections_count > 0:
            print(f"Found {no_detections_count} images with no detections")
    
    def create_directory_structure(self):
        """Create the output directory structure"""
        print(f"Creating directory structure in {self.output_dir}...")
        
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each detected number
        directories_created = []
        for number in sorted(self.all_detected_numbers, key=int):
            number_dir = self.output_dir / str(number)
            number_dir.mkdir(exist_ok=True)
            directories_created.append(str(number))
        
        # Create 'no_numbers' directory if there are images with no detections
        no_detections_count = sum(1 for digits in self.image_detections.values() if not digits)
        if no_detections_count > 0:
            no_numbers_dir = self.output_dir / 'no_numbers'
            no_numbers_dir.mkdir(exist_ok=True)
            directories_created.append('no_numbers')
        
        if directories_created:
            print(f"Created directories: {', '.join(directories_created)}")
        else:
            print("No directories needed - no detections found")
    
    def copy_images(self):
        """Copy images to appropriate directories"""
        print("Copying images to organized directories...")
        
        copy_stats = Counter()
        
        # Process each image
        for filename, detected_numbers in tqdm(self.image_detections.items(), desc="Copying images"):
            source_path = self.source_dir / filename
            
            # Check if source image exists
            if not source_path.exists():
                self.errors.append(f"Source image not found: {filename}")
                copy_stats['missing_source'] += 1
                continue
            
            try:
                if detected_numbers:
                    # Image has detections - copy to each detected number directory
                    for number in detected_numbers:
                        dest_dir = self.output_dir / str(number)
                        dest_path = dest_dir / filename
                        
                        # Copy the image
                        shutil.copy2(source_path, dest_path)
                        copy_stats[f'copied_to_{number}'] += 1
                        copy_stats['total_copies'] += 1
                    
                    copy_stats['images_with_detections'] += 1
                else:
                    # No detections - copy to no_numbers directory
                    dest_dir = self.output_dir / 'no_numbers'
                    dest_path = dest_dir / filename
                    
                    # Copy the image
                    shutil.copy2(source_path, dest_path)
                    copy_stats['copied_to_no_numbers'] += 1
                    copy_stats['total_copies'] += 1
                    copy_stats['images_without_detections'] += 1
            
            except Exception as e:
                error_msg = f"Failed to copy {filename}: {e}"
                self.errors.append(error_msg)
                copy_stats['copy_errors'] += 1
        
        self.copy_stats = copy_stats
        
        print(f"Copying completed!")
        print(f"Total copies made: {copy_stats.get('total_copies', 0)}")
        print(f"Images processed: {len(self.image_detections)}")
        
        if self.errors:
            print(f"Errors encountered: {len(self.errors)}")
            print("First few errors:")
            for error in self.errors[:5]:
                print(f"  - {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more errors")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*50)
        print("ORGANIZATION SUMMARY")
        print("="*50)
        
        # Overall statistics
        total_images = len(self.image_detections)
        images_with_detections = self.copy_stats.get('images_with_detections', 0)
        images_without_detections = self.copy_stats.get('images_without_detections', 0)
        total_copies = self.copy_stats.get('total_copies', 0)
        
        print(f"Total images processed: {total_images}")
        print(f"Images with detections: {images_with_detections}")
        print(f"Images without detections: {images_without_detections}")
        print(f"Total file copies made: {total_copies}")
        
        if self.errors:
            print(f"Errors encountered: {len(self.errors)}")
        
        print("\nBreakdown by detected numbers:")
        print("-" * 30)
        
        # Show breakdown by number
        for number in sorted(self.all_detected_numbers, key=int):
            count = self.copy_stats.get(f'copied_to_{number}', 0)
            print(f"  {number}: {count} images")
        
        # Show no_numbers count
        no_numbers_count = self.copy_stats.get('copied_to_no_numbers', 0)
        if no_numbers_count > 0:
            print(f"  no_numbers: {no_numbers_count} images")
        
        print("\nDirectory structure created:")
        print("-" * 30)
        print(f"Output directory: {self.output_dir}")
        
        # List created directories
        subdirs = []
        for number in sorted(self.all_detected_numbers, key=int):
            subdirs.append(f"  ├── {number}/")
        if no_numbers_count > 0:
            subdirs.append("  └── no_numbers/")
        
        if subdirs:
            for subdir in subdirs:
                print(subdir)
        else:
            print("  (No subdirectories created)")
        
        # Note about multiple detections
        multi_detection_images = [
            filename for filename, detections in self.image_detections.items()
            if len(detections) > 1
        ]
        
        if multi_detection_images:
            print(f"\nNote: {len(multi_detection_images)} images had multiple numbers detected")
            print("These images were copied to multiple directories:")
            for filename in multi_detection_images[:5]:  # Show first 5
                numbers = self.image_detections[filename]
                print(f"  - {filename}: detected {', '.join(numbers)}")
            if len(multi_detection_images) > 5:
                print(f"  ... and {len(multi_detection_images) - 5} more")
        
        print("\nOrganization complete!")
        
        if self.errors:
            print(f"\nWarning: {len(self.errors)} errors occurred during processing.")
            print("Check error messages above for details.")
    
    def organize(self):
        """Run the full organization process"""
        try:
            self.parse_results_file()
            self.create_directory_structure()
            self.copy_images()
            self.print_summary()
        except Exception as e:
            print(f"Error during organization: {e}")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Organize images by detected digits from OCR results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python organize_by_digits.py -s images/ -r results.txt
  
  # Specify custom output directory
  python organize_by_digits.py -s photos/ -r detection_results.txt -o organized_images/
  
  # Use default results file
  python organize_by_digits.py -s ./images/
        """
    )
    
    parser.add_argument(
        '-r', '--results-file',
        default='probably_not_pib_digits.txt',
        help='OCR results file (default: probably_not_pib_digits.txt)'
    )
    parser.add_argument(
        '-s', '--source-dir',
        required=True,
        help='Directory containing source images'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='predicted_digits',
        help='Output directory for organized images (default: predicted_digits)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.results_file).exists():
        print(f"Error: Results file '{args.results_file}' not found")
        sys.exit(1)
    
    if not Path(args.source_dir).exists():
        print(f"Error: Source directory '{args.source_dir}' not found")
        sys.exit(1)
    
    # Run the organizer
    organizer = ImageOrganizer(
        results_file=args.results_file,
        source_dir=args.source_dir,
        output_dir=args.output_dir
    )
    
    organizer.organize()


if __name__ == "__main__":
    main()