"""
Shipment Validation Script

Validates shipments by checking if the address number appears in any of the shipment's images.
Creates 'valid_shipments' and 'invalid_shipments' directories with the organized results.
"""

import pandas as pd
import os
import shutil
import re
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

# Import the building number detector
from building_number_detector import BuildingNumberDetector


class ShipmentValidator:
    """Validates shipments by checking address numbers against OCR results"""
    
    def __init__(self, 
                 dataframe_path: str = "pod_df.pkl",
                 images_dir: str = "daily_images",
                 output_dir: str = "shipment_validation"):
        """
        Initialize the validator
        
        Args:
            dataframe_path: Path to the pickled DataFrame
            images_dir: Directory containing shipment image directories  
            output_dir: Base output directory for results
        """
        self.dataframe_path = Path(dataframe_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        
        # Output directories
        self.valid_dir = self.output_dir / "valid_shipments"
        self.invalid_dir = self.output_dir / "invalid_shipments"
        
        # Storage for results
        self.df = None
        self.shipment_data = {}  # {shipment_uid: {'address_number': str, 'images': [str]}}
        self.validation_results = {}  # {shipment_uid: str}
        self.errors = []
        
        # Initialize OCR detector
        print("Initializing EasyOCR...")
        self.detector = BuildingNumberDetector(verbose=False)
        
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
    
    def extract_address_number(self, address_line_1: str, address_line_2: str = None) -> Optional[str]:
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
        if address_line_1 and str(address_line_1).strip() != 'nan':
            address_text = str(address_line_1).strip()
        
        if address_line_2 and str(address_line_2).strip() not in ['nan', '', '.', 'None']:
            address_text += " " + str(address_line_2).strip()
        
        if not address_text:
            return None
        
        # Find the first number in the address
        match = re.search(r'\d+', address_text)
        return match.group() if match else None
    
    def process_shipments(self):
        """Process DataFrame to extract shipment data"""
        print("Processing shipments from DataFrame...")
        
        # Group by shipment_uid
        shipment_groups = self.df.groupby('shipment_uid')
        
        for shipment_uid, group in tqdm(shipment_groups, desc="Processing shipments"):
            # Get address information (should be same for all rows in group)
            first_row = group.iloc[0]
            address_number = self.extract_address_number(
                first_row['address_line_1'], 
                first_row['address_line_2']
            )
            
            # Get list of images for this shipment
            images = group['asset_name'].tolist()
            
            self.shipment_data[shipment_uid] = {
                'address_number': address_number,
                'images': images,
                'address_line_1': first_row['address_line_1'],
                'address_line_2': first_row['address_line_2']
            }
        
        # Filter out shipments without address numbers
        shipments_without_numbers = [
            uid for uid, data in self.shipment_data.items() 
            if not data['address_number']
        ]
        
        if shipments_without_numbers:
            print(f"Warning: {len(shipments_without_numbers)} shipments have no address number")
            # Remove them from processing
            for uid in shipments_without_numbers:
                del self.shipment_data[uid]
        
        print(f"Found {len(self.shipment_data)} shipments with address numbers to validate")
        
        # Show some statistics
        address_numbers = [data['address_number'] for data in self.shipment_data.values()]
        number_counts = Counter(address_numbers)
        print(f"Most common address numbers: {number_counts.most_common(10)}")
    
    def validate_shipment(self, shipment_uid: str, shipment_data: Dict) -> bool:
        """
        Validate a single shipment by checking if address number appears in images
        
        Args:
            shipment_uid: The shipment UUID
            shipment_data: Dictionary containing address_number and images list
            
        Returns:
            True if address number found in at least one image, False otherwise
        """
        address_number = shipment_data['address_number']
        images = shipment_data['images']
        
        shipment_image_dir = self.images_dir / shipment_uid
        
        # Check if shipment directory exists
        if not shipment_image_dir.exists():
            self.errors.append(f"Shipment directory not found: {shipment_uid}")
            return None
        
        # Check each image in the shipment
        for image_name in images:
            image_path = shipment_image_dir / image_name
            
            if not image_path.exists():
                # Log missing image but continue with other images
                continue
            
            try:
                # Use the check_for_expected_number method for targeted detection
                result = self.detector.check_for_expected_number(
                    str(image_path),
                    address_number,
                    fuzzy_threshold=0.7,  # Slightly lower threshold for more flexibility
                    template_threshold=0.6
                )
                
                # If number found with reasonable confidence, shipment is valid
                if result['found'] and result['confidence'] > 0.7:
                    return image_name
                    
            except Exception as e:
                error_msg = f"Error processing {image_path}: {e}"
                self.errors.append(error_msg)
                continue
        
        # No valid detection found in any image
        return None
    
    def validate_all_shipments(self):
        """Validate all shipments"""
        print(f"Validating {len(self.shipment_data)} shipments...")
        
        valid_count = 0
        invalid_count = 0
        
        for shipment_uid, shipment_data in tqdm(self.shipment_data.items(), desc="Validating shipments"):
            valid_image_name = self.validate_shipment(shipment_uid, shipment_data)
            self.validation_results[shipment_uid] = valid_image_name
            
            if valid_image_name is not None:
                valid_count += 1
            else:
                invalid_count += 1
        
        print(f"Validation complete:")
        print(f"  Valid shipments: {valid_count}")
        print(f"  Invalid shipments: {invalid_count}")
        print(f"  Total processed: {len(self.validation_results)}")
        
        if self.errors:
            print(f"  Errors encountered: {len(self.errors)}")
    
    def create_output_directories(self):
        """Create output directory structure"""
        print(f"Creating output directories...")
        
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.valid_dir.mkdir(exist_ok=True)
        self.invalid_dir.mkdir(exist_ok=True)
        
        print(f"Created directories:")
        print(f"  {self.valid_dir}")
        print(f"  {self.invalid_dir}")
    
    def copy_shipment_directories(self):
        """Copy shipment directories to appropriate output directories"""
        print("Copying shipment directories...")
        
        copy_stats = Counter()
        
        for shipment_uid, valid_image_name in tqdm(self.validation_results.items(), desc="Copying directories"):
            source_dir = self.images_dir / shipment_uid
            
            # Determine destination
            if valid_image_name is not None:
                dest_dir = self.valid_dir / shipment_uid
                copy_stats['valid'] += 1
            else:
                dest_dir = self.invalid_dir / shipment_uid
                copy_stats['invalid'] += 1
            
            # Check if source exists
            if not source_dir.exists():
                self.errors.append(f"Source directory not found for copying: {shipment_uid}")
                copy_stats['missing_source'] += 1
                continue
            
            try:
                # Copy entire directory
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(source_dir, dest_dir)
                copy_stats['copied'] += 1
                
            except Exception as e:
                error_msg = f"Failed to copy {shipment_uid}: {e}"
                self.errors.append(error_msg)
                copy_stats['copy_errors'] += 1
        
        print(f"Directory copying completed:")
        print(f"  Valid shipments copied: {copy_stats['valid']}")
        print(f"  Invalid shipments copied: {copy_stats['invalid']}")
        print(f"  Total copied: {copy_stats['copied']}")
        
        if copy_stats['copy_errors'] > 0:
            print(f"  Copy errors: {copy_stats['copy_errors']}")
        if copy_stats['missing_source'] > 0:
            print(f"  Missing source directories: {copy_stats['missing_source']}")
    
    def save_results_summary(self):
        """Save detailed results to a file"""
        results_file = self.output_dir / "validation_results.txt"
        
        print(f"Saving results summary to {results_file}...")
        
        with open(results_file, 'w') as f:
            f.write("Shipment Validation Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            valid_count = sum(1 for result in self.validation_results.values() if result)
            invalid_count = len(self.validation_results) - valid_count
            
            f.write(f"Total shipments processed: {len(self.validation_results)}\n")
            f.write(f"Valid shipments: {valid_count}\n")
            f.write(f"Invalid shipments: {invalid_count}\n")
            f.write(f"Success rate: {valid_count/len(self.validation_results)*100:.1f}%\n\n")
            
            if self.errors:
                f.write(f"Errors encountered: {len(self.errors)}\n\n")
            
            # Detailed results
            f.write("Detailed Results:\n")
            f.write("-" * 30 + "\n")
            
            for shipment_uid, valid_image_name in self.validation_results.items():
                shipment_data = self.shipment_data[shipment_uid]
                status = "VALID" if valid_image_name is not None else "INVALID"
                address_num = shipment_data['address_number']
                address_line_1 = shipment_data['address_line_1']
                num_images = len(shipment_data['images'])
                valid_image_name = valid_image_name if valid_image_name is not None else "None"
                f.write(f"{shipment_uid}: {status}\n")
                f.write(f"  Address number: {address_num}\n")
                f.write(f"  Address: {address_line_1}\n")
                f.write(f"  Valid image: {valid_image_name}\n")
                f.write(f"  Images: {num_images}\n\n")

            
            # Errors
            if self.errors:
                f.write("Errors:\n")
                f.write("-" * 30 + "\n")
                for error in self.errors:
                    f.write(f"- {error}\n")
        
        print(f"Results summary saved to {results_file}")
    
    def run_validation(self):
        """Run the complete validation process"""
        try:
            # Load data
            self.load_dataframe()
            self.process_shipments()
            
            # Validate shipments
            self.validate_all_shipments()
            
            # Create outputs
            self.create_output_directories()
            self.copy_shipment_directories()
            self.save_results_summary()
            
            print("\nValidation completed successfully!")
            
        except Exception as e:
            print(f"Error during validation: {e}")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate shipments by checking if address numbers appear in images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python validate_shipments.py
  
  # Specify custom paths
  python validate_shipments.py -d custom_data.pkl -i custom_images/ -o custom_output/
        """
    )
    
    parser.add_argument(
        '-d', '--dataframe',
        default='pod_df.pkl',
        help='Path to pickled DataFrame (default: pod_df.pkl)'
    )
    parser.add_argument(
        '-i', '--images-dir',
        default='daily_images',
        help='Directory containing shipment image directories (default: daily_images)'
    )
    parser.add_argument(
        '-o', '--output-dir', 
        default='shipment_validation',
        help='Output directory for validation results (default: shipment_validation)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.dataframe).exists():
        print(f"Error: DataFrame file '{args.dataframe}' not found")
        sys.exit(1)
    
    if not Path(args.images_dir).exists():
        print(f"Error: Images directory '{args.images_dir}' not found")
        sys.exit(1)
    
    # Run validation
    validator = ShipmentValidator(
        dataframe_path=args.dataframe,
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )
    
    validator.run_validation()


if __name__ == "__main__":
    main()