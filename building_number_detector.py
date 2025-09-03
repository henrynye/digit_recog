"""
Building Number Detection using EasyOCR

A simple, production-ready solution for detecting numbers on building facades.
Based on evaluation showing EasyOCR's 83% success rate on building images.
"""

import easyocr
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import re
import argparse
import sys


class BuildingNumberDetector:
    """Detects building numbers in images using EasyOCR"""
    
    def __init__(self, languages=['en'], verbose=False):
        """
        Initialize the detector
        
        Args:
            languages: List of language codes (default: ['en'])
            verbose: Enable verbose output during model loading
        """
        self.reader = easyocr.Reader(languages, verbose=verbose)
        
    def detect_numbers(self, image_path: str, 
                      min_confidence: float = 0.5,
                      digits_only: bool = True) -> List[dict]:
        """
        Detect building numbers in an image
        
        Args:
            image_path: Path to image file
            min_confidence: Minimum confidence threshold (0.0-1.0)
            digits_only: If True, only return results containing digits
            
        Returns:
            List of detection results with format:
            [
                {
                    'text': '123',
                    'confidence': 0.95,
                    'bbox': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                    'digits': '123'
                }
            ]
        """
        try:
            # Perform OCR
            results = self.reader.readtext(str(image_path))
            
            # Process results
            detections = []
            for bbox, text, confidence in results:
                # Apply confidence threshold
                if confidence < min_confidence:
                    continue
                
                # Extract digits from text
                digits = re.sub(r'[^0-9]', '', text)
                
                # Filter for digits if requested
                if digits_only and not digits:
                    continue
                
                detection = {
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'digits': digits
                }
                detections.append(detection)
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    def get_building_numbers(self, image_path: str, 
                           min_confidence: float = 0.8) -> List[str]:
        """
        Simple interface to get just the building numbers as strings
        
        Args:
            image_path: Path to image file
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected building numbers as strings
        """
        detections = self.detect_numbers(image_path, min_confidence, digits_only=True)
        return [det['digits'] for det in detections if det['digits']]
    
    def process_batch(self, image_folder: str, 
                     output_file: Optional[str] = None) -> dict:
        """
        Process multiple images in a folder
        
        Args:
            image_folder: Path to folder containing images
            output_file: Optional path to save results as text file
            
        Returns:
            Dictionary mapping image filenames to detection results
        """
        folder_path = Path(image_folder)
        results = {}
        
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        for ext in extensions:
            for image_path in folder_path.glob(ext):
                print(f"Processing {image_path.name}...")
                detections = self.detect_numbers(str(image_path))
                results[image_path.name] = detections
        
        # Save results to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Building Number Detection Results\n")
                f.write("=" * 40 + "\n\n")
                
                for filename, detections in results.items():
                    f.write(f"{filename}:\n")
                    if detections:
                        for det in detections:
                            f.write(f"  Number: {det['digits']} "
                                   f"(confidence: {det['confidence']:.1%})\n")
                    else:
                        f.write("  No numbers detected\n")
                    f.write("\n")
        
        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Detect building numbers in images using EasyOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific images
  python building_number_detector.py -i image1.jpg image2.jpg image3.jpg
  
  # Process a folder with custom confidence threshold
  python building_number_detector.py -f example_images --confidence 0.9
  
  # Save detailed results to a file
  python building_number_detector.py -i house1.jpg -o results.txt --detailed
  
  # Run with verbose output and custom languages
  python building_number_detector.py -f photos/ --verbose --languages en es
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        '-i', '--images', 
        nargs='+',
        help='Specific image files to process'
    )
    input_group.add_argument(
        '-f', '--folder',
        help='Folder containing images to process (default: example_images)'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        help='Output file to save results (default: print to console)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed results including confidence scores and bounding boxes'
    )
    
    # Detection parameters
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.8,
        help='Minimum confidence threshold (0.0-1.0, default: 0.8)'
    )
    parser.add_argument(
        '--digits-only',
        action='store_true',
        default=True,
        help='Only return results containing digits (default: True)'
    )
    parser.add_argument(
        '--include-text',
        action='store_true',
        help='Include non-numeric text in results'
    )
    
    # Model parameters
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['en'],
        help='Language codes for OCR (default: en)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output during model loading'
    )
    
    return parser.parse_args()


def process_single_images(detector, image_paths, args):
    """Process individual image files"""
    results = {}
    
    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists():
            print(f"Warning: {image_path} not found")
            continue
            
        print(f"Processing {path.name}...")
        
        # Get detection results
        detections = detector.detect_numbers(
            str(path), 
            min_confidence=args.confidence,
            digits_only=not args.include_text
        )
        
        results[path.name] = detections
    
    return results


def display_results(results, args):
    """Display or save results based on arguments"""
    output_lines = []
    
    if args.detailed:
        output_lines.append("Building Number Detection Results (Detailed)")
        output_lines.append("=" * 50)
        output_lines.append("")
        
        for filename, detections in results.items():
            output_lines.append(f"{filename}:")
            
            if detections:
                for i, det in enumerate(detections, 1):
                    output_lines.append(f"  Detection {i}:")
                    output_lines.append(f"    Text: '{det['text']}'")
                    output_lines.append(f"    Digits: {det['digits'] or 'None'}")
                    output_lines.append(f"    Confidence: {det['confidence']:.1%}")
                    if 'bbox' in det:
                        bbox = det['bbox']
                        output_lines.append(f"    Bounding box: {bbox}")
                    output_lines.append("")
            else:
                output_lines.append("  No detections found")
                output_lines.append("")
    else:
        # Simple summary format
        output_lines.append("Building Number Detection Results")
        output_lines.append("=" * 40)
        output_lines.append("")
        
        for filename, detections in results.items():
            numbers = [det['digits'] for det in detections if det['digits']]
            if numbers:
                confidence_info = ""
                if len(detections) == 1:
                    confidence_info = f" (confidence: {detections[0]['confidence']:.1%})"
                output_lines.append(f"{filename}: {', '.join(numbers)}{confidence_info}")
            else:
                output_lines.append(f"{filename}: No numbers detected")
    
    # Output results
    output_text = "\n".join(output_lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Results saved to {args.output}")
    else:
        print(output_text)


def main():
    """Main CLI entry point"""
    args = parse_args()
    
    # Handle confidence validation
    if not (0.0 <= args.confidence <= 1.0):
        print("Error: Confidence must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Initialize detector
    print("Initializing EasyOCR...")
    detector = BuildingNumberDetector(
        languages=args.languages,
        verbose=args.verbose
    )
    
    # Determine what to process
    if args.images:
        # Process specific images
        print(f"Processing {len(args.images)} specific images...")
        results = process_single_images(detector, args.images, args)
        
    else:
        # Process folder
        folder = args.folder if args.folder else "example_images"
        folder_path = Path(folder)
        
        if not folder_path.exists():
            print(f"Error: Folder '{folder}' not found")
            sys.exit(1)
        
        print(f"Processing images in folder: {folder}")
        results = detector.process_batch(folder, None)  # Don't auto-save, we'll handle output
    
    # Display results
    if results:
        print(f"\nProcessed {len(results)} images")
        display_results(results, args)
    else:
        print("No images were processed successfully")


if __name__ == "__main__":
    main()