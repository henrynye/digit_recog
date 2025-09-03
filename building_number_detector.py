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

# Import our custom modules
from fuzzy_matcher import FuzzyMatcher
from template_matcher import TemplateMatcher


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
        self.fuzzy_matcher = FuzzyMatcher(enable_ocr_corrections=True)
        self.template_matcher = TemplateMatcher()
        
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
    
    def check_for_expected_number(self, image_path: str, expected_number: str,
                                 fuzzy_threshold: float = 0.8,
                                 template_threshold: float = 0.7,
                                 use_template_matching: bool = True) -> dict:
        """
        Check if an expected number exists in the image with confidence scoring
        
        This method combines OCR with fuzzy matching and optional template matching
        to provide a confidence score for whether a specific expected number
        appears somewhere in the image.
        
        Args:
            image_path: Path to image file
            expected_number: The specific number to search for (e.g., "123", "42")
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0.0-1.0)
            template_threshold: Minimum confidence for template matching (0.0-1.0)
            use_template_matching: Whether to use template matching as fallback
            
        Returns:
            Dictionary with detection results:
            {
                'found': bool,              # Whether expected number was found
                'confidence': float,        # Overall confidence score (0.0-1.0)
                'method': str,              # Method that found the match
                'ocr_result': dict,         # Results from OCR + fuzzy matching
                'template_result': dict,    # Results from template matching (if used)
                'best_match': str,          # Text of best match found
                'location': tuple,          # Location of match (x, y) or None
                'details': dict            # Additional details about the detection
            }
        """
        result = {
            'found': False,
            'confidence': 0.0,
            'method': 'none',
            'ocr_result': None,
            'template_result': None,
            'best_match': None,
            'location': None,
            'details': {}
        }
        
        try:
            # Step 1: Try OCR with fuzzy matching
            # Use allowlist to restrict to digits only for better accuracy
            ocr_results = self.reader.readtext(str(image_path), allowlist='0123456789')
            
            # Extract just the text from OCR results
            detected_texts = [text for bbox, text, conf in ocr_results if conf > 0.3]
            
            # Use fuzzy matcher to check for expected number
            fuzzy_result = self.fuzzy_matcher.check_expected_number(
                expected_number, detected_texts, threshold=fuzzy_threshold
            )
            
            result['ocr_result'] = fuzzy_result
            
            # If we found a good match with OCR + fuzzy matching
            if fuzzy_result['found']:
                result.update({
                    'found': True,
                    'confidence': fuzzy_result['similarity'],
                    'method': fuzzy_result['method'],
                    'best_match': fuzzy_result['best_match']
                })
                
                # Find location of the matched text in original OCR results
                for bbox, text, conf in ocr_results:
                    if text == fuzzy_result['best_match']:
                        # Get center point of bounding box
                        center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                        center_y = int((bbox[0][1] + bbox[2][1]) / 2)
                        result['location'] = (center_x, center_y)
                        break
            
            # Step 2: If OCR didn't find a good match and template matching is enabled
            if (not result['found'] or result['confidence'] < 0.7) and use_template_matching:
                template_result = self.template_matcher.check_for_number(
                    image_path, expected_number, threshold=template_threshold
                )
                
                result['template_result'] = template_result
                
                # If template matching found a better match
                if template_result['found'] and template_result['confidence'] > result['confidence']:
                    result.update({
                        'found': True,
                        'confidence': template_result['confidence'],
                        'method': 'template_matching',
                        'best_match': expected_number,
                        'location': template_result['location']
                    })
            
            # Step 3: Combine results for final confidence if both methods were used
            if result['ocr_result'] and result['template_result']:
                ocr_conf = result['ocr_result']['similarity'] if result['ocr_result']['found'] else 0.0
                template_conf = result['template_result']['confidence'] if result['template_result']['found'] else 0.0
                
                # Use weighted average (OCR weighted higher as it's more specific)
                combined_confidence = (0.7 * ocr_conf + 0.3 * template_conf)
                
                if combined_confidence > result['confidence']:
                    result['confidence'] = combined_confidence
                    result['method'] = 'combined'
            
            # Add additional details
            result['details'] = {
                'expected_number': expected_number,
                'fuzzy_threshold': fuzzy_threshold,
                'template_threshold': template_threshold,
                'use_template_matching': use_template_matching,
                'ocr_detections_count': len(detected_texts),
                'detected_texts': detected_texts[:5]  # Limit to first 5 for brevity
            }
            
            return result
            
        except Exception as e:
            result['details']['error'] = str(e)
            return result
    
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
  
  # Check for a specific expected number
  python building_number_detector.py -i house.jpg --expected-number 123
  
  # Check for expected number with custom thresholds
  python building_number_detector.py -i house.jpg --expected-number 42 --fuzzy-threshold 0.9 --template-threshold 0.8
  
  # Disable template matching (OCR + fuzzy matching only)
  python building_number_detector.py -i house.jpg --expected-number 999 --disable-template-matching
  
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
    
    # Expected number detection parameters
    parser.add_argument(
        '--expected-number',
        help='Specific number to search for and return confidence score'
    )
    parser.add_argument(
        '--fuzzy-threshold',
        type=float,
        default=0.8,
        help='Minimum similarity score for fuzzy matching (0.0-1.0, default: 0.8)'
    )
    parser.add_argument(
        '--template-threshold',
        type=float,
        default=0.7,
        help='Minimum confidence for template matching (0.0-1.0, default: 0.7)'
    )
    parser.add_argument(
        '--disable-template-matching',
        action='store_true',
        help='Disable template matching fallback'
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
        
        # Check if we're looking for a specific expected number
        if args.expected_number:
            # Use expected number detection
            expected_result = detector.check_for_expected_number(
                str(path),
                args.expected_number,
                fuzzy_threshold=args.fuzzy_threshold,
                template_threshold=args.template_threshold,
                use_template_matching=not args.disable_template_matching
            )
            results[path.name] = expected_result
        else:
            # Use regular number detection
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
    
    # Check if we're displaying expected number results
    is_expected_number_mode = args.expected_number is not None
    
    if is_expected_number_mode:
        # Expected number detection results
        if args.detailed:
            output_lines.append(f"Expected Number Detection Results (Detailed) - Looking for: '{args.expected_number}'")
            output_lines.append("=" * 70)
            output_lines.append("")
            
            for filename, result in results.items():
                output_lines.append(f"{filename}:")
                output_lines.append(f"  Expected: '{args.expected_number}'")
                output_lines.append(f"  Found: {result['found']}")
                output_lines.append(f"  Confidence: {result['confidence']:.1%}")
                output_lines.append(f"  Method: {result['method']}")
                
                if result['best_match']:
                    output_lines.append(f"  Best match: '{result['best_match']}'")
                
                if result['location']:
                    output_lines.append(f"  Location: {result['location']}")
                
                if result.get('details', {}).get('detected_texts'):
                    texts = result['details']['detected_texts']
                    output_lines.append(f"  All detected texts: {texts}")
                
                if 'error' in result.get('details', {}):
                    output_lines.append(f"  Error: {result['details']['error']}")
                
                output_lines.append("")
        else:
            # Simple expected number results
            output_lines.append(f"Expected Number Detection Results - Looking for: '{args.expected_number}'")
            output_lines.append("=" * 60)
            output_lines.append("")
            
            for filename, result in results.items():
                status = "FOUND" if result['found'] else "NOT FOUND"
                confidence_str = f"({result['confidence']:.1%})" if result['found'] else ""
                method_str = f"via {result['method']}" if result['found'] and result['method'] != 'none' else ""
                
                output_lines.append(f"{filename}: {status} {confidence_str} {method_str}".strip())
    
    else:
        # Regular detection results
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
    
    # Handle expected number threshold validation
    if not (0.0 <= args.fuzzy_threshold <= 1.0):
        print("Error: Fuzzy threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not (0.0 <= args.template_threshold <= 1.0):
        print("Error: Template threshold must be between 0.0 and 1.0")
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
        
        # If we're looking for an expected number, we need to process each image individually
        if args.expected_number:
            # Get all image files in the folder
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            for ext in extensions:
                image_files.extend(folder_path.glob(ext))
            
            # Convert to strings for processing
            image_paths = [str(img) for img in image_files]
            results = process_single_images(detector, image_paths, args)
        else:
            # Use regular batch processing
            results = detector.process_batch(folder, None)  # Don't auto-save, we'll handle output
    
    # Display results
    if results:
        print(f"\nProcessed {len(results)} images")
        display_results(results, args)
    else:
        print("No images were processed successfully")


if __name__ == "__main__":
    main()