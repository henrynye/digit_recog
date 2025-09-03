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


def main():
    """Example usage"""
    # Initialize detector
    detector = BuildingNumberDetector()
    
    # Test on example images
    if Path("example_images").exists():
        print("Testing building number detection on example images...")
        results = detector.process_batch("example_images", "detection_results.txt")
        
        print(f"\nProcessed {len(results)} images")
        print("Results saved to detection_results.txt")
        
        # Show summary
        for filename, detections in results.items():
            numbers = [det['digits'] for det in detections if det['digits']]
            if numbers:
                print(f"{filename}: {', '.join(numbers)}")
            else:
                print(f"{filename}: No numbers detected")
    else:
        print("No example_images folder found")
        
    # Example of single image processing
    # numbers = detector.get_building_numbers("path/to/building.jpg")
    # print(f"Detected building numbers: {numbers}")


if __name__ == "__main__":
    main()