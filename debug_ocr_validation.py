"""
Debug OCR validation to understand why it's failing
"""

import cv2
import numpy as np
from template_matcher import TemplateMatcher
from pathlib import Path
import easyocr
import os

def debug_ocr_validation():
    """Debug OCR validation step by step"""
    
    matcher = TemplateMatcher()
    matcher.template_cache = {}
    
    # Test one specific case
    image_path = Path("example_images/84.jpg")
    expected_number = "84"
    
    print("=== OCR Validation Debug ===")
    print(f"Testing: {image_path} looking for '{expected_number}'")
    
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    print(f"Image loaded: {image.shape if image is not None else 'FAILED'}")
    
    # Get templates
    templates = matcher.get_templates(expected_number)
    print(f"Templates generated: {len(templates)}")
    
    # Run ensemble template matching (without OCR validation)
    ensemble_result = matcher._ensemble_template_matching(image, templates, expected_number)
    print(f"Template ensemble confidence: {ensemble_result['ensemble_confidence']:.3f}")
    print(f"Best template index: {ensemble_result['best_template_idx']}")
    
    # Check bounding box
    bbox = ensemble_result['bounding_box']
    if bbox:
        print(f"Bounding box: {bbox}")
        
        # Extract region manually
        padding = 10
        x1 = max(0, bbox['top_left'][0] - padding)
        y1 = max(0, bbox['top_left'][1] - padding) 
        x2 = min(image.shape[1], bbox['bottom_right'][0] + padding)
        y2 = min(image.shape[0], bbox['bottom_right'][1] + padding)
        
        roi = image[y1:y2, x1:x2]
        print(f"ROI shape: {roi.shape}")
        
        # Save ROI for visual inspection
        debug_dir = "debug_ocr"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/roi_84.png", roi)
        print(f"Saved ROI to {debug_dir}/roi_84.png")
        
        # Try OCR on the region
        print("\\nTrying OCR on extracted region...")
        try:
            reader = easyocr.Reader(['en'], verbose=False)
            ocr_results = reader.readtext(roi, detail=1)  # detail=1 for full info
            print(f"OCR results: {ocr_results}")
            
            # Try with different preprocessing
            # 1. Resize larger
            roi_large = cv2.resize(roi, (roi.shape[1]*3, roi.shape[0]*3), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{debug_dir}/roi_84_large.png", roi_large)
            ocr_results_large = reader.readtext(roi_large, detail=1)
            print(f"OCR results (3x larger): {ocr_results_large}")
            
            # 2. Try contrast enhancement
            roi_contrast = cv2.convertScaleAbs(roi, alpha=2.0, beta=0)
            cv2.imwrite(f"{debug_dir}/roi_84_contrast.png", roi_contrast)
            ocr_results_contrast = reader.readtext(roi_contrast, detail=1)
            print(f"OCR results (enhanced contrast): {ocr_results_contrast}")
            
        except Exception as e:
            print(f"OCR error: {e}")
    else:
        print("No bounding box found!")
    
    # Also test OCR on the full image for comparison
    print("\\nTesting OCR on full image...")
    try:
        reader = easyocr.Reader(['en'], verbose=False)
        full_ocr = reader.readtext(image, detail=1)
        print(f"Full image OCR: {full_ocr}")
    except Exception as e:
        print(f"Full image OCR error: {e}")


if __name__ == "__main__":
    debug_ocr_validation()