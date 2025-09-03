"""
Debug script for template matching false positives
"""

import cv2
import numpy as np
from template_matcher import TemplateMatcher
import os
from pathlib import Path

def test_template_matching():
    """Test template matching with different scenarios"""
    matcher = TemplateMatcher()
    test_images = Path("example_images")
    
    # Test cases: (image_file, actual_number, test_number, should_match)
    test_cases = [
        ("84.jpg", "84", "84", True),      # Should match
        ("84.jpg", "84", "20", False),     # Should NOT match
        ("84.jpg", "84", "64", False),     # Should NOT match 
        ("40.jpg", "40", "40", True),      # Should match
        ("40.jpg", "40", "84", False),     # Should NOT match
        ("68_64.jpg", "64", "64", True),   # Should match
        ("68_64.jpg", "64", "20", False),  # Should NOT match
        ("20.jpg", "20", "20", True),      # Should match
        ("20.jpg", "20", "40", False),     # Should NOT match
    ]
    
    print("Template Matching Debug Test")
    print("=" * 60)
    
    for image_file, actual_number, test_number, should_match in test_cases:
        image_path = test_images / image_file
        if not image_path.exists():
            print(f"Skipping {image_file} - file not found")
            continue
            
        # Run template matching
        result = matcher.check_for_number(str(image_path), test_number, threshold=0.7)
        
        # Check if result is as expected
        match_status = "PASS" if result['found'] == should_match else "FAIL"
        expected_text = "MATCH" if should_match else "NO MATCH"
        actual_text = "MATCH" if result['found'] else "NO MATCH"
        
        print(f"\n[{match_status}] Testing {image_file}: Looking for '{test_number}' (actual: '{actual_number}')")
        print(f"  Expected: {expected_text}, Got: {actual_text}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Template index: {result.get('template_index', -1)}")
        
        # If this is a false positive (matched when it shouldn't)
        if result['found'] and not should_match:
            print(f"  WARNING: FALSE POSITIVE - Found '{test_number}' but actual is '{actual_number}'")
            
            # Debug: Show template matching scores for all methods
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            templates = matcher.get_templates(test_number)
            
            print(f"  Detailed scores for {len(templates)} templates:")
            for i, template in enumerate(templates):
                # Try different matching methods
                methods = {
                    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
                    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
                    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
                }
                
                for method_name, method in methods.items():
                    match_res = matcher.match_template_single(image, template, method)
                    conf = match_res['confidence']
                    if conf > 0.5:  # Only show significant matches
                        print(f"    Template {i} with {method_name}: {conf:.3f}")


def analyze_matching_algorithm():
    """Analyze why template matching gives false positives"""
    print("\n" + "=" * 60)
    print("ANALYSIS OF TEMPLATE MATCHING ALGORITHM")
    print("=" * 60)
    
    # Load a test image
    image_path = "example_images/84.jpg"
    if not os.path.exists(image_path):
        print(f"Cannot find {image_path}")
        return
        
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(f"\nImage shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image value range: {image.min()} - {image.max()}")
    
    # Generate templates
    matcher = TemplateMatcher()
    
    # Test with correct and incorrect numbers
    test_numbers = ["84", "20", "64"]
    
    for number in test_numbers:
        templates = matcher.get_templates(number)
        print(f"\nTesting number '{number}' on image with actual '84':")
        
        # Analyze first template
        template = templates[0]
        print(f"  Template shape: {template.shape}")
        print(f"  Template value range: {template.min()} - {template.max()}")
        
        # Get the matching result
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"  Raw match score: {max_val:.3f}")
        print(f"  Match location: {max_loc}")
        
        # Show why this might be problematic
        if max_val > 0.7 and number != "84":
            print(f"  WARNING: PROBLEM - High score for wrong number!")
            
            # Calculate normalized cross-correlation manually for insight
            # This helps understand what's being matched
            h, w = template.shape
            roi = image[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]
            
            if roi.shape == template.shape:
                # Calculate similarity metrics
                mse = np.mean((roi - template) ** 2)
                ssim_approx = np.corrcoef(roi.flatten(), template.flatten())[0, 1]
                
                print(f"    ROI vs Template MSE: {mse:.1f}")
                print(f"    ROI vs Template correlation: {ssim_approx:.3f}")
                
                # Save the ROI and template for visual inspection
                debug_dir = "template_debug"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/roi_{number}.png", roi)
                cv2.imwrite(f"{debug_dir}/template_{number}.png", template)
                print(f"    Saved ROI and template to {debug_dir}/")


if __name__ == "__main__":
    # Run the tests
    test_template_matching()
    analyze_matching_algorithm()
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("Template matching can give false positives because:")
    print("1. It looks for similar patterns, not exact matches")
    print("2. Numbers with similar shapes (8 and 0, 6 and 8) can match highly")
    print("3. The correlation-based methods are sensitive to lighting/contrast")
    print("4. Small template sizes may not capture enough distinctive features")