import easyocr
import os
from pathlib import Path

def test_easyocr_on_images():
    # Initialize EasyOCR reader (verbose=False to avoid progress bar issues)
    reader = easyocr.Reader(['en'], verbose=False)
    
    # Path to example images
    image_folder = Path("example_images")
    
    if not image_folder.exists():
        print(f"Image folder {image_folder} not found")
        return
    
    # Test each image
    for image_file in image_folder.glob("*.jpg"):
        print(f"\n--- Testing {image_file.name} ---")
        
        # Read text from image
        results = reader.readtext(str(image_file))
        
        if not results:
            print("No text detected")
            continue
            
        # Display results
        for i, (bbox, text, confidence) in enumerate(results):
            print(f"Text {i+1}: '{text}' (confidence: {confidence:.3f})")
            print(f"  Bounding box: {bbox}")
            
            # Check if detected text contains digits
            digits_only = ''.join(c for c in text if c.isdigit())
            if digits_only:
                print(f"  Extracted digits: {digits_only}")

if __name__ == "__main__":
    test_easyocr_on_images()