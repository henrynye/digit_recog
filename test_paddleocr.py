from paddleocr import PaddleOCR
import os
from pathlib import Path

def test_paddleocr_on_images():
    # Initialize PaddleOCR reader (English only)
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    # Path to example images
    image_folder = Path("example_images")
    
    if not image_folder.exists():
        print(f"Image folder {image_folder} not found")
        return
    
    # Test each image
    for image_file in image_folder.glob("*.jpg"):
        print(f"\n--- Testing {image_file.name} ---")
        
        # Read text from image
        results = ocr.predict(str(image_file))
        
        if not results or not results[0]:
            print("No text detected")
            continue
            
        # PaddleOCR returns a dictionary with various fields
        result_dict = results[0]
        
        # Extract text and confidence scores
        if 'rec_texts' in result_dict and 'rec_scores' in result_dict:
            texts = result_dict['rec_texts']
            scores = result_dict['rec_scores']
            
            if texts:
                for i, (text, score) in enumerate(zip(texts, scores)):
                    print(f"Text {i+1}: '{text}' (confidence: {score:.3f})")
                    
                    # Check if detected text contains digits
                    digits_only = ''.join(c for c in text if c.isdigit())
                    if digits_only:
                        print(f"  Extracted digits: {digits_only}")
            else:
                print("No text detected")
        else:
            print("Unexpected result format. Available keys:")
            for key in result_dict.keys():
                print(f"  {key}")

if __name__ == "__main__":
    test_paddleocr_on_images()