# Building Number Detection

A machine learning solution for detecting numbers on building facades using computer vision and OCR.

## 🎯 Solution Overview

After testing multiple out-of-the-box OCR solutions, **EasyOCR** was selected as the best option for building number detection with **83% accuracy** on test images.

## 📊 Evaluation Results

### Models Tested:
- ✅ **EasyOCR** - 83% success rate (Winner)
- ❌ **PaddleOCR** - 67% success rate
- 📚 **Research completed on**: YOLO models, Cloud APIs (Google Vision, AWS Textract, Azure OCR), TrOCR

### Test Results:
- **20.jpg**: ✅ Detected "20" (94.8% confidence)
- **34.jpg**: ❌ No detection
- **36.jpg**: ✅ Detected "36" (100% confidence)  
- **40.jpg**: ✅ Detected "40" twice (100% confidence each)
- **68_64.jpg**: ✅ Detected "68", "64", "62" (99.9-100% confidence)
- **84.jpg**: ✅ Detected "84" (100% confidence)

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies (already done)
pip install easyocr
```

### 2. Command Line Usage

#### Process Specific Images
```bash
# Process individual images
python building_number_detector.py -i house1.jpg house2.jpg house3.jpg

# Process with custom confidence threshold
python building_number_detector.py -i building.jpg --confidence 0.9

# Save detailed results to file
python building_number_detector.py -i house.jpg --detailed -o results.txt
```

#### Expected Number Detection ⭐ New Feature
```bash
# Check if a specific number exists in an image
python building_number_detector.py -i house.jpg --expected-number 123

# Check with detailed output showing confidence and method used
python building_number_detector.py -i house.jpg --expected-number 42 --detailed

# Batch process looking for a specific number
python building_number_detector.py -f photos/ --expected-number 256

# Adjust detection sensitivity
python building_number_detector.py -i house.jpg --expected-number 99 --fuzzy-threshold 0.9

# Use OCR-only mode (disable template matching)
python building_number_detector.py -i house.jpg --expected-number 77 --disable-template-matching
```

#### Process Image Folders
```bash
# Process all images in a folder
python building_number_detector.py -f example_images

# Process folder with high confidence threshold
python building_number_detector.py -f photos/ --confidence 0.95

# Include non-numeric text in results
python building_number_detector.py -f images/ --include-text
```

#### Get Help
```bash
python building_number_detector.py --help
```

### 3. Python API Usage

#### Regular Number Detection
```python
from building_number_detector import BuildingNumberDetector

# Initialize detector
detector = BuildingNumberDetector()

# Detect numbers in a single image
numbers = detector.get_building_numbers("path/to/building.jpg")
print(f"Building numbers: {numbers}")

# Get detailed results with confidence scores
detections = detector.detect_numbers("path/to/building.jpg")
for det in detections:
    print(f"Number: {det['digits']} (confidence: {det['confidence']:.1%})")
```

#### Expected Number Detection ⭐ New
```python
from building_number_detector import BuildingNumberDetector

# Initialize detector
detector = BuildingNumberDetector()

# Check if a specific number exists in the image
result = detector.check_for_expected_number("house.jpg", "123")

print(f"Expected number '123' found: {result['found']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Detection method: {result['method']}")

if result['found']:
    print(f"Best match: '{result['best_match']}'")
    print(f"Location: {result['location']}")

# Use with custom thresholds
result = detector.check_for_expected_number(
    "house.jpg", "456", 
    fuzzy_threshold=0.9, 
    template_threshold=0.8,
    use_template_matching=False  # OCR-only mode
)
```

### 4. Batch Processing
```python
# Process all images in a folder
results = detector.process_batch("images/", "results.txt")
```

## 📁 Project Structure

```
digit_recog/
├── venv/                          # Virtual environment
├── example_images/                # Test images (6 building photos)
├── building_number_detector.py    # Main detection class
├── ocr_comparison.md             # Detailed comparison results
├── detection_results.txt         # Test results
├── test_easyocr.py              # EasyOCR test script
├── test_paddleocr.py            # PaddleOCR test script
└── README.md                    # This file
```

## 🛠 Features

- **High Accuracy**: 83% success rate on building images
- **Expected Number Detection**: Check if a specific number exists with confidence scoring
- **Hybrid Detection Methods**: Combines OCR + fuzzy matching + template matching
- **Smart Error Correction**: Handles common OCR errors (O→0, l→1, etc.)
- **Simple API**: Easy-to-use Python class interface
- **Batch Processing**: Process multiple images at once
- **Confidence Scores**: Get detection confidence levels
- **Flexible Output**: Raw text or digits-only extraction
- **Production Ready**: Error handling and logging included

## 📋 API Reference

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --images` | Specific image files to process | - |
| `-f, --folder` | Folder containing images | example_images |
| `-o, --output` | Output file to save results | Print to console |
| `--detailed` | Show detailed results with bounding boxes | False |
| `--confidence` | Minimum confidence threshold (0.0-1.0) | 0.8 |
| `--include-text` | Include non-numeric text in results | False |
| `--languages` | Language codes for OCR | en |
| `--verbose` | Enable verbose model loading | False |
| `--expected-number` | Specific number to search for with confidence | - |
| `--fuzzy-threshold` | Minimum similarity for fuzzy matching (0.0-1.0) | 0.8 |
| `--template-threshold` | Minimum confidence for template matching (0.0-1.0) | 0.7 |
| `--disable-template-matching` | Disable template matching fallback | False |

### BuildingNumberDetector Class

#### `__init__(languages=['en'], verbose=False)`
Initialize the detector with specified languages.

#### `detect_numbers(image_path, min_confidence=0.5, digits_only=True)`
Detect building numbers with detailed results including bounding boxes.

#### `get_building_numbers(image_path, min_confidence=0.8)`
Simple interface returning just the detected numbers as strings.

#### `process_batch(image_folder, output_file=None)`
Process multiple images in a folder, optionally saving results to file.

#### `check_for_expected_number(image_path, expected_number, fuzzy_threshold=0.8, template_threshold=0.7, use_template_matching=True)` ⭐ New
Check if a specific expected number exists in the image with confidence scoring. Returns detailed results including detection method used and confidence level.

## 🔧 Configuration Options

- **min_confidence**: Adjust detection threshold (0.0-1.0)
- **digits_only**: Filter to only numeric results
- **languages**: Support for 70+ languages
- **verbose**: Enable detailed logging during model loading

## 🚧 Limitations

- Requires good image quality and lighting
- May struggle with very stylized fonts
- Occasional false positives on business signage
- CPU processing (GPU support available for faster inference)

## 💡 Alternative Options

If EasyOCR doesn't meet your needs:

1. **Cloud APIs** (higher accuracy, cost per request):
   - Google Vision API
   - AWS Textract  
   - Azure OCR

2. **YOLO Models** (for specialized training):
   - Train on SVHN dataset
   - Custom building number dataset

3. **Hybrid Approach**:
   - Use YOLO for number localization
   - Use OCR for text recognition

## 📈 Performance Notes

- First run downloads ~50MB of model files
- Subsequent runs are much faster
- Processing time: ~1-3 seconds per image (CPU)
- Memory usage: ~200MB after model loading

## 🔍 Next Steps

1. **Improve accuracy** by:
   - Adding image preprocessing (contrast, sharpening)
   - Implementing ensemble methods
   - Fine-tuning on building-specific dataset

2. **Scale for production** by:
   - Adding GPU support
   - Implementing caching
   - Adding web API interface