# Mobile Building Number Detection Model Training Plan

## Overview
Create a lightweight YOLOv8 Nano + MobileNetV3 pipeline for building number detection on mobile devices, targeting ~10MB total model size.

**Dataset**: 953 valid shipments with confirmed building numbers + 1,690 invalid shipments
**Target**: 90%+ accuracy, <10MB TFLite models, <100ms inference time

---

## Phase 1: Dataset Preparation

### Step 1.1: Parse Validation Results
Create training data extraction script.

**File**: `parse_validation_results.py`
```python
# Parse validation_results.txt to extract:
# - Positive samples: Images with confirmed numbers
# - Address numbers for each positive image  
# - Negative samples: Images without numbers
```

**Action**: 
- Read validation_results.txt line by line
- For each VALID shipment, record (shipment_uid, address_number, valid_image_name)
- For each INVALID shipment, record all images as negative samples
- Output: CSV with columns [image_path, has_number, number_value]

**Test**: Verify 953 positive samples and 1,690+ negative samples extracted

### Step 1.2: Create File Structure
Set up organized training directory.

**Action**:
```bash
mkdir -p model_training/{data,models,scripts,results}
mkdir -p model_training/data/{positive,negative,annotations}
```

**Test**: Directory structure exists and is writable

### Step 1.3: Copy Training Images
Organize images by category.

**Action**:
- Copy confirmed positive images to `model_training/data/positive/`
- Copy 2-3 random images from each invalid shipment to `model_training/data/negative/`
- Rename files with format: `{shipment_uid}_{original_name}`

**Test**: ~953 positive images, ~3000+ negative images copied

### Step 1.4: Generate Bounding Box Labels
Create object detection annotations.

**File**: `create_bbox_annotations.py`
```python
# For each positive image:
# 1. Run EasyOCR to detect all text
# 2. Find bounding box containing the target number
# 3. Save in YOLO format: class_id x_center y_center width height
```

**Action**:
- Process each positive image with EasyOCR
- Filter detections to find target address number
- Convert bounding boxes to YOLO format (normalized coordinates)
- Save annotations as `image_name.txt` files

**Test**: Each positive image has corresponding `.txt` annotation file

---

## Phase 2: YOLOv8 Nano Detection Model

### Step 2.1: Install YOLOv8
Set up training environment.

**Action**:
```bash
pip install ultralytics opencv-python pillow tqdm
```

**Test**: `yolo --version` runs without error

### Step 2.2: Create YOLO Dataset Config
Set up training configuration.

**File**: `model_training/data/dataset.yaml`
```yaml
path: /path/to/model_training/data
train: train.txt
val: val.txt
nc: 1
names: ['number']
```

**Action**:
- Create train/val splits (80/20) from positive images
- Generate train.txt and val.txt with image paths
- Include some negative images (no annotations) for hard negative mining

**Test**: YOLO can load dataset without errors

### Step 2.3: Train Detection Model
Fine-tune YOLOv8 nano on building numbers.

**File**: `train_detector.py`
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='cpu'  # or 'mps' for Apple Silicon
)
```

**Action**:
- Start with 50 epochs
- Monitor validation mAP@0.5
- Stop early if no improvement for 10 epochs

**Test**: Model achieves >0.7 mAP@0.5 on validation set

### Step 2.4: Export Detection Model
Convert to TensorFlow Lite.

**Action**:
```python
model = YOLO('runs/detect/train/weights/best.pt')
model.export(format='tflite', int8=True, imgsz=640)
```

**Test**: TFLite model <6MB, runs inference successfully

---

## Phase 3: MobileNetV3 Recognition Model  

### Step 3.1: Create Number Recognition Dataset
Prepare cropped number images.

**File**: `create_recognition_dataset.py`
```python
# For each positive image:
# 1. Load image and YOLO detection
# 2. Crop bounding box region  
# 3. Save as training sample with number label
```

**Action**:
- Crop detected regions from positive images
- Resize to consistent size (128x32 pixels)
- Create labels file mapping image to digit sequence
- Split into train/val (80/20)

**Test**: ~800 cropped number images with correct labels

### Step 3.2: Generate Synthetic Data
Augment training set with synthetic numbers.

**File**: `generate_synthetic.py`
```python
# Generate numbers 1-999 with variations:
# - Multiple fonts (Arial, Times, etc.)
# - Different backgrounds  
# - Noise, blur, rotation
# - Various lighting conditions
```

**Action**:
- Generate 2000 synthetic number images
- Use common building number ranges (1-999)
- Apply realistic augmentations
- Save with same format as real data

**Test**: 2000+ synthetic images look realistic and diverse

### Step 3.3: Build Recognition Model
Create lightweight CNN for number recognition.

**File**: `train_recognizer.py`
```python
import tensorflow as tf
from tensorflow.keras import layers

def create_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(32, 128, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1000, activation='softmax')  # Numbers 0-999
    ])
    return model
```

**Action**:
- Use MobileNetV3Small as backbone (optional)
- Train for 100 epochs with early stopping
- Use categorical crossentropy loss
- Apply data augmentation during training

**Test**: Model achieves >90% accuracy on validation set

### Step 3.4: Convert Recognition Model
Export to TensorFlow Lite.

**Action**:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.INT8]
tflite_model = converter.convert()
```

**Test**: TFLite model <4MB, maintains >85% accuracy

---

## Phase 4: Model Integration & Testing

### Step 4.1: Create Inference Pipeline
Combine detection and recognition models.

**File**: `mobile_inference.py`
```python
class BuildingNumberDetector:
    def __init__(self, detector_path, recognizer_path):
        self.detector = tf.lite.Interpreter(detector_path)
        self.recognizer = tf.lite.Interpreter(recognizer_path)
    
    def detect_numbers(self, image):
        # 1. Run YOLO detection
        # 2. Crop detected regions
        # 3. Run recognition on crops
        # 4. Return numbers with confidence
        pass
```

**Action**:
- Load both TFLite models
- Implement preprocessing/postprocessing
- Add confidence thresholding
- Return detected numbers with positions

**Test**: Pipeline correctly detects numbers in test images

### Step 4.2: Benchmark Performance
Measure accuracy and speed.

**File**: `evaluate_models.py`
```python
# Test on held-out validation set:
# - Detection accuracy (mAP)
# - Recognition accuracy 
# - End-to-end accuracy
# - Inference time per image
```

**Action**:
- Test on 100 random images from invalid_shipments
- Measure false positive rate
- Time inference on mobile-sized images (1920x1080)
- Record model sizes

**Test**: 
- End-to-end accuracy >80%
- Inference time <100ms
- Total model size <10MB

### Step 4.3: Create Demo App
Build simple camera demo.

**File**: `camera_demo.py`
```python
import cv2
# Simple OpenCV app that:
# 1. Captures camera feed
# 2. Runs inference on each frame
# 3. Overlays detected numbers
```

**Action**:
- Use OpenCV for camera capture
- Display bounding boxes and numbers
- Add confidence scores
- Test with phone camera via USB/wireless

**Test**: Real-time detection works on live camera feed

---

## Phase 5: Production Optimization

### Step 5.1: Model Quantization
Further optimize model sizes.

**Action**:
- Apply post-training quantization
- Test quantization-aware training if accuracy drops
- Benchmark INT8 vs FP16 performance
- Target <8MB total size

**Test**: Quantized models maintain >85% accuracy

### Step 5.2: Mobile Integration
Create deployment-ready code.

**File**: `mobile_wrapper.py`
```python
class MobileBuildingDetector:
    # Optimized for Android/iOS deployment
    # - Batch processing
    # - GPU acceleration support  
    # - Memory efficient inference
```

**Action**:
- Add TensorFlow Lite GPU delegate support
- Optimize memory usage
- Add batch processing for multiple images
- Create simple API interface

**Test**: Runs efficiently on mobile device emulator

---

## Success Metrics

- **Accuracy**: >85% end-to-end number detection accuracy
- **Size**: Detection + Recognition models <10MB total
- **Speed**: <100ms inference time on mobile CPU
- **Coverage**: Works on building numbers 1-999
- **Robustness**: <5% false positive rate on images without numbers

## Next Steps

1. Start with Phase 1, Step 1.1
2. Test each step before proceeding
3. Iterate on model architecture if accuracy targets not met
4. Consider ensemble methods if single model insufficient