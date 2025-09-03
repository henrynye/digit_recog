# Mobile Deployment Guide

## Model: craft_detector

### Quick Start
```python
from mobile_craft_detector import MobileCraftDetector

# Initialize detector
detector = MobileCraftDetector()

# Run inference
results = detector.predict("path/to/image.jpg")
print(results)
```

### Model Information
- **Model Size**: 79.21 MB
- **Input Shape**: ['batch_size', 3, 640, 640]
- **Output Count**: 2
- **Runtime**: ONNX Runtime (CPU optimized)

### Mobile Integration Options

#### 1. Python Mobile App (Kivy/BeeWare)
```python
# Direct integration in Python mobile apps
from mobile_craft_detector import MobileCraftDetector
detector = MobileCraftDetector()
```

#### 2. Android (via Chaquopy)
```python
# Use in Android apps with Chaquopy Python integration
from mobile_craft_detector import MobileCraftDetector
```

#### 3. React Native (via bridge)
Create a Python service and communicate via bridge.

#### 4. Native Apps (C++/Java/Swift)
Use ONNX Runtime C++ API for optimal performance.

### Performance Optimization

#### For Smaller Size:
1. Use `onnxruntime-mobile` instead of full `onnxruntime`
2. Replace `opencv-python` with `Pillow` for image ops
3. Quantize model to INT8 (can reduce size by 75%)

#### For Faster Inference:
1. Use GPU providers if available
2. Enable all CPU optimizations
3. Consider model pruning

### Memory Usage
- Model loading: ~79MB
- Inference (per image): ~5-10MB
- Total app overhead: ~20-30MB

### Next Steps
1. Test the mobile class: `python mobile_craft_detector.py`
2. Integrate into your mobile app framework
3. Optimize for your specific deployment target
4. Consider converting to platform-specific formats (Core ML, TensorFlow Lite, etc.)
