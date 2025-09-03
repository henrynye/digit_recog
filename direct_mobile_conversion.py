"""
Direct Mobile Conversion Script
Alternative approach that bypasses problematic onnx-tf dependencies.
Creates a mobile-ready inference pipeline using ONNX Runtime for mobile deployment.
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
import json
import pickle
import shutil


class DirectMobileConverter:
    """Convert ONNX models directly to mobile-ready format"""
    
    def __init__(self):
        self.models_dir = Path("mobile_models")
        self.models_dir.mkdir(exist_ok=True)
        
    def analyze_onnx_model(self, onnx_path):
        """Analyze ONNX model to understand its structure"""
        print(f"\n=== Analyzing ONNX Model ===")
        print(f"Model: {onnx_path}")
        
        try:
            # Load ONNX session
            session = ort.InferenceSession(str(onnx_path))
            
            # Get model info
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            model_info = {
                'path': str(onnx_path),
                'inputs': [],
                'outputs': [],
                'providers': session.get_providers()
            }
            
            print(f"[INFO] Available providers: {session.get_providers()}")
            
            for inp in inputs:
                input_info = {
                    'name': inp.name,
                    'shape': inp.shape,
                    'type': str(inp.type)
                }
                model_info['inputs'].append(input_info)
                print(f"[INFO] Input: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
            
            for out in outputs:
                output_info = {
                    'name': out.name,
                    'shape': out.shape,
                    'type': str(out.type)
                }
                model_info['outputs'].append(output_info)
                print(f"[INFO] Output: {out.name}, Shape: {out.shape}, Type: {out.type}")
            
            return model_info, session
            
        except Exception as e:
            print(f"[ERROR] Failed to analyze model: {e}")
            return None, None
    
    def create_mobile_inference_class(self, model_info, model_name="craft_detector"):
        """Create a Python class for mobile inference"""
        
        class_code = f'''"""
Mobile {model_name.upper()} Inference Class
Generated automatically for mobile deployment.
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path


class Mobile{model_name.title().replace('_', '')}:
    """Mobile-optimized {model_name} inference"""
    
    def __init__(self, model_path="{model_info['path']}", providers=None):
        """
        Initialize mobile inference
        
        Args:
            model_path: Path to ONNX model
            providers: List of execution providers (default: ['CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CPUExecutionProvider']  # CPU-only for mobile
            
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Model metadata
        self.input_name = "{model_info['inputs'][0]['name']}"
        self.input_shape = {model_info['inputs'][0]['shape']}
        self.output_names = {[out['name'] for out in model_info['outputs']]}
        
        print(f"[INFO] {{self.__class__.__name__}} initialized")
        print(f"[INFO] Input shape: {{self.input_shape}}")
        print(f"[INFO] Providers: {{self.session.get_providers()}}")
        
    def preprocess(self, image):
        """
        Preprocess input image for inference
        
        Args:
            image: Input image (numpy array or path)
            
        Returns:
            preprocessed: Preprocessed tensor ready for inference
        """
        if isinstance(image, (str, Path)):
            # Load image from path
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {{image}}")
        else:
            # Use provided numpy array
            img = image
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get target size from model input shape
        target_height = self.input_shape[2] if len(self.input_shape) > 2 else 640
        target_width = self.input_shape[3] if len(self.input_shape) > 3 else 640
        
        # Resize image
        img = cv2.resize(img, (target_width, target_height))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert HWC to CHW and add batch dimension
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        
        return img
    
    def predict(self, image):
        """
        Run inference on image
        
        Args:
            image: Input image (numpy array or path)
            
        Returns:
            outputs: Dictionary of model outputs
        """
        # Preprocess input
        preprocessed = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {{self.input_name: preprocessed}})
        
        # Create output dictionary
        result = {{}}
        for i, name in enumerate(self.output_names):
            result[name] = outputs[i]
            
        return result
    
    def get_model_size_mb(self):
        """Get approximate model size in MB"""
        model_path = Path("{model_info['path']}")
        if model_path.exists():
            return model_path.stat().st_size / (1024 * 1024)
        return 0
        
    def benchmark(self, num_runs=10, input_size=None):
        """
        Benchmark inference speed
        
        Args:
            num_runs: Number of inference runs
            input_size: Input size tuple (H, W) or None for default
            
        Returns:
            avg_time_ms: Average inference time in milliseconds
        """
        import time
        
        # Create dummy input
        if input_size is None:
            height = self.input_shape[2] if len(self.input_shape) > 2 else 640
            width = self.input_shape[3] if len(self.input_shape) > 3 else 640
        else:
            height, width = input_size
            
        dummy_input = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            self.predict(dummy_input)
            
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.predict(dummy_input)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to milliseconds
            
        avg_time = np.mean(times)
        print(f"[BENCHMARK] Average inference time: {{avg_time:.2f}} ms")
        print(f"[BENCHMARK] Model size: {{self.get_model_size_mb():.2f}} MB")
        
        return avg_time


def main():
    """Example usage"""
    import sys
    
    # Initialize mobile detector
    detector = Mobile{model_name.title().replace('_', '')}()
    
    # Test with example image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "example_images/20.jpg"
        
    if Path(image_path).exists():
        print(f"\\nTesting with image: {{image_path}}")
        results = detector.predict(image_path)
        
        for name, output in results.items():
            print(f"Output '{{name}}': shape={{output.shape}}, range=[{{output.min():.3f}}, {{output.max():.3f}}]")
            
        # Benchmark
        detector.benchmark()
    else:
        print(f"Test image not found: {{image_path}}")
        print("Running benchmark with dummy data...")
        detector.benchmark()


if __name__ == "__main__":
    main()
'''
        
        # Save the class to file
        class_file = self.models_dir / f"mobile_{model_name}.py"
        with open(class_file, 'w') as f:
            f.write(class_code)
            
        print(f"[SUCCESS] Mobile inference class created: {class_file}")
        return class_file
    
    def create_mobile_package(self, onnx_path, model_name="craft_detector"):
        """Create a complete mobile package"""
        print(f"\n=== Creating Mobile Package for {model_name} ===")
        
        # Analyze model
        model_info, session = self.analyze_onnx_model(onnx_path)
        if not model_info:
            return None
            
        # Copy ONNX model to mobile directory
        model_dest = self.models_dir / f"{model_name}.onnx"
        shutil.copy2(onnx_path, model_dest)
        print(f"[INFO] ONNX model copied to: {model_dest}")
        
        # Update model info with new path
        model_info['path'] = str(model_dest)
        
        # Create mobile inference class
        class_file = self.create_mobile_inference_class(model_info, model_name)
        
        # Save model metadata
        metadata_file = self.models_dir / f"{model_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"[INFO] Model metadata saved: {metadata_file}")
        
        # Create requirements file
        requirements_file = self.models_dir / "requirements.txt"
        requirements_content = """# Mobile inference requirements
onnxruntime>=1.22.0  # Core inference engine (12-20MB)
numpy>=1.21.0        # Numerical operations
opencv-python>=4.5.0  # Image processing (can be replaced with PIL for smaller size)

# For minimal mobile deployment, consider:
# onnxruntime-mobile  # Even smaller runtime for mobile devices
# Pillow              # Lighter alternative to opencv-python
"""
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        print(f"[INFO] Requirements file created: {requirements_file}")
        
        # Create deployment guide
        guide_file = self.models_dir / "deployment_guide.md"
        guide_content = f"""# Mobile Deployment Guide

## Model: {model_name}

### Quick Start
```python
from mobile_{model_name} import Mobile{model_name.title().replace('_', '')}

# Initialize detector
detector = Mobile{model_name.title().replace('_', '')}()

# Run inference
results = detector.predict("path/to/image.jpg")
print(results)
```

### Model Information
- **Model Size**: {Path(onnx_path).stat().st_size / (1024*1024):.2f} MB
- **Input Shape**: {model_info['inputs'][0]['shape']}
- **Output Count**: {len(model_info['outputs'])}
- **Runtime**: ONNX Runtime (CPU optimized)

### Mobile Integration Options

#### 1. Python Mobile App (Kivy/BeeWare)
```python
# Direct integration in Python mobile apps
from mobile_{model_name} import Mobile{model_name.title().replace('_', '')}
detector = Mobile{model_name.title().replace('_', '')}()
```

#### 2. Android (via Chaquopy)
```python
# Use in Android apps with Chaquopy Python integration
from mobile_{model_name} import Mobile{model_name.title().replace('_', '')}
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
- Model loading: ~{Path(onnx_path).stat().st_size / (1024*1024):.0f}MB
- Inference (per image): ~5-10MB
- Total app overhead: ~20-30MB

### Next Steps
1. Test the mobile class: `python mobile_{model_name}.py`
2. Integrate into your mobile app framework
3. Optimize for your specific deployment target
4. Consider converting to platform-specific formats (Core ML, TensorFlow Lite, etc.)
"""
        
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        print(f"[INFO] Deployment guide created: {guide_file}")
        
        return {
            'model_file': model_dest,
            'class_file': class_file,
            'metadata_file': metadata_file,
            'requirements_file': requirements_file,
            'guide_file': guide_file
        }
    
    def process_all_models(self):
        """Process all available ONNX models"""
        print("=== Direct Mobile Conversion Pipeline ===")
        
        onnx_dir = Path("exported_models")
        results = {}
        
        # Process CRAFT detector
        craft_path = onnx_dir / "craft_detector.onnx"
        if craft_path.exists():
            print(f"\nProcessing CRAFT detector...")
            craft_package = self.create_mobile_package(craft_path, "craft_detector")
            results['craft_detector'] = craft_package
        else:
            print(f"[WARNING] CRAFT model not found: {craft_path}")
        
        # Process CRNN recognizer if available
        crnn_path = onnx_dir / "crnn_recognizer.onnx"
        if crnn_path.exists():
            print(f"\nProcessing CRNN recognizer...")
            crnn_package = self.create_mobile_package(crnn_path, "crnn_recognizer")
            results['crnn_recognizer'] = crnn_package
        else:
            print(f"[INFO] CRNN model not available (expected): {crnn_path}")
        
        return results


def main():
    """Main conversion function"""
    try:
        converter = DirectMobileConverter()
        results = converter.process_all_models()
        
        print("\n=== Mobile Conversion Results ===")
        for model_name, package in results.items():
            if package:
                print(f"[SUCCESS] {model_name}: {package['class_file']}")
            else:
                print(f"[ERROR] {model_name}: Failed")
        
        print(f"\n[SUCCESS] Mobile packages created in: {converter.models_dir}")
        print(f"[INFO] Total models converted: {len([r for r in results.values() if r])}")
        
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()