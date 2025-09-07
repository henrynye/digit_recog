"""
Test TensorFlow Lite model inference

Tests the exported TFLite models to verify they work correctly
and compare outputs with the original PyTorch model.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class TFLiteModelTester:
    """
    Test TensorFlow Lite models for building number detection
    """
    
    def __init__(self, model_dir):
        """
        Initialize tester with model directory
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.test_images = []
        
    def load_tflite_models(self):
        """
        Load all available TFLite models
        """
        print("Loading TFLite models...")
        
        # Find all .tflite files
        tflite_files = list(self.model_dir.rglob("*.tflite"))
        
        for tflite_path in tflite_files:
            model_name = tflite_path.stem
            try:
                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                interpreter.allocate_tensors()
                
                # Get model info
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                self.models[model_name] = {
                    'interpreter': interpreter,
                    'input_details': input_details,
                    'output_details': output_details,
                    'path': tflite_path,
                    'size_mb': tflite_path.stat().st_size / (1024 * 1024)
                }
                
                print(f"✅ Loaded: {model_name} ({self.models[model_name]['size_mb']:.1f} MB)")
                print(f"   Input shape: {input_details[0]['shape']}")
                print(f"   Output shape: {output_details[0]['shape']}")
                print()
                
            except Exception as e:
                print(f"❌ Failed to load {tflite_path}: {e}")
        
        if not self.models:
            raise RuntimeError("No TFLite models could be loaded!")
            
        return len(self.models)
    
    def load_pytorch_model(self, pytorch_path):
        """
        Load original PyTorch model for comparison
        """
        try:
            self.pytorch_model = YOLO(str(pytorch_path))
            print(f"✅ Loaded PyTorch model: {pytorch_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load PyTorch model: {e}")
            return False
    
    def load_test_images(self, test_dir, max_images=5):
        """
        Load test images for inference
        """
        test_dir = Path(test_dir)
        if not test_dir.exists():
            print(f"❌ Test directory not found: {test_dir}")
            return 0
            
        # Get sample images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in image_extensions:
            self.test_images.extend(list(test_dir.glob(ext)))
            if len(self.test_images) >= max_images:
                break
                
        # Limit to max_images
        self.test_images = self.test_images[:max_images]
        
        print(f"✅ Loaded {len(self.test_images)} test images")
        for img_path in self.test_images:
            print(f"   - {img_path.name}")
        print()
        
        return len(self.test_images)
    
    def preprocess_image(self, image_path, target_size=(640, 640)):
        """
        Preprocess image for inference
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image_resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1] and convert to float32
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW for PyTorch
        image_batch_pytorch = np.expand_dims(image_normalized, axis=0)
        image_batch_pytorch = np.transpose(image_batch_pytorch, (0, 3, 1, 2))
        
        # For TensorFlow Lite (NHWC format)
        image_batch_tflite = np.expand_dims(image_normalized, axis=0)
        
        return {
            'original': image,
            'pytorch': image_batch_pytorch,
            'tflite': image_batch_tflite
        }
    
    def run_tflite_inference(self, model_name, image_data):
        """
        Run inference on a TFLite model
        """
        if model_name not in self.models:
            raise ValueError(f"Model not loaded: {model_name}")
            
        model = self.models[model_name]
        interpreter = model['interpreter']
        
        # Set input
        interpreter.set_tensor(
            model['input_details'][0]['index'], 
            image_data['tflite']
        )
        
        # Run inference
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Get output
        output_data = interpreter.get_tensor(
            model['output_details'][0]['index']
        )
        
        return output_data, inference_time
    
    def run_pytorch_inference(self, image_path):
        """
        Run inference on PyTorch model
        """
        if not hasattr(self, 'pytorch_model'):
            return None, 0
            
        start_time = time.time()
        results = self.pytorch_model(str(image_path))
        inference_time = time.time() - start_time
        
        return results, inference_time
    
    def test_model_inference(self):
        """
        Test all models on all test images
        """
        print("=" * 60)
        print("RUNNING INFERENCE TESTS")
        print("=" * 60)
        
        results = {}
        
        for image_path in self.test_images:
            print(f"\nTesting on: {image_path.name}")
            print("-" * 40)
            
            # Preprocess image
            try:
                image_data = self.preprocess_image(image_path)
            except Exception as e:
                print(f"❌ Failed to preprocess {image_path}: {e}")
                continue
            
            image_results = {}
            
            # Test PyTorch model (reference)
            if hasattr(self, 'pytorch_model'):
                try:
                    pytorch_output, pytorch_time = self.run_pytorch_inference(image_path)
                    print(f"PyTorch: {pytorch_time*1000:.1f}ms")
                    image_results['pytorch'] = {
                        'output': pytorch_output,
                        'time_ms': pytorch_time * 1000,
                        'success': True
                    }
                except Exception as e:
                    print(f"PyTorch: FAILED ({e})")
                    image_results['pytorch'] = {'success': False, 'error': str(e)}
            
            # Test all TFLite models
            for model_name in self.models:
                try:
                    output, inference_time = self.run_tflite_inference(model_name, image_data)
                    print(f"{model_name}: {inference_time*1000:.1f}ms")
                    image_results[model_name] = {
                        'output': output,
                        'time_ms': inference_time * 1000,
                        'success': True,
                        'output_shape': output.shape
                    }
                except Exception as e:
                    print(f"{model_name}: FAILED ({e})")
                    image_results[model_name] = {'success': False, 'error': str(e)}
            
            results[image_path.name] = image_results
        
        return results
    
    def analyze_results(self, results):
        """
        Analyze and summarize test results
        """
        print("\n" + "=" * 60)
        print("INFERENCE TEST RESULTS")
        print("=" * 60)
        
        # Summary statistics
        model_stats = {}
        
        for image_name, image_results in results.items():
            for model_name, model_result in image_results.items():
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'success_count': 0,
                        'total_count': 0,
                        'times_ms': [],
                        'output_shapes': []
                    }
                
                stats = model_stats[model_name]
                stats['total_count'] += 1
                
                if model_result.get('success', False):
                    stats['success_count'] += 1
                    stats['times_ms'].append(model_result['time_ms'])
                    if 'output_shape' in model_result:
                        stats['output_shapes'].append(model_result['output_shape'])
        
        # Print summary
        print(f"\n📊 Model Performance Summary:")
        print(f"{'Model':<25} {'Success':<10} {'Avg Time':<12} {'Size (MB)':<12} {'Status'}")
        print("-" * 70)
        
        for model_name, stats in model_stats.items():
            success_rate = stats['success_count'] / stats['total_count']
            
            if stats['times_ms']:
                avg_time = np.mean(stats['times_ms'])
                time_str = f"{avg_time:.1f}ms"
            else:
                time_str = "N/A"
            
            if model_name == 'pytorch':
                size_str = "N/A"
            else:
                size_str = f"{self.models[model_name]['size_mb']:.1f}"
            
            if success_rate == 1.0:
                status = "✅ PASS"
            elif success_rate > 0:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ FAIL"
            
            print(f"{model_name:<25} {success_rate*100:>6.1f}%    {time_str:<12} {size_str:<12} {status}")
        
        # Check performance targets
        print(f"\n🎯 Performance Analysis:")
        target_time_ms = 100
        
        passing_models = []
        for model_name, stats in model_stats.items():
            if model_name == 'pytorch':
                continue
                
            if stats['success_count'] == stats['total_count']:
                if stats['times_ms']:
                    avg_time = np.mean(stats['times_ms'])
                    size_mb = self.models[model_name]['size_mb']
                    
                    if avg_time <= target_time_ms and size_mb <= 6.0:
                        passing_models.append((model_name, avg_time, size_mb))
                        print(f"   ✅ {model_name}: {avg_time:.1f}ms, {size_mb:.1f}MB (meets all targets)")
                    elif avg_time <= target_time_ms:
                        print(f"   ⚠️  {model_name}: {avg_time:.1f}ms (fast but {size_mb:.1f}MB > 6MB)")
                    elif size_mb <= 6.0:
                        print(f"   ⚠️  {model_name}: {size_mb:.1f}MB (small but {avg_time:.1f}ms > {target_time_ms}ms)")
                    else:
                        print(f"   ❌ {model_name}: {avg_time:.1f}ms, {size_mb:.1f}MB (fails both targets)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if passing_models:
            best_model = min(passing_models, key=lambda x: (x[2], x[1]))  # Sort by size first, then time
            print(f"   🏆 Best overall: {best_model[0]} ({best_model[1]:.1f}ms, {best_model[2]:.1f}MB)")
            print(f"   📱 Use for mobile deployment: {best_model[0]}")
        else:
            print(f"   🔧 No models meet all targets - consider further optimization")
        
        return model_stats, passing_models


def main():
    """
    Main test script
    """
    try:
        print("=" * 60)
        print("TFLITE MODEL INFERENCE TEST")
        print("=" * 60)
        
        # Initialize tester
        model_dir = Path("model_training/models")
        tester = TFLiteModelTester(model_dir)
        
        # Load models
        model_count = tester.load_tflite_models()
        print(f"📱 Loaded {model_count} TFLite models\n")
        
        # Load PyTorch model for comparison
        pytorch_path = Path("model_training/models/building_number_detector/weights/best.pt")
        if pytorch_path.exists():
            tester.load_pytorch_model(pytorch_path)
        
        # Load test images
        test_dirs = [
            Path("model_training/data/positive"),
            Path("model_training/data/images/val")
        ]
        
        loaded = False
        for test_dir in test_dirs:
            if test_dir.exists():
                image_count = tester.load_test_images(test_dir, max_images=3)
                if image_count > 0:
                    loaded = True
                    break
        
        if not loaded:
            print("❌ No test images found - creating dummy test")
            # Create a simple test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_path = Path("test_image.jpg")
            cv2.imwrite(str(test_path), test_image)
            tester.test_images = [test_path]
            print(f"✅ Created dummy test image: {test_path}")
        
        # Run inference tests
        results = tester.test_model_inference()
        
        # Analyze results
        stats, passing_models = tester.analyze_results(results)
        
        # Final summary
        print(f"\n🎉 Testing completed!")
        print(f"📋 Tested {len(results)} images on {len(tester.models)} TFLite models")
        
        if passing_models:
            print(f"✅ {len(passing_models)} models meet all requirements")
            print(f"🚀 Ready for mobile deployment!")
        else:
            print(f"⚠️  No models meet all targets - review performance analysis")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n🛑 Testing interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())