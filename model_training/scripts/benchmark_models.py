"""
Benchmark TensorFlow Lite and PyTorch models

Comprehensive performance testing including inference times,
memory usage, and accuracy comparison for mobile deployment.
"""

import sys
import os
import time
import statistics
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
import psutil
import gc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class ModelBenchmark:
    """
    Comprehensive benchmark suite for YOLOv8 models
    """
    
    def __init__(self, model_dir):
        """
        Initialize benchmark with model directory
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.pytorch_model = None
        self.test_images = []
        self.results = {}
        
    def load_models(self):
        """
        Load all available models
        """
        print("Loading models for benchmarking...")
        
        # Load TFLite models
        tflite_files = list(self.model_dir.rglob("*.tflite"))
        
        for tflite_path in tflite_files:
            model_name = tflite_path.stem
            try:
                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                self.models[model_name] = {
                    'type': 'tflite',
                    'interpreter': interpreter,
                    'input_details': input_details,
                    'output_details': output_details,
                    'path': tflite_path,
                    'size_mb': tflite_path.stat().st_size / (1024 * 1024)
                }
                
                print(f"✅ TFLite: {model_name} ({self.models[model_name]['size_mb']:.1f} MB)")
                
            except Exception as e:
                print(f"❌ Failed to load {tflite_path}: {e}")
        
        # Load PyTorch model
        pytorch_path = self.model_dir / "building_number_detector/weights/best.pt"
        if pytorch_path.exists():
            try:
                self.pytorch_model = YOLO(str(pytorch_path))
                pytorch_size = pytorch_path.stat().st_size / (1024 * 1024)
                
                self.models['pytorch_original'] = {
                    'type': 'pytorch',
                    'model': self.pytorch_model,
                    'path': pytorch_path,
                    'size_mb': pytorch_size
                }
                
                print(f"✅ PyTorch: pytorch_original ({pytorch_size:.1f} MB)")
                
            except Exception as e:
                print(f"❌ Failed to load PyTorch model: {e}")
        
        print(f"\n📱 Loaded {len(self.models)} models total\n")
        return len(self.models)
    
    def load_test_images(self, test_dir, num_images=10):
        """
        Load test images for benchmarking
        """
        test_dir = Path(test_dir)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in image_extensions:
            self.test_images.extend(list(test_dir.glob(ext)))
            if len(self.test_images) >= num_images:
                break
                
        self.test_images = self.test_images[:num_images]
        
        print(f"✅ Loaded {len(self.test_images)} test images")
        return len(self.test_images)
    
    def preprocess_image(self, image_path, target_size=(640, 640)):
        """
        Preprocess image for inference
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, target_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # TensorFlow Lite format (NHWC)
        image_batch_tflite = np.expand_dims(image_normalized, axis=0)
        
        return {
            'original': image,
            'tflite': image_batch_tflite,
            'path': image_path
        }
    
    def benchmark_tflite_model(self, model_name, image_data_list, num_runs=5):
        """
        Benchmark a TFLite model
        """
        if model_name not in self.models:
            return None
            
        model = self.models[model_name]
        interpreter = model['interpreter']
        
        times = []
        memory_usage = []
        outputs = []
        
        # Warmup runs
        for _ in range(2):
            interpreter.set_tensor(
                model['input_details'][0]['index'], 
                image_data_list[0]['tflite']
            )
            interpreter.invoke()
        
        # Benchmark runs
        for run in range(num_runs):
            run_times = []
            run_memory = []
            
            for image_data in image_data_list:
                # Measure memory before
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Set input tensor
                interpreter.set_tensor(
                    model['input_details'][0]['index'], 
                    image_data['tflite']
                )
                
                # Run inference with timing
                start_time = time.perf_counter()
                interpreter.invoke()
                end_time = time.perf_counter()
                
                inference_time = (end_time - start_time) * 1000  # ms
                
                # Get output
                output = interpreter.get_tensor(
                    model['output_details'][0]['index']
                )
                
                # Measure memory after
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                
                run_times.append(inference_time)
                run_memory.append(max(0, mem_after - mem_before))
                
                if run == 0:  # Store outputs from first run
                    outputs.append(output)
            
            times.extend(run_times)
            memory_usage.extend(run_memory)
            
            # Force garbage collection between runs
            gc.collect()
        
        return {
            'times_ms': times,
            'memory_mb': memory_usage,
            'outputs': outputs,
            'avg_time_ms': statistics.mean(times),
            'std_time_ms': statistics.stdev(times) if len(times) > 1 else 0,
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'avg_memory_mb': statistics.mean(memory_usage),
            'success': True
        }
    
    def benchmark_pytorch_model(self, image_paths, num_runs=5):
        """
        Benchmark PyTorch model
        """
        if self.pytorch_model is None:
            return None
            
        times = []
        memory_usage = []
        outputs = []
        
        # Warmup runs
        for _ in range(2):
            _ = self.pytorch_model(str(image_paths[0]))
        
        # Benchmark runs
        for run in range(num_runs):
            run_times = []
            run_memory = []
            
            for image_path in image_paths:
                # Measure memory before
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Run inference with timing
                start_time = time.perf_counter()
                results = self.pytorch_model(str(image_path))
                end_time = time.perf_counter()
                
                inference_time = (end_time - start_time) * 1000  # ms
                
                # Measure memory after
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                
                run_times.append(inference_time)
                run_memory.append(max(0, mem_after - mem_before))
                
                if run == 0:  # Store outputs from first run
                    outputs.append(results)
            
            times.extend(run_times)
            memory_usage.extend(run_memory)
            
            # Force garbage collection between runs
            gc.collect()
        
        return {
            'times_ms': times,
            'memory_mb': memory_usage,
            'outputs': outputs,
            'avg_time_ms': statistics.mean(times),
            'std_time_ms': statistics.stdev(times) if len(times) > 1 else 0,
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'avg_memory_mb': statistics.mean(memory_usage),
            'success': True
        }
    
    def run_comprehensive_benchmark(self, num_runs=5):
        """
        Run comprehensive benchmark on all models
        """
        print("=" * 60)
        print("COMPREHENSIVE MODEL BENCHMARK")
        print("=" * 60)
        print(f"📊 Images: {len(self.test_images)}")
        print(f"🔄 Runs per model: {num_runs}")
        print(f"🧮 Total inferences per model: {len(self.test_images) * num_runs}")
        print()
        
        # Preprocess all test images
        print("Preprocessing test images...")
        image_data_list = []
        image_paths = []
        
        for image_path in self.test_images:
            try:
                image_data = self.preprocess_image(image_path)
                image_data_list.append(image_data)
                image_paths.append(image_path)
            except Exception as e:
                print(f"⚠️  Failed to preprocess {image_path}: {e}")
        
        print(f"✅ Preprocessed {len(image_data_list)} images\n")
        
        # Benchmark each model
        self.results = {}
        
        for model_name, model_info in self.models.items():
            print(f"Benchmarking {model_name}...")
            print("-" * 40)
            
            try:
                if model_info['type'] == 'tflite':
                    result = self.benchmark_tflite_model(
                        model_name, image_data_list, num_runs
                    )
                elif model_info['type'] == 'pytorch':
                    result = self.benchmark_pytorch_model(
                        image_paths, num_runs
                    )
                else:
                    result = None
                
                if result:
                    result.update({
                        'model_size_mb': model_info['size_mb'],
                        'model_type': model_info['type']
                    })
                    self.results[model_name] = result
                    
                    print(f"✅ Completed: {result['avg_time_ms']:.1f}±{result['std_time_ms']:.1f}ms")
                else:
                    print(f"❌ Failed to benchmark {model_name}")
                    self.results[model_name] = {'success': False}
                
            except Exception as e:
                print(f"❌ Error benchmarking {model_name}: {e}")
                self.results[model_name] = {'success': False, 'error': str(e)}
            
            print()
        
        return self.results
    
    def generate_benchmark_report(self):
        """
        Generate comprehensive benchmark report
        """
        print("=" * 80)
        print("BENCHMARK REPORT")
        print("=" * 80)
        
        # Model comparison table
        print(f"\n📊 Performance Comparison:")
        print(f"{'Model':<25} {'Type':<8} {'Size (MB)':<10} {'Avg Time (ms)':<15} {'Memory (MB)':<12} {'Status'}")
        print("-" * 85)
        
        successful_models = []
        
        for model_name, result in self.results.items():
            if result.get('success', False):
                model_type = result['model_type']
                size_mb = result['model_size_mb']
                avg_time = result['avg_time_ms']
                std_time = result['std_time_ms']
                avg_memory = result['avg_memory_mb']
                
                time_str = f"{avg_time:.1f}±{std_time:.1f}"
                status = "✅ PASS"
                
                successful_models.append((model_name, avg_time, size_mb))
                
                print(f"{model_name:<25} {model_type:<8} {size_mb:<10.1f} {time_str:<15} {avg_memory:<12.1f} {status}")
            else:
                print(f"{model_name:<25} {'N/A':<8} {'N/A':<10} {'FAILED':<15} {'N/A':<12} ❌ FAIL")
        
        # Performance analysis
        print(f"\n🎯 Performance Analysis:")
        print(f"   Target: <100ms inference, <6MB model size")
        print()
        
        mobile_ready = []
        
        for model_name, avg_time, size_mb in successful_models:
            result = self.results[model_name]
            
            meets_time = avg_time <= 100
            meets_size = size_mb <= 6.0
            
            if meets_time and meets_size:
                mobile_ready.append((model_name, avg_time, size_mb))
                print(f"   ✅ {model_name}: {avg_time:.1f}ms, {size_mb:.1f}MB (mobile-ready)")
            elif meets_time:
                print(f"   ⚠️  {model_name}: {avg_time:.1f}ms (fast but {size_mb:.1f}MB > 6MB)")
            elif meets_size:
                print(f"   ⚠️  {model_name}: {size_mb:.1f}MB (compact but {avg_time:.1f}ms > 100ms)")
            else:
                print(f"   ❌ {model_name}: {avg_time:.1f}ms, {size_mb:.1f}MB (fails both criteria)")
        
        # Recommendations
        print(f"\n💡 Deployment Recommendations:")
        
        if mobile_ready:
            # Sort by size first (smaller is better), then by speed
            best_model = min(mobile_ready, key=lambda x: (x[2], x[1]))
            fastest_model = min(mobile_ready, key=lambda x: x[1])
            smallest_model = min(mobile_ready, key=lambda x: x[2])
            
            print(f"   🏆 Best Overall: {best_model[0]} ({best_model[1]:.1f}ms, {best_model[2]:.1f}MB)")
            print(f"   🚀 Fastest: {fastest_model[0]} ({fastest_model[1]:.1f}ms)")
            print(f"   💾 Smallest: {smallest_model[0]} ({smallest_model[2]:.1f}MB)")
            print(f"   📱 Recommended for mobile: {best_model[0]}")
            
            # Device-specific recommendations
            print(f"\n📱 Device Recommendations:")
            print(f"   • High-end mobile (>4GB RAM): {fastest_model[0]}")
            print(f"   • Budget mobile (<2GB RAM): {smallest_model[0]}")
            print(f"   • Balanced deployment: {best_model[0]}")
        else:
            print(f"   🔧 No models meet mobile deployment criteria")
            print(f"   📋 Consider further optimization or relaxed requirements")
        
        # Technical details
        print(f"\n🔧 Technical Details:")
        for model_name, result in self.results.items():
            if result.get('success', False):
                min_time = result['min_time_ms']
                max_time = result['max_time_ms']
                std_time = result['std_time_ms']
                
                print(f"   {model_name}:")
                print(f"     • Range: {min_time:.1f}ms - {max_time:.1f}ms")
                print(f"     • Std Dev: {std_time:.1f}ms")
                print(f"     • Memory Usage: {result['avg_memory_mb']:.1f}MB avg")
        
        # Summary
        total_models = len(self.results)
        successful = len([r for r in self.results.values() if r.get('success', False)])
        mobile_count = len(mobile_ready)
        
        print(f"\n📋 Summary:")
        print(f"   • Total models tested: {total_models}")
        print(f"   • Successful benchmarks: {successful}")
        print(f"   • Mobile-ready models: {mobile_count}")
        print(f"   • Test images: {len(self.test_images)}")
        
        return mobile_ready


def main():
    """
    Main benchmark script
    """
    try:
        print("=" * 60)
        print("MODEL BENCHMARK SUITE")
        print("=" * 60)
        
        # Initialize benchmark
        model_dir = Path("model_training/models")
        benchmark = ModelBenchmark(model_dir)
        
        # Load models
        model_count = benchmark.load_models()
        if model_count == 0:
            print("❌ No models found to benchmark!")
            return 1
        
        # Load test images
        test_dirs = [
            Path("model_training/data/images/val"),
            Path("model_training/data/positive")
        ]
        
        loaded = False
        for test_dir in test_dirs:
            if test_dir.exists():
                image_count = benchmark.load_test_images(test_dir, num_images=5)
                if image_count > 0:
                    loaded = True
                    break
        
        if not loaded:
            print("❌ No test images found!")
            return 1
        
        # Run benchmark
        results = benchmark.run_comprehensive_benchmark(num_runs=3)
        
        # Generate report
        mobile_ready = benchmark.generate_benchmark_report()
        
        # Final status
        if mobile_ready:
            print(f"\n🎉 Benchmark completed successfully!")
            print(f"✅ {len(mobile_ready)} models ready for mobile deployment")
        else:
            print(f"\n⚠️  Benchmark completed but no models meet mobile criteria")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n🛑 Benchmark interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())