"""
Mobile CRAFT_DETECTOR Inference Class
Generated automatically for mobile deployment.
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path


class MobileCraftDetector:
    """Mobile-optimized craft_detector inference"""
    
    def __init__(self, model_path="mobile_models\craft_detector.onnx", providers=None):
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
        self.input_name = "input_image"
        self.input_shape = ['batch_size', 3, 640, 640]
        self.output_names = ['text_region_score', 'affinity_score']
        
        print(f"[INFO] {self.__class__.__name__} initialized")
        print(f"[INFO] Input shape: {self.input_shape}")
        print(f"[INFO] Providers: {self.session.get_providers()}")
        
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
                raise ValueError(f"Could not load image: {image}")
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
        outputs = self.session.run(None, {self.input_name: preprocessed})
        
        # Create output dictionary
        result = {}
        for i, name in enumerate(self.output_names):
            result[name] = outputs[i]
            
        return result
    
    def get_model_size_mb(self):
        """Get approximate model size in MB"""
        model_path = Path("mobile_models\craft_detector.onnx")
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
        print(f"[BENCHMARK] Average inference time: {avg_time:.2f} ms")
        print(f"[BENCHMARK] Model size: {self.get_model_size_mb():.2f} MB")
        
        return avg_time


def main():
    """Example usage"""
    import sys
    
    # Initialize mobile detector
    detector = MobileCraftDetector()
    
    # Test with example image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "example_images/20.jpg"
        
    if Path(image_path).exists():
        print(f"\nTesting with image: {image_path}")
        results = detector.predict(image_path)
        
        for name, output in results.items():
            print(f"Output '{name}': shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")
            
        # Benchmark
        detector.benchmark()
    else:
        print(f"Test image not found: {image_path}")
        print("Running benchmark with dummy data...")
        detector.benchmark()


if __name__ == "__main__":
    main()
