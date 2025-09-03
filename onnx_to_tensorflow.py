"""
ONNX to TensorFlow Conversion Script
Converts ONNX models to TensorFlow SavedModel format.
"""

import os
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
import numpy as np
from pathlib import Path


class ONNXToTensorFlowConverter:
    """Convert ONNX models to TensorFlow format"""
    
    def __init__(self):
        self.input_dir = Path("exported_models")
        self.output_dir = Path("tensorflow_models")
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_craft_detector(self):
        """Convert CRAFT detector from ONNX to TensorFlow"""
        print("=== Converting CRAFT Detector: ONNX -> TensorFlow ===")
        
        onnx_path = self.input_dir / "craft_detector.onnx"
        tf_path = self.output_dir / "craft_detector_tf"
        
        if not onnx_path.exists():
            print(f"[ERROR] ONNX model not found: {onnx_path}")
            return False
            
        try:
            # Load ONNX model
            print(f"[INFO] Loading ONNX model from: {onnx_path}")
            onnx_model = onnx.load(str(onnx_path))
            
            # Check ONNX model
            onnx.checker.check_model(onnx_model)
            print("[INFO] ONNX model validation passed")
            
            # Print model info
            print(f"[INFO] ONNX model opset version: {onnx_model.opset_import[0].version}")
            print(f"[INFO] Number of nodes: {len(onnx_model.graph.node)}")
            
            # Convert to TensorFlow
            print("[INFO] Converting ONNX to TensorFlow...")
            tf_rep = prepare(onnx_model)
            
            # Export TensorFlow model
            print(f"[INFO] Exporting TensorFlow model to: {tf_path}")
            tf_rep.export_graph(str(tf_path))
            
            print("[SUCCESS] CRAFT detector converted to TensorFlow")
            
            # Get model size
            tf_size_mb = self._get_directory_size(tf_path) / (1024 * 1024)
            onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            
            print(f"[INFO] Original ONNX size: {onnx_size_mb:.2f} MB")
            print(f"[INFO] TensorFlow SavedModel size: {tf_size_mb:.2f} MB")
            
            return str(tf_path)
            
        except Exception as e:
            print(f"[ERROR] Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_tensorflow_model(self, tf_model_path):
        """Test the converted TensorFlow model"""
        if not tf_model_path or not Path(tf_model_path).exists():
            print("[ERROR] TensorFlow model not found")
            return False
            
        print(f"\n=== Testing TensorFlow Model ===")
        print(f"Model path: {tf_model_path}")
        
        try:
            # Load TensorFlow model
            model = tf.saved_model.load(tf_model_path)
            
            # Get signature info
            signatures = list(model.signatures.keys())
            print(f"[INFO] Available signatures: {signatures}")
            
            # Use default signature
            signature_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            if signature_key in model.signatures:
                infer = model.signatures[signature_key]
                
                # Print input/output info
                print(f"[INFO] Input specs: {infer.structured_input_signature}")
                print(f"[INFO] Output specs: {infer.structured_outputs}")
                
                # Create dummy input
                dummy_input = tf.random.normal([1, 3, 640, 640])
                print(f"[INFO] Testing with input shape: {dummy_input.shape}")
                
                # Run inference
                outputs = infer(dummy_input)
                
                print("[SUCCESS] TensorFlow model inference completed")
                for key, output in outputs.items():
                    print(f"[INFO] Output '{key}' shape: {output.shape}")
                    print(f"[INFO] Output '{key}' range: [{tf.reduce_min(output):.3f}, {tf.reduce_max(output):.3f}]")
                
                return True
            else:
                print(f"[ERROR] Default signature not found. Available: {signatures}")
                return False
                
        except Exception as e:
            print(f"[ERROR] TensorFlow model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_directory_size(self, directory_path):
        """Get total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def convert_all(self):
        """Convert all available ONNX models"""
        print("=== ONNX to TensorFlow Conversion Pipeline ===")
        
        # Convert CRAFT detector
        craft_tf_path = self.convert_craft_detector()
        
        # Test converted model
        if craft_tf_path:
            self.test_tensorflow_model(craft_tf_path)
        
        # Summary
        print("\n=== Conversion Summary ===")
        if craft_tf_path:
            print(f"[SUCCESS] CRAFT Detector: {craft_tf_path}")
        else:
            print("[ERROR] CRAFT Detector: Failed")
            
        return craft_tf_path


def main():
    """Main conversion function"""
    try:
        converter = ONNXToTensorFlowConverter()
        craft_path = converter.convert_all()
        
        if craft_path:
            print("\n[SUCCESS] ONNX to TensorFlow conversion completed!")
        else:
            print("\n[ERROR] Conversion failed")
            
    except Exception as e:
        print(f"[ERROR] Conversion process failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()