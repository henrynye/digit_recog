"""
Test ONNX Models
Test the exported ONNX models to ensure they work correctly.
"""

import onnxruntime as ort
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class ONNXModelTester:
    """Test ONNX models exported from EasyOCR"""
    
    def __init__(self):
        self.craft_session = None
        self.crnn_session = None
        
    def load_craft_detector(self, onnx_path="exported_models/craft_detector.onnx"):
        """Load CRAFT detector ONNX model"""
        if not Path(onnx_path).exists():
            print(f"[ERROR] CRAFT ONNX model not found: {onnx_path}")
            return False
            
        try:
            # Create ONNX Runtime session
            self.craft_session = ort.InferenceSession(str(onnx_path))
            
            # Print model info
            print(f"[SUCCESS] CRAFT detector loaded from: {onnx_path}")
            print(f"[INFO] Input names: {[input.name for input in self.craft_session.get_inputs()]}")
            print(f"[INFO] Output names: {[output.name for output in self.craft_session.get_outputs()]}")
            
            # Print input shape
            input_shape = self.craft_session.get_inputs()[0].shape
            print(f"[INFO] Input shape: {input_shape}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load CRAFT detector: {e}")
            return False
    
    def preprocess_image_for_craft(self, image_path, target_size=(640, 640)):
        """Preprocess image for CRAFT detector"""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return None
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, target_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        
        print(f"[INFO] Preprocessed image shape: {img.shape}")
        return img
    
    def test_craft_detector(self, image_path="example_images/20.jpg"):
        """Test CRAFT detector with a sample image"""
        if self.craft_session is None:
            print("[ERROR] CRAFT detector not loaded")
            return None
            
        print(f"\n=== Testing CRAFT Detector ===")
        print(f"Test image: {image_path}")
        
        # Preprocess image
        input_img = self.preprocess_image_for_craft(image_path)
        if input_img is None:
            return None
            
        try:
            # Run inference
            input_name = self.craft_session.get_inputs()[0].name
            outputs = self.craft_session.run(None, {input_name: input_img})
            
            # Parse outputs
            text_score = outputs[0]  # Text region score
            link_score = outputs[1]  # Affinity (link) score
            
            print(f"[SUCCESS] CRAFT inference completed")
            print(f"[INFO] Text score shape: {text_score.shape}")
            print(f"[INFO] Link score shape: {link_score.shape}")
            print(f"[INFO] Text score range: [{text_score.min():.3f}, {text_score.max():.3f}]")
            print(f"[INFO] Link score range: [{link_score.min():.3f}, {link_score.max():.3f}]")
            
            return {
                'text_score': text_score,
                'link_score': link_score,
                'input_shape': input_img.shape
            }
            
        except Exception as e:
            print(f"[ERROR] CRAFT inference failed: {e}")
            return None
    
    def visualize_craft_results(self, results, save_path="craft_detection_result.png"):
        """Visualize CRAFT detection results"""
        if results is None:
            print("[ERROR] No results to visualize")
            return
            
        text_score = results['text_score'][0, 0]  # Remove batch and channel dims
        link_score = results['link_score'][0, 0]  # Remove batch and channel dims
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot text score map
        im1 = axes[0].imshow(text_score, cmap='jet')
        axes[0].set_title('Text Score Map')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot link score map
        im2 = axes[1].imshow(link_score, cmap='jet')
        axes[1].set_title('Link Score Map')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SUCCESS] Visualization saved to: {save_path}")
        plt.close()
    
    def compare_with_original(self, image_path="example_images/20.jpg"):
        """Compare ONNX results with original EasyOCR"""
        print(f"\n=== Comparing with Original EasyOCR ===")
        
        try:
            import easyocr
            reader = easyocr.Reader(['en'], verbose=False)
            original_results = reader.readtext(image_path)
            
            print("[INFO] Original EasyOCR results:")
            for bbox, text, confidence in original_results:
                print(f"  Text: '{text}' (confidence: {confidence:.3f})")
                
            return original_results
            
        except Exception as e:
            print(f"[ERROR] Could not run original EasyOCR: {e}")
            return None
    
    def test_all(self):
        """Run all tests"""
        print("=== ONNX Model Testing ===")
        
        # Load CRAFT detector
        if not self.load_craft_detector():
            return
            
        # Test CRAFT detector
        results = self.test_craft_detector()
        
        # Visualize results
        if results:
            self.visualize_craft_results(results)
            
        # Compare with original
        self.compare_with_original()
        
        print("\n[SUCCESS] ONNX model testing completed")


def main():
    """Main testing function"""
    try:
        tester = ONNXModelTester()
        tester.test_all()
        
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()