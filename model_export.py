"""
EasyOCR Model Export Script
Extracts the CRAFT detector and CRNN recognizer models from EasyOCR and exports them to ONNX format.
"""

import os
import torch
import onnx
import easyocr
import numpy as np
from pathlib import Path


class EasyOCRModelExporter:
    """Export EasyOCR models to ONNX format"""
    
    def __init__(self, languages=['en'], verbose=False):
        """Initialize EasyOCR reader to access internal models"""
        print("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(languages, verbose=verbose)
        
        # Access the internal models
        self.detector = self.reader.detector
        self.recognizer = self.reader.recognizer
        
        # Create export directory
        self.export_dir = Path("exported_models")
        self.export_dir.mkdir(exist_ok=True)
        
    def get_model_info(self):
        """Print information about the loaded models"""
        print("\n=== EasyOCR Model Information ===")
        print(f"Detector type: {type(self.detector)}")
        print(f"Recognizer type: {type(self.recognizer)}")
        
        # Print model parameters
        detector_params = sum(p.numel() for p in self.detector.parameters())
        recognizer_params = sum(p.numel() for p in self.recognizer.parameters())
        
        print(f"Detector parameters: {detector_params:,}")
        print(f"Recognizer parameters: {recognizer_params:,}")
        
    def export_craft_detector(self, image_height=640, image_width=640):
        """
        Export CRAFT text detection model to ONNX
        
        Args:
            image_height: Height of input image (default: 640)
            image_width: Width of input image (default: 640)
        """
        print(f"\n=== Exporting CRAFT Detector ===")
        print(f"Input shape: [1, 3, {image_height}, {image_width}]")
        
        # Set model to evaluation mode
        self.detector.eval()
        
        # Create dummy input tensor
        dummy_input = torch.randn(1, 3, image_height, image_width)
        
        # Export path
        onnx_path = self.export_dir / "craft_detector.onnx"
        
        try:
            # Export to ONNX
            torch.onnx.export(
                self.detector,                     # Model to export
                dummy_input,                       # Dummy input
                str(onnx_path),                   # Output path
                export_params=True,               # Store trained parameters
                opset_version=11,                 # ONNX opset version
                do_constant_folding=True,         # Optimize constant folding
                input_names=['input_image'],      # Input names
                output_names=['text_region_score', 'affinity_score'],  # Output names
                dynamic_axes={
                    'input_image': {0: 'batch_size'},
                    'text_region_score': {0: 'batch_size'},
                    'affinity_score': {0: 'batch_size'}
                }
            )
            
            print(f"[SUCCESS] CRAFT detector exported to: {onnx_path}")
            
            # Verify the exported model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print("[SUCCESS] ONNX model verification passed")
            
            # Print model info
            file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"[INFO] Model size: {file_size_mb:.2f} MB")
            
            return str(onnx_path)
            
        except Exception as e:
            print(f"[ERROR] Error exporting CRAFT detector: {e}")
            return None
    
    def export_crnn_recognizer(self, image_height=32, image_width=128):
        """
        Export CRNN text recognition model to ONNX
        
        Args:
            image_height: Height of text region (default: 32)
            image_width: Width of text region (default: 128)
        """
        print(f"\n=== Exporting CRNN Recognizer ===")
        print(f"Input shape: [1, 1, {image_height}, {image_width}]")
        
        # Set model to evaluation mode
        self.recognizer.eval()
        
        # Create dummy input tensor (grayscale image as expected by CRNN)
        dummy_input = torch.randn(1, 1, image_height, image_width)
        
        # Create dummy text input (used for training, but should be None for inference)
        dummy_text = torch.LongTensor([0] * 25).unsqueeze(0)  # Max text length in EasyOCR
        
        # Export path
        onnx_path = self.export_dir / "crnn_recognizer.onnx"
        
        try:
            # First, let's try to understand the model's forward method
            print("[INFO] Analyzing recognizer model structure...")
            
            # Try different approaches based on EasyOCR's recognizer
            # The recognizer typically takes just the image input during inference
            
            # Create a wrapper that handles the forward call properly
            class RecognizerWrapper(torch.nn.Module):
                def __init__(self, recognizer):
                    super().__init__()
                    self.recognizer = recognizer
                
                def forward(self, image):
                    # EasyOCR recognizer expects RGB input and no text during inference
                    try:
                        # Try inference mode (text=None)
                        return self.recognizer(image, text=None)
                    except:
                        # Try with empty text tensor
                        batch_size = image.size(0)
                        empty_text = torch.LongTensor([0] * 25).unsqueeze(0).repeat(batch_size, 1)
                        return self.recognizer(image, text=empty_text)
            
            # Wrap the recognizer
            wrapped_model = RecognizerWrapper(self.recognizer)
            wrapped_model.eval()
            
            # Export to ONNX
            torch.onnx.export(
                wrapped_model,                    # Model to export
                dummy_input,                      # Dummy input (RGB image)
                str(onnx_path),                  # Output path
                export_params=True,              # Store trained parameters
                opset_version=11,                # ONNX opset version
                do_constant_folding=True,        # Optimize constant folding
                input_names=['text_image'],      # Input names
                output_names=['character_probabilities'],  # Output names
                dynamic_axes={
                    'text_image': {0: 'batch_size'},
                    'character_probabilities': {0: 'batch_size'}
                }
            )
            
            print(f"[SUCCESS] CRNN recognizer exported to: {onnx_path}")
            
            # Verify the exported model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print("[SUCCESS] ONNX model verification passed")
            
            # Print model info
            file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"[INFO] Model size: {file_size_mb:.2f} MB")
            
            return str(onnx_path)
            
        except Exception as e:
            print(f"[ERROR] Error exporting CRNN recognizer: {e}")
            print(f"[INFO] Trying alternative export approach...")
            
            # Alternative approach: Try to extract just the feature extraction and prediction parts
            try:
                # Create a simpler wrapper that only does image->features->predictions
                class SimpleRecognizerWrapper(torch.nn.Module):
                    def __init__(self, recognizer):
                        super().__init__()
                        self.feature_extraction = recognizer.FeatureExtraction
                        self.sequence_modeling = recognizer.SequenceModeling
                        self.prediction = recognizer.Prediction
                    
                    def forward(self, image):
                        # Feature extraction
                        visual_feature = self.feature_extraction(image)
                        visual_feature = visual_feature.permute(0, 3, 1, 2)  # [b, c, h, w] -> [b, w, h, c]
                        visual_feature = visual_feature.squeeze(2)  # [b, w, h, c] -> [b, w, c]
                        
                        # Sequence modeling
                        contextual_feature = self.sequence_modeling(visual_feature)
                        
                        # Prediction
                        prediction = self.prediction(contextual_feature)
                        
                        return prediction
                
                simple_wrapper = SimpleRecognizerWrapper(self.recognizer)
                simple_wrapper.eval()
                
                torch.onnx.export(
                    simple_wrapper,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['text_image'],
                    output_names=['character_probabilities'],
                    dynamic_axes={
                        'text_image': {0: 'batch_size'},
                        'character_probabilities': {0: 'batch_size'}
                    }
                )
                
                print(f"[SUCCESS] CRNN recognizer (alternative) exported to: {onnx_path}")
                
                # Verify the exported model
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                print("[SUCCESS] ONNX model verification passed")
                
                # Print model info
                file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
                print(f"[INFO] Model size: {file_size_mb:.2f} MB")
                
                return str(onnx_path)
                
            except Exception as e2:
                print(f"[ERROR] Alternative export also failed: {e2}")
                return None
    
    def test_original_models(self, test_image_path="example_images/20.jpg"):
        """Test the original EasyOCR models for comparison"""
        if not Path(test_image_path).exists():
            print(f"[ERROR] Test image not found: {test_image_path}")
            return None
            
        print(f"\n=== Testing Original EasyOCR Models ===")
        print(f"Test image: {test_image_path}")
        
        try:
            results = self.reader.readtext(test_image_path)
            print("[INFO] OCR Results:")
            for bbox, text, confidence in results:
                print(f"  Text: '{text}' (confidence: {confidence:.3f})")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Error testing original models: {e}")
            return None
    
    def export_all(self):
        """Export all models"""
        print("[INFO] Starting EasyOCR Model Export Process")
        
        # Print model information
        self.get_model_info()
        
        # Test original models first
        self.test_original_models()
        
        # Export CRAFT detector
        craft_path = self.export_craft_detector()
        
        # Export CRNN recognizer  
        crnn_path = self.export_crnn_recognizer()
        
        # Summary
        print("\n=== Export Summary ===")
        if craft_path:
            print(f"[SUCCESS] CRAFT Detector: {craft_path}")
        else:
            print("[ERROR] CRAFT Detector: Failed")
            
        if crnn_path:
            print(f"[SUCCESS] CRNN Recognizer: {crnn_path}")
        else:
            print("[ERROR] CRNN Recognizer: Failed")
        
        return craft_path, crnn_path


def main():
    """Main export function"""
    try:
        # Initialize exporter
        exporter = EasyOCRModelExporter(languages=['en'], verbose=False)
        
        # Export all models
        craft_path, crnn_path = exporter.export_all()
        
        print("\n[SUCCESS] Model export process completed!")
        
    except Exception as e:
        print(f"[ERROR] Export process failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()