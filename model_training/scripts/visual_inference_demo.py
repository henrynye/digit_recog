"""
Interactive Visual Inference Demo

Choose a model and image, then see detection results with visual overlay.
Perfect for testing and demonstrating mobile model performance.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class VisualInferenceDemo:
    """
    Interactive demo for mobile model inference with visual results
    """
    
    def __init__(self, model_dir="model_training/models"):
        """
        Initialize demo with model directory
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.model_specs = {}
        self.load_model_specs()
        
    def load_model_specs(self):
        """
        Load model specifications from accuracy evaluation results
        """
        results_file = Path("model_training/results/accuracy_evaluation.json")
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.model_specs = json.load(f)
        else:
            print("⚠️  No accuracy evaluation results found. Run evaluate_model_accuracy.py first.")
    
    def discover_models(self):
        """
        Discover available mobile-ready models
        """
        print("🔍 Discovering available models...")
        
        # Find TFLite models
        tflite_files = list(self.model_dir.rglob("*.tflite"))
        
        mobile_models = []
        
        for tflite_path in tflite_files:
            model_name = tflite_path.stem
            size_mb = tflite_path.stat().st_size / (1024 * 1024)
            
            # Filter for mobile-ready models (≤6MB)
            if size_mb <= 6.0:
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
                        'size_mb': size_mb
                    }
                    
                    mobile_models.append(model_name)
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {model_name}: {e}")
        
        # Add PyTorch model if it's mobile-ready
        pytorch_path = self.model_dir / "building_number_detector/weights/best.pt"
        if pytorch_path.exists():
            size_mb = pytorch_path.stat().st_size / (1024 * 1024)
            if size_mb <= 6.0:
                try:
                    model = YOLO(str(pytorch_path))
                    
                    self.models['pytorch_original'] = {
                        'type': 'pytorch',
                        'model': model,
                        'path': pytorch_path,
                        'size_mb': size_mb
                    }
                    
                    mobile_models.append('pytorch_original')
                    
                except Exception as e:
                    print(f"⚠️  Failed to load PyTorch model: {e}")
        
        print(f"✅ Found {len(mobile_models)} mobile-ready models")
        return mobile_models
    
    def display_model_menu(self, available_models):
        """
        Display interactive model selection menu
        """
        print("\n" + "=" * 60)
        print("📱 MOBILE MODEL SELECTION")
        print("=" * 60)
        
        print(f"{'#':<3} {'Model':<25} {'Type':<8} {'Size':<8} {'mAP@0.5':<8} {'Status'}")
        print("-" * 60)
        
        for i, model_name in enumerate(available_models, 1):
            model_info = self.models[model_name]
            model_type = model_info['type']
            size_mb = model_info['size_mb']
            
            # Get accuracy info if available
            if model_name in self.model_specs:
                map_50 = self.model_specs[model_name]['mAP@0.5']
                map_str = f"{map_50:.3f}"
                
                if map_50 >= 0.95:
                    status = "🏆 Excellent"
                elif map_50 >= 0.85:
                    status = "🎯 Good"
                elif map_50 >= 0.75:
                    status = "📱 Mobile"
                else:
                    status = "⚠️  Basic"
            else:
                map_str = "N/A"
                status = "📱 Mobile"
            
            print(f"{i:<3} {model_name:<25} {model_type:<8} {size_mb:<8.1f} {map_str:<8} {status}")
        
        print("-" * 60)
        return available_models
    
    def get_test_images(self, num_images=10):
        """
        Get available test images
        """
        test_dirs = [
            Path("model_training/data/images/val"),
            Path("model_training/data/positive")
        ]
        
        test_images = []
        
        for test_dir in test_dirs:
            if test_dir.exists():
                images = list(test_dir.glob("*.jpg"))
                test_images.extend(images[:num_images])
                if len(test_images) >= num_images:
                    break
        
        return test_images[:num_images]
    
    def display_image_menu(self, test_images):
        """
        Display image selection menu
        """
        print("\n" + "=" * 60)
        print("🖼️  TEST IMAGE SELECTION")
        print("=" * 60)
        
        print(f"{'#':<3} {'Image Name':<50} {'Size'}")
        print("-" * 60)
        
        for i, img_path in enumerate(test_images, 1):
            # Get image dimensions
            image = cv2.imread(str(img_path))
            if image is not None:
                height, width = image.shape[:2]
                size_str = f"{width}x{height}"
            else:
                size_str = "Unknown"
            
            print(f"{i:<3} {img_path.name:<50} {size_str}")
        
        print("-" * 60)
        return test_images
    
    def preprocess_image_for_tflite(self, image_path, target_size=(640, 640)):
        """
        Preprocess image for TFLite inference
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Store original for visualization
        original_image = image.copy()
        
        # Convert BGR to RGB for TFLite
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, target_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return np.expand_dims(image_normalized, axis=0), original_image
    
    def postprocess_tflite_output(self, output, image_width, image_height, conf_threshold=0.25):
        """
        Post-process TFLite YOLO output
        """
        detections = output[0].transpose()  # [8400, 5]
        
        results = []
        
        for detection in detections:
            x_center, y_center, width, height, confidence = detection
            
            if confidence < conf_threshold:
                continue
            
            # Scale coordinates to original image size
            x_center_px = x_center * image_width
            y_center_px = y_center * image_height
            width_px = width * image_width
            height_px = height * image_height
            
            # Convert to [x1, y1, x2, y2] format
            x1 = max(0, x_center_px - width_px / 2)
            y1 = max(0, y_center_px - height_px / 2)
            x2 = min(image_width, x_center_px + width_px / 2)
            y2 = min(image_height, y_center_px + height_px / 2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class': 'number'
            })
        
        return results
    
    def run_tflite_inference(self, model_name, image_path):
        """
        Run inference using TFLite model
        """
        model_info = self.models[model_name]
        interpreter = model_info['interpreter']
        input_details = model_info['input_details']
        output_details = model_info['output_details']
        
        # Preprocess image
        image_input, original_image = self.preprocess_image_for_tflite(image_path)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Get original image dimensions
        height, width = original_image.shape[:2]
        
        # Post-process output
        detections = self.postprocess_tflite_output(output, width, height, conf_threshold=0.1)
        
        return detections, original_image
    
    def run_pytorch_inference(self, model_name, image_path):
        """
        Run inference using PyTorch model
        """
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Run inference
        results = model(str(image_path), verbose=False)
        
        # Load original image
        original_image = cv2.imread(str(image_path))
        
        # Extract detections
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            detection_boxes = boxes.xyxy.cpu().numpy()
            detection_scores = boxes.conf.cpu().numpy()
            detection_classes = boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(detection_boxes, detection_scores, detection_classes):
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(score),
                    'class': 'number'
                })
        
        return detections, original_image
    
    def load_ground_truth(self, image_path):
        """
        Load ground truth annotations if available
        """
        # Try to find corresponding annotation file
        image_path = Path(image_path)
        
        # Check multiple possible label locations
        possible_label_paths = [
            image_path.parent.parent / 'labels' / image_path.parent.name / f"{image_path.stem}.txt",
            image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"
        ]
        
        for label_path in possible_label_paths:
            if label_path.exists():
                # Load image to get dimensions
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                    
                img_height, img_width = image.shape[:2]
                
                # Parse YOLO format annotations
                gt_boxes = []
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            
                            # Convert to [x1, y1, x2, y2] format
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            gt_boxes.append({
                                'bbox': [x1, y1, x2, y2],
                                'class': 'number'
                            })
                
                return gt_boxes
        
        return []
    
    def visualize_results(self, image, detections, ground_truth=None, model_name="", image_name=""):
        """
        Create visual overlay of detection results
        """
        # Convert BGR to RGB for matplotlib
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        ax.set_title(f"Detection Results: {model_name}\nImage: {image_name}", fontsize=14)
        ax.axis('off')
        
        # Draw ground truth boxes (if available)
        if ground_truth:
            for gt in ground_truth:
                x1, y1, x2, y2 = gt['bbox']
                width = x2 - x1
                height = y2 - y1
                
                # Green boxes for ground truth
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=3, edgecolor='lime', facecolor='none', 
                               linestyle='--', label='Ground Truth' if gt == ground_truth[0] else "")
                ax.add_patch(rect)
        
        # Draw predictions
        if detections:
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                width = x2 - x1
                height = y2 - y1
                confidence = det['confidence']
                
                # Color based on confidence
                if confidence >= 0.7:
                    color = 'red'  # High confidence
                elif confidence >= 0.5:
                    color = 'orange'  # Medium confidence
                else:
                    color = 'yellow'  # Low confidence
                
                # Draw bounding box
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none',
                               label=f'Prediction (conf ≥ {confidence:.2f})' if i == 0 else "")
                ax.add_patch(rect)
                
                # Add confidence text
                ax.text(x1, y1-5, f'{confidence:.3f}', 
                       fontsize=10, color=color, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add legend
        if ground_truth or detections:
            ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # Add model info
        info_text = []
        if model_name in self.model_specs:
            specs = self.model_specs[model_name]
            info_text.append(f"mAP@0.5: {specs['mAP@0.5']:.3f}")
            info_text.append(f"Size: {specs['size_mb']:.1f}MB")
        
        if info_text:
            ax.text(0.02, 0.98, '\n'.join(info_text), 
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Add detection summary
        summary = f"Detections: {len(detections)}"
        if ground_truth:
            summary += f" | Ground Truth: {len(ground_truth)}"
        
        ax.text(0.02, 0.02, summary, 
               transform=ax.transAxes, fontsize=10, 
               verticalalignment='bottom',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def run_interactive_demo(self):
        """
        Run interactive demo
        """
        print("🚀 MOBILE MODEL VISUAL INFERENCE DEMO")
        print("=" * 60)
        
        # Discover models
        available_models = self.discover_models()
        
        if not available_models:
            print("❌ No mobile-ready models found!")
            print("💡 Run export_detector.py first to create TFLite models")
            return
        
        # Model selection
        self.display_model_menu(available_models)
        
        while True:
            try:
                choice = input(f"\n👆 Select model (1-{len(available_models)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print("👋 Goodbye!")
                    return
                
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(available_models):
                    selected_model = available_models[model_idx]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(available_models)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q'")
        
        print(f"\n✅ Selected model: {selected_model}")
        
        # Image selection
        test_images = self.get_test_images()
        
        if not test_images:
            print("❌ No test images found!")
            return
        
        self.display_image_menu(test_images)
        
        while True:
            try:
                choice = input(f"\n👆 Select image (1-{len(test_images)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print("👋 Goodbye!")
                    return
                
                image_idx = int(choice) - 1
                if 0 <= image_idx < len(test_images):
                    selected_image = test_images[image_idx]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(test_images)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q'")
        
        print(f"\n✅ Selected image: {selected_image.name}")
        
        # Run inference
        print(f"\n🔍 Running inference with {selected_model}...")
        
        try:
            model_info = self.models[selected_model]
            
            if model_info['type'] == 'tflite':
                detections, original_image = self.run_tflite_inference(selected_model, selected_image)
            else:
                detections, original_image = self.run_pytorch_inference(selected_model, selected_image)
            
            # Load ground truth
            ground_truth = self.load_ground_truth(selected_image)
            
            print(f"✅ Inference complete!")
            print(f"   Detections found: {len(detections)}")
            if ground_truth:
                print(f"   Ground truth boxes: {len(ground_truth)}")
            
            # Show results
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                print(f"   Detection {i+1}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], conf={conf:.3f}")
            
            # Visualize
            print(f"\n📊 Generating visualization...")
            
            fig = self.visualize_results(
                original_image, 
                detections, 
                ground_truth, 
                selected_model, 
                selected_image.name
            )
            
            # Save and show
            output_path = Path(f"model_training/results/visual_demo_{selected_model}_{selected_image.stem}.png")
            output_path.parent.mkdir(exist_ok=True)
            
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved: {output_path}")
            
            # Show plot
            plt.show()
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Ask for another run
        while True:
            choice = input(f"\n🔄 Run another test? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                self.run_interactive_demo()
                return
            elif choice in ['n', 'no']:
                print("👋 Thanks for using the demo!")
                return
            else:
                print("❌ Please enter 'y' or 'n'")


def main():
    """
    Main demo entry point
    """
    try:
        demo = VisualInferenceDemo()
        demo.run_interactive_demo()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Demo interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()