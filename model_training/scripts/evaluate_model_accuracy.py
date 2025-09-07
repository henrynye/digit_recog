"""
Model Accuracy Evaluation Script

Evaluates and compares detection accuracy metrics (mAP, precision, recall) 
across all model formats using the validation dataset.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
import yaml
from collections import defaultdict
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class DetectionMetrics:
    """
    Calculate object detection metrics including mAP, precision, recall
    """
    
    def __init__(self, iou_thresholds=None, conf_threshold=0.001):
        """
        Initialize metrics calculator
        
        Args:
            iou_thresholds: List of IoU thresholds for mAP calculation
            conf_threshold: Minimum confidence threshold for predictions
        """
        if iou_thresholds is None:
            self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        else:
            self.iou_thresholds = iou_thresholds
            
        self.conf_threshold = conf_threshold
        
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1, box2: Boxes in format [x1, y1, x2, y2]
        
        Returns:
            IoU value between 0 and 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def non_max_suppression(self, boxes, scores, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            boxes: Array of boxes in format [x1, y1, x2, y2]
            scores: Array of confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return []
            
        # Sort by confidence score (descending)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Keep the highest scoring box
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
                
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_indices = indices[1:]
            
            ious = np.array([
                self.calculate_iou(current_box, boxes[i]) 
                for i in remaining_indices
            ])
            
            # Keep only boxes with IoU < threshold
            indices = remaining_indices[ious < iou_threshold]
            
        return keep
    
    def calculate_ap(self, precisions, recalls):
        """
        Calculate Average Precision from precision-recall curve
        
        Args:
            precisions: Array of precision values
            recalls: Array of recall values
            
        Returns:
            Average precision value
        """
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls = recalls[sorted_indices]
        precisions = precisions[sorted_indices]
        
        # Add endpoints for integration
        recalls = np.concatenate([[0], recalls, [1]])
        precisions = np.concatenate([[0], precisions, [0]])
        
        # Make precision non-decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
            
        # Find recall threshold changes
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        
        # Calculate AP using trapezoidal rule
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        
        return ap
    
    def evaluate_predictions(self, predictions, ground_truths):
        """
        Evaluate predictions against ground truths
        
        Args:
            predictions: List of predictions for each image
                Format: [{'boxes': [[x1,y1,x2,y2], ...], 'scores': [conf, ...], 'labels': [cls, ...]}]
            ground_truths: List of ground truths for each image
                Format: [{'boxes': [[x1,y1,x2,y2], ...], 'labels': [cls, ...]}]
                
        Returns:
            Dictionary with evaluation metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match number of ground truths")
            
        results = {}
        
        # For each IoU threshold
        for iou_thresh in self.iou_thresholds:
            tp_sum = 0
            fp_sum = 0
            fn_sum = 0
            all_scores = []
            all_matches = []
            
            for pred, gt in zip(predictions, ground_truths):
                pred_boxes = np.array(pred.get('boxes', []))
                pred_scores = np.array(pred.get('scores', []))
                pred_labels = np.array(pred.get('labels', []))
                
                gt_boxes = np.array(gt.get('boxes', []))
                gt_labels = np.array(gt.get('labels', []))
                
                # Filter predictions by confidence threshold
                valid_preds = pred_scores >= self.conf_threshold
                pred_boxes = pred_boxes[valid_preds] if len(pred_boxes) > 0 else np.array([])
                pred_scores = pred_scores[valid_preds] if len(pred_scores) > 0 else np.array([])
                pred_labels = pred_labels[valid_preds] if len(pred_labels) > 0 else np.array([])
                
                # Apply NMS to predictions
                if len(pred_boxes) > 0:
                    keep = self.non_max_suppression(pred_boxes, pred_scores)
                    pred_boxes = pred_boxes[keep]
                    pred_scores = pred_scores[keep]
                    pred_labels = pred_labels[keep]
                
                # Match predictions to ground truths
                gt_matched = np.zeros(len(gt_boxes), dtype=bool)
                
                for i, (pred_box, pred_score, pred_label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    # Find best matching ground truth
                    for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_matched[j] or pred_label != gt_label:
                            continue
                            
                        iou = self.calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                    
                    # Record match result
                    all_scores.append(pred_score)
                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        all_matches.append(True)
                        gt_matched[best_gt_idx] = True
                        tp_sum += 1
                    else:
                        all_matches.append(False)
                        fp_sum += 1
                
                # Count unmatched ground truths as false negatives
                fn_sum += len(gt_boxes) - np.sum(gt_matched)
            
            # Calculate metrics for this IoU threshold
            precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
            recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate AP using precision-recall curve
            if len(all_scores) > 0:
                # Sort by confidence score (descending)
                sorted_indices = np.argsort(all_scores)[::-1]
                sorted_matches = np.array(all_matches)[sorted_indices]
                
                # Calculate precision and recall at each threshold
                tp_cumsum = np.cumsum(sorted_matches)
                fp_cumsum = np.cumsum(~sorted_matches)
                
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
                recalls = tp_cumsum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else np.zeros_like(tp_cumsum)
                
                ap = self.calculate_ap(precisions, recalls)
            else:
                ap = 0.0
            
            results[f'iou_{iou_thresh:.2f}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ap': ap,
                'tp': tp_sum,
                'fp': fp_sum,
                'fn': fn_sum
            }
        
        # Calculate mAP@0.5 and mAP@0.5:0.95
        results['mAP@0.5'] = results['iou_0.50']['ap']
        results['mAP@0.5:0.95'] = np.mean([results[f'iou_{t:.2f}']['ap'] for t in self.iou_thresholds])
        
        # Overall metrics (using IoU=0.5)
        results['overall'] = results['iou_0.50']
        
        return results


class ModelEvaluator:
    """
    Evaluate and compare multiple model formats
    """
    
    def __init__(self, model_dir, dataset_config):
        """
        Initialize evaluator
        
        Args:
            model_dir: Directory containing models
            dataset_config: Path to dataset YAML configuration
        """
        self.model_dir = Path(model_dir)
        self.dataset_config = Path(dataset_config)
        self.models = {}
        self.dataset_info = None
        self.validation_data = []
        self.metrics_calculator = DetectionMetrics(conf_threshold=0.001)
        
    def load_dataset_config(self):
        """
        Load dataset configuration
        """
        with open(self.dataset_config, 'r') as f:
            self.dataset_info = yaml.safe_load(f)
        
        print(f"✅ Dataset config loaded: {self.dataset_info['nc']} classes")
        return self.dataset_info
    
    def load_validation_data(self):
        """
        Load validation dataset with annotations
        """
        if not self.dataset_info:
            self.load_dataset_config()
            
        # Get validation data path
        dataset_root = Path(self.dataset_info['path'])
        val_path = dataset_root / self.dataset_info['val']
        
        if not val_path.exists():
            raise FileNotFoundError(f"Validation path not found: {val_path}")
            
        # Check if val_path is a directory or file
        if val_path.is_dir():
            # If directory, list all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(val_path.glob(f"*{ext}"))
                image_paths.extend(val_path.glob(f"*{ext.upper()}"))
            image_paths = [str(p) for p in image_paths]
        else:
            # If file, read image paths from file
            with open(val_path, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
        
        # Load annotations
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.is_absolute():
                img_path = dataset_root / img_path
                
            # Find corresponding annotation file
            if val_path.is_dir():
                # Labels are in parallel directory structure
                # Images: data/images/val/ -> Labels: data/labels/val/
                labels_root = img_path.parent.parent.parent / 'labels' / img_path.parent.name
                label_path = labels_root / f"{img_path.stem}.txt"
            else:
                label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
            
            if not img_path.exists():
                print(f"⚠️  Image not found: {img_path}")
                continue
                
            # Load image to get dimensions
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️  Cannot read image: {img_path}")
                continue
                
            img_height, img_width = image.shape[:2]
            
            # Parse YOLO format annotations
            boxes = []
            labels = []
            
            if label_path.exists():
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
                            
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id)
            else:
                print(f"⚠️  No annotation file found: {label_path}")
            
            self.validation_data.append({
                'image_path': str(img_path),
                'boxes': boxes,
                'labels': labels,
                'width': img_width,
                'height': img_height
            })
        
        print(f"✅ Loaded {len(self.validation_data)} validation images")
        return len(self.validation_data)
    
    def load_models(self):
        """
        Load all available models
        """
        print("Loading models for evaluation...")
        
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
                model = YOLO(str(pytorch_path))
                pytorch_size = pytorch_path.stat().st_size / (1024 * 1024)
                
                self.models['pytorch_original'] = {
                    'type': 'pytorch',
                    'model': model,
                    'path': pytorch_path,
                    'size_mb': pytorch_size
                }
                
                print(f"✅ PyTorch: pytorch_original ({pytorch_size:.1f} MB)")
                
            except Exception as e:
                print(f"❌ Failed to load PyTorch model: {e}")
        
        print(f"\n📱 Loaded {len(self.models)} models total\n")
        return len(self.models)
    
    def preprocess_image_for_tflite(self, image_path, target_size=(640, 640)):
        """
        Preprocess image for TFLite inference
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, target_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return np.expand_dims(image_normalized, axis=0)
    
    def postprocess_tflite_output(self, output, image_width, image_height, conf_threshold=0.001):
        """
        Post-process TFLite YOLO output to extract detections
        
        Args:
            output: Raw TFLite model output [1, 5, 8400]
            image_width: Original image width
            image_height: Original image height
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary with boxes, scores, and labels
        """
        # Reshape output from [1, 5, 8400] to [8400, 5]
        detections = output[0].transpose()  # [8400, 5]
        
        boxes = []
        scores = []
        labels = []
        
        for detection in detections:
            x_center, y_center, width, height, confidence = detection
            
            if confidence < conf_threshold:
                continue
            
            # The model input was resized to 640x640, but coordinates are normalized [0,1]
            # Scale coordinates to original image size
            x_center_px = x_center * image_width
            y_center_px = y_center * image_height
            width_px = width * image_width
            height_px = height * image_height
            
            # Convert to [x1, y1, x2, y2] format
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2
            x2 = x_center_px + width_px / 2
            y2 = y_center_px + height_px / 2
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            labels.append(0)  # Single class (building number)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
    
    def evaluate_pytorch_model(self, model_name):
        """
        Evaluate PyTorch YOLO model
        """
        model_info = self.models[model_name]
        model = model_info['model']
        
        predictions = []
        
        print(f"Evaluating {model_name}...")
        
        for i, data in enumerate(self.validation_data):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(self.validation_data)}")
                
            # Run inference
            results = model(data['image_path'], verbose=False)
            
            # Extract predictions
            boxes = []
            scores = []
            labels = []
            
            if len(results) > 0 and results[0].boxes is not None:
                # Get detection data
                detection_boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                detection_scores = results[0].boxes.conf.cpu().numpy()
                detection_classes = results[0].boxes.cls.cpu().numpy()
                
                for box, score, cls in zip(detection_boxes, detection_scores, detection_classes):
                    boxes.append(box.tolist())
                    scores.append(float(score))
                    labels.append(int(cls))
            
            predictions.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        
        # Prepare ground truths
        ground_truths = [
            {'boxes': data['boxes'], 'labels': data['labels']}
            for data in self.validation_data
        ]
        
        # Calculate metrics
        metrics = self.metrics_calculator.evaluate_predictions(predictions, ground_truths)
        
        return metrics
    
    def evaluate_tflite_model(self, model_name):
        """
        Evaluate TFLite model
        """
        model_info = self.models[model_name]
        interpreter = model_info['interpreter']
        input_details = model_info['input_details']
        output_details = model_info['output_details']
        
        predictions = []
        
        print(f"Evaluating {model_name}...")
        
        for i, data in enumerate(self.validation_data):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(self.validation_data)}")
                
            # Preprocess image
            image_input = self.preprocess_image_for_tflite(data['image_path'])
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], image_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Post-process output
            detection_result = self.postprocess_tflite_output(
                output, data['width'], data['height']
            )
            
            predictions.append(detection_result)
        
        # Prepare ground truths
        ground_truths = [
            {'boxes': data['boxes'], 'labels': data['labels']}
            for data in self.validation_data
        ]
        
        # Calculate metrics
        metrics = self.metrics_calculator.evaluate_predictions(predictions, ground_truths)
        
        return metrics
    
    def evaluate_all_models(self):
        """
        Evaluate all loaded models
        """
        print("=" * 60)
        print("MODEL ACCURACY EVALUATION")
        print("=" * 60)
        
        results = {}
        
        for model_name, model_info in self.models.items():
            print(f"\nEvaluating {model_name}...")
            print("-" * 40)
            
            try:
                start_time = time.time()
                
                if model_info['type'] == 'pytorch':
                    metrics = self.evaluate_pytorch_model(model_name)
                elif model_info['type'] == 'tflite':
                    metrics = self.evaluate_tflite_model(model_name)
                else:
                    continue
                
                eval_time = time.time() - start_time
                
                results[model_name] = {
                    'metrics': metrics,
                    'model_info': model_info,
                    'eval_time': eval_time
                }
                
                print(f"✅ Completed in {eval_time:.1f}s")
                print(f"   mAP@0.5: {metrics['mAP@0.5']:.3f}")
                print(f"   mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.3f}")
                print(f"   Precision: {metrics['overall']['precision']:.3f}")
                print(f"   Recall: {metrics['overall']['recall']:.3f}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                results[model_name] = {
                    'error': str(e),
                    'model_info': model_info
                }
        
        return results
    
    def generate_accuracy_report(self, results):
        """
        Generate comprehensive accuracy comparison report
        """
        print("\n" + "=" * 80)
        print("MODEL ACCURACY COMPARISON REPORT")
        print("=" * 80)
        
        # Results table
        print(f"\n📊 Accuracy Metrics Comparison:")
        print(f"{'Model':<25} {'Type':<8} {'Size (MB)':<10} {'mAP@0.5':<10} {'mAP@0.5:0.95':<13} {'Precision':<10} {'Recall':<8} {'F1':<8}")
        print("-" * 95)
        
        successful_results = []
        baseline_metrics = None
        
        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                model_info = result['model_info']
                
                # Store baseline (PyTorch) for comparison
                if model_info['type'] == 'pytorch':
                    baseline_metrics = metrics
                
                successful_results.append((model_name, result))
                
                print(f"{model_name:<25} {model_info['type']:<8} {model_info['size_mb']:<10.1f} "
                      f"{metrics['mAP@0.5']:<10.3f} {metrics['mAP@0.5:0.95']:<13.3f} "
                      f"{metrics['overall']['precision']:<10.3f} {metrics['overall']['recall']:<8.3f} "
                      f"{metrics['overall']['f1']:<8.3f}")
            else:
                print(f"{model_name:<25} {'FAILED':<8} {'N/A':<10} {'N/A':<10} {'N/A':<13} {'N/A':<10} {'N/A':<8} {'N/A':<8}")
        
        # Accuracy preservation analysis
        if baseline_metrics:
            print(f"\n🎯 Accuracy Preservation Analysis (vs PyTorch baseline):")
            print(f"{'Model':<25} {'mAP@0.5 Drop':<15} {'Precision Drop':<15} {'Overall Status':<15}")
            print("-" * 70)
            
            for model_name, result in successful_results:
                if 'metrics' in result and result['model_info']['type'] != 'pytorch':
                    metrics = result['metrics']
                    
                    map_drop = baseline_metrics['mAP@0.5'] - metrics['mAP@0.5']
                    prec_drop = baseline_metrics['overall']['precision'] - metrics['overall']['precision']
                    
                    if map_drop <= 0.01 and prec_drop <= 0.01:
                        status = "✅ Excellent"
                    elif map_drop <= 0.03 and prec_drop <= 0.03:
                        status = "🟢 Good"
                    elif map_drop <= 0.05 and prec_drop <= 0.05:
                        status = "🟡 Acceptable"
                    else:
                        status = "🔴 Poor"
                    
                    print(f"{model_name:<25} {map_drop:<15.3f} {prec_drop:<15.3f} {status:<15}")
        
        # Trade-off analysis
        print(f"\n⚖️  Accuracy vs Size vs Speed Trade-offs:")
        print(f"{'Model':<25} {'Accuracy Score':<15} {'Size Score':<12} {'Overall Score':<15} {'Recommendation'}")
        print("-" * 85)
        
        recommendations = []
        
        for model_name, result in successful_results:
            if 'metrics' not in result:
                continue
                
            metrics = result['metrics']
            model_info = result['model_info']
            
            # Calculate normalized scores (0-1 scale)
            accuracy_score = metrics['mAP@0.5']  # Already 0-1
            size_score = max(0, 1 - (model_info['size_mb'] / 15))  # Smaller is better, cap at 15MB
            
            # Combined score with weighting
            overall_score = 0.6 * accuracy_score + 0.4 * size_score
            
            # Determine recommendation category
            if overall_score >= 0.8 and model_info['size_mb'] <= 6:
                recommendation = "🏆 Best Overall"
            elif accuracy_score >= 0.9:
                recommendation = "🎯 High Accuracy"
            elif model_info['size_mb'] <= 3:
                recommendation = "💾 Ultra Compact"
            elif model_info['size_mb'] <= 6:
                recommendation = "📱 Mobile Ready"
            else:
                recommendation = "⚠️  Large Size"
            
            recommendations.append((overall_score, model_name, recommendation, result))
            
            print(f"{model_name:<25} {accuracy_score:<15.3f} {size_score:<12.3f} {overall_score:<15.3f} {recommendation}")
        
        # Final recommendations
        if recommendations:
            recommendations.sort(reverse=True, key=lambda x: x[0])
            best_model = recommendations[0]
            
            print(f"\n💡 Deployment Recommendations:")
            print(f"   🥇 Best Overall Model: {best_model[1]} (Score: {best_model[0]:.3f})")
            
            # Find best in each category
            mobile_models = [(s, n, r, res) for s, n, r, res in recommendations 
                           if res['model_info']['size_mb'] <= 6]
            if mobile_models:
                print(f"   📱 Best Mobile Model: {mobile_models[0][1]} "
                      f"({mobile_models[0][3]['model_info']['size_mb']:.1f}MB, "
                      f"mAP: {mobile_models[0][3]['metrics']['mAP@0.5']:.3f})")
            
            high_acc_models = [(s, n, r, res) for s, n, r, res in recommendations 
                             if res['metrics']['mAP@0.5'] >= 0.9]
            if high_acc_models:
                print(f"   🎯 Best Accuracy Model: {high_acc_models[0][1]} "
                      f"(mAP: {high_acc_models[0][3]['metrics']['mAP@0.5']:.3f})")
        
        # Summary statistics
        total_models = len(results)
        successful_evals = len(successful_results)
        mobile_ready = len([r for _, r in successful_results 
                          if r['model_info']['size_mb'] <= 6 and r['metrics']['mAP@0.5'] >= 0.85])
        
        print(f"\n📋 Evaluation Summary:")
        print(f"   • Total models: {total_models}")
        print(f"   • Successful evaluations: {successful_evals}")
        print(f"   • Mobile-ready models (≤6MB, mAP≥0.85): {mobile_ready}")
        print(f"   • Validation images: {len(self.validation_data)}")
        
        return recommendations


def main():
    """
    Main evaluation script
    """
    try:
        print("=" * 60)
        print("MODEL ACCURACY EVALUATION SUITE")
        print("=" * 60)
        
        # Initialize evaluator
        model_dir = Path("model_training/models")
        dataset_config = Path("model_training/data/dataset.yaml")
        
        evaluator = ModelEvaluator(model_dir, dataset_config)
        
        # Load dataset and models
        evaluator.load_dataset_config()
        data_count = evaluator.load_validation_data()
        
        # For testing, limit to first 20 images
        if len(evaluator.validation_data) > 20:
            print(f"🚀 Quick test mode: Using first 20 images (out of {len(evaluator.validation_data)})")
            evaluator.validation_data = evaluator.validation_data[:20]
        
        model_count = evaluator.load_models()
        
        if data_count == 0:
            print("❌ No validation data found!")
            return 1
            
        if model_count == 0:
            print("❌ No models found!")
            return 1
        
        # Run evaluation
        results = evaluator.evaluate_all_models()
        
        # Generate report
        recommendations = evaluator.generate_accuracy_report(results)
        
        # Save results
        results_file = Path("model_training/results/accuracy_evaluation.json")
        results_file.parent.mkdir(exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            if 'metrics' in result:
                json_results[model_name] = {
                    'mAP@0.5': result['metrics']['mAP@0.5'],
                    'mAP@0.5:0.95': result['metrics']['mAP@0.5:0.95'],
                    'precision': result['metrics']['overall']['precision'],
                    'recall': result['metrics']['overall']['recall'],
                    'f1': result['metrics']['overall']['f1'],
                    'size_mb': result['model_info']['size_mb'],
                    'type': result['model_info']['type'],
                    'eval_time': result['eval_time']
                }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        if recommendations:
            print(f"\n🎉 Evaluation completed successfully!")
            print(f"✅ Best model: {recommendations[0][1]}")
        else:
            print(f"\n⚠️  Evaluation completed but no successful results")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n🛑 Evaluation interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())