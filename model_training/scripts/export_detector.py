"""
Export YOLOv8 detection model to TensorFlow Lite format

Converts the trained YOLOv8 nano model to TFLite with multiple quantization options
for mobile deployment.
"""

import sys
import os
from pathlib import Path
from ultralytics import YOLO
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def export_model_to_tflite(model_path, dataset_yaml, output_dir, formats=None):
    """
    Export YOLOv8 model to multiple TFLite formats
    
    Args:
        model_path: Path to the trained .pt model
        dataset_yaml: Path to dataset configuration for calibration
        output_dir: Directory to save exported models
        formats: List of export formats to generate
    """
    if formats is None:
        formats = ['fp32', 'fp16', 'int8']
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    export_results = {}
    
    for format_type in formats:
        print(f"\n{'='*50}")
        print(f"Exporting to TFLite {format_type.upper()} format")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            if format_type == 'fp32':
                # Export FP32 TFLite (no quantization)
                result = model.export(
                    format='tflite',
                    imgsz=640,
                    data=dataset_yaml,
                    optimize=True
                )
                
            elif format_type == 'fp16':
                # Export FP16 TFLite (half precision)
                result = model.export(
                    format='tflite',
                    imgsz=640,
                    data=dataset_yaml,
                    half=True,
                    optimize=True
                )
                
            elif format_type == 'int8':
                # Export INT8 TFLite (8-bit quantization)
                result = model.export(
                    format='tflite',
                    imgsz=640,
                    data=dataset_yaml,
                    int8=True,
                    optimize=True
                )
            
            export_time = time.time() - start_time
            
            # Get the exported file path
            exported_file = Path(result)
            
            # Move to organized output directory with descriptive name
            new_name = f"detector_{format_type}.tflite"
            target_path = output_dir / new_name
            
            if exported_file.exists():
                exported_file.rename(target_path)
                
                # Get file size
                file_size_mb = target_path.stat().st_size / (1024 * 1024)
                
                export_results[format_type] = {
                    'path': str(target_path),
                    'size_mb': file_size_mb,
                    'export_time': export_time,
                    'success': True
                }
                
                print(f"✅ Export successful!")
                print(f"📁 Saved to: {target_path}")
                print(f"📏 File size: {file_size_mb:.2f} MB")
                print(f"⏱️  Export time: {export_time:.1f} seconds")
                
            else:
                print(f"❌ Export failed - file not found: {exported_file}")
                export_results[format_type] = {
                    'success': False,
                    'error': 'File not found after export'
                }
                
        except Exception as e:
            print(f"❌ Export failed: {str(e)}")
            export_results[format_type] = {
                'success': False,
                'error': str(e),
                'export_time': time.time() - start_time
            }
    
    return export_results


def analyze_export_results(results):
    """
    Analyze and display export results summary
    """
    print(f"\n{'='*60}")
    print(f"EXPORT RESULTS SUMMARY")
    print(f"{'='*60}")
    
    successful_exports = []
    failed_exports = []
    
    for format_type, result in results.items():
        if result.get('success', False):
            successful_exports.append((format_type, result))
            print(f"✅ {format_type.upper()} TFLite:")
            print(f"   📁 Path: {result['path']}")
            print(f"   📏 Size: {result['size_mb']:.2f} MB")
            print(f"   ⏱️  Time: {result['export_time']:.1f}s")
        else:
            failed_exports.append((format_type, result))
            print(f"❌ {format_type.upper()} TFLite: FAILED")
            print(f"   🚫 Error: {result.get('error', 'Unknown error')}")
        print()
    
    # Check size requirements
    print(f"📋 Size Analysis:")
    target_size = 6.0  # MB
    
    for format_type, result in successful_exports:
        size_mb = result['size_mb']
        if size_mb <= target_size:
            print(f"   ✅ {format_type.upper()}: {size_mb:.2f} MB (meets <{target_size}MB target)")
        else:
            print(f"   ⚠️  {format_type.upper()}: {size_mb:.2f} MB (exceeds {target_size}MB target)")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    if any(r['size_mb'] <= target_size for _, r in successful_exports):
        best_format = min(successful_exports, key=lambda x: x[1]['size_mb'])
        print(f"   • Use {best_format[0].upper()} format for production ({best_format[1]['size_mb']:.2f} MB)")
    
    if failed_exports:
        print(f"   • Check failed exports: {', '.join(f[0] for f in failed_exports)}")
        print(f"   • Consider fallback to successful formats")
    
    return successful_exports, failed_exports


def main():
    """
    Main export script
    """
    try:
        print(f"{'='*60}")
        print(f"YOLOV8 DETECTION MODEL TFLITE EXPORT")
        print(f"{'='*60}")
        
        # Define paths
        model_path = Path("model_training/models/building_number_detector/weights/best.pt")
        dataset_yaml = Path("model_training/data/dataset.yaml")
        output_dir = Path("model_training/models/tflite_exports")
        
        # Validate inputs
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
        
        print(f"📂 Model: {model_path}")
        print(f"📂 Dataset: {dataset_yaml}")
        print(f"📂 Output: {output_dir}")
        print()
        
        # Export models
        formats_to_export = ['fp32', 'fp16', 'int8']
        results = export_model_to_tflite(
            model_path=model_path,
            dataset_yaml=dataset_yaml,
            output_dir=output_dir,
            formats=formats_to_export
        )
        
        # Analyze results
        successful, failed = analyze_export_results(results)
        
        # Final status
        if successful:
            print(f"\n🎉 Export completed successfully!")
            print(f"📋 {len(successful)} formats exported successfully")
            print(f"📁 Models saved in: {output_dir}")
            
            if failed:
                print(f"⚠️  {len(failed)} formats failed - check logs above")
        else:
            print(f"\n❌ All exports failed!")
            print(f"🔍 Check error messages above for troubleshooting")
            return 1
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Test TFLite model inference")
        print(f"   2. Benchmark model performance")
        print(f"   3. Validate detection accuracy")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n🛑 Export interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())