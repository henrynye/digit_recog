"""
Train YOLOv8 nano detection model for building number detection

Fine-tunes YOLOv8 nano on building number detection dataset.
"""

from ultralytics import YOLO
from pathlib import Path
import torch

def train_detection_model():
    """
    Train YOLOv8 nano model on building number detection dataset
    """
    # Check available devices
    if torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
        print("Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA GPU acceleration")
    else:
        device = 'cpu'
        print("Using CPU training")
    
    # Define paths
    dataset_yaml = Path("../data/dataset/dataset.yaml")
    output_dir = Path("../models")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate dataset exists
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {dataset_yaml}")
    
    print(f"Training YOLOv8 nano on dataset: {dataset_yaml}")
    print(f"Output directory: {output_dir}")
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 nano
    
    # Configure training parameters
    train_args = {
        'data': str(dataset_yaml),
        'epochs': 50,
        'imgsz': 640,           # Image size
        'batch': 16,            # Batch size
        'device': device,       # Use available device
        'patience': 10,         # Early stopping patience
        'save_period': 10,      # Save checkpoint every 10 epochs
        'val': True,            # Enable validation
        'plots': True,          # Generate training plots
        'cache': False,         # Don't cache images (to save memory)
        'workers': 4,           # Number of dataloader workers
        'project': str(output_dir),  # Save to our models directory
        'name': 'building_number_detector',  # Experiment name
        'exist_ok': True,       # Overwrite existing experiment
        'verbose': True,        # Verbose training output
        'seed': 42,             # Reproducible results
    }
    
    print(f"\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    print(f"\nStarting training...")
    print(f"Target: >0.7 mAP@0.5 on validation set")
    print(f"Early stopping: {train_args['patience']} epochs without improvement")
    print("-" * 60)
    
    # Start training
    try:
        results = model.train(**train_args)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        
        # Print final results
        if hasattr(results, 'results_dict'):
            final_results = results.results_dict
            print(f"Final Results:")
            for key, value in final_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        # Find the best model
        best_model_path = output_dir / "building_number_detector" / "weights" / "best.pt"
        last_model_path = output_dir / "building_number_detector" / "weights" / "last.pt"
        
        if best_model_path.exists():
            print(f"\n✅ Best model saved: {best_model_path}")
        if last_model_path.exists():
            print(f"📁 Last model saved: {last_model_path}")
        
        # Check if target accuracy achieved
        # Note: Exact metric extraction depends on YOLOv8 version
        print(f"\n📊 Check training plots in: {output_dir}/building_number_detector/")
        print(f"🎯 Target achieved if mAP@0.5 > 0.7")
        
        return results
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

def validate_training_results():
    """
    Validate that training completed successfully and check results
    """
    output_dir = Path("model_training/models/building_number_detector")
    
    if not output_dir.exists():
        print("❌ Training output directory not found!")
        return False
    
    # Check for model files
    best_model = output_dir / "weights" / "best.pt"
    last_model = output_dir / "weights" / "last.pt"
    
    if best_model.exists():
        print(f"✅ Best model found: {best_model}")
        print(f"📈 Model size: {best_model.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("❌ Best model not found!")
        return False
    
    if last_model.exists():
        print(f"✅ Last model found: {last_model}")
    
    # Check for training plots
    plots_dir = output_dir
    plot_files = ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']
    
    found_plots = []
    for plot_file in plot_files:
        plot_path = plots_dir / plot_file
        if plot_path.exists():
            found_plots.append(plot_file)
    
    if found_plots:
        print(f"✅ Training plots generated: {', '.join(found_plots)}")
    
    print(f"\n🎯 Next step: Check {output_dir}/results.png for training metrics")
    print(f"💡 Look for mAP@0.5 > 0.7 in validation results")
    
    return True

def main():
    """Main entry point"""
    try:
        print("="*60)
        print("BUILDING NUMBER DETECTION MODEL TRAINING")
        print("="*60)
        
        # Train the model
        results = train_detection_model()
        
        # Validate results
        if validate_training_results():
            print(f"\n🎉 Training completed successfully!")
            print(f"📋 Review training metrics and proceed to Step 2.4 (Export Detection Model)")
        else:
            print(f"\n⚠️ Training completed but validation failed")
            print(f"🔍 Check output directory for issues")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()