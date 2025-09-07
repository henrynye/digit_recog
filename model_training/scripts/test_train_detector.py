#!/usr/bin/env python3
"""
Test training script with small dataset for quick debugging
"""

from ultralytics import YOLO
from pathlib import Path
import torch


def test_train_detection_model():
    """
    Test YOLOv8 nano model on small test dataset
    """
    # Check available devices
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU acceleration")
    else:
        device = "cpu"
        print("Using CPU training")

    # Define paths
    dataset_yaml = Path("model_training/data/test_dataset.yaml")
    output_dir = Path("model_training/test_models")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate dataset exists
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Test dataset configuration not found: {dataset_yaml}")

    print(f"Testing YOLOv8 nano on test dataset: {dataset_yaml}")
    print(f"Output directory: {output_dir}")

    # Initialize model
    model = YOLO("yolo11n.pt")  # Load pretrained YOLOv8 nano

    # Configure training parameters for quick test
    train_args = {
        "data": str(dataset_yaml),
        "epochs": 10,  # Very few epochs for quick testing
        "imgsz": 640,  # Image size
        "batch": 4,  # Small batch size for test
        "device": "cpu",  # Use available device
        "patience": 5,  # Early stopping patience
        "save_period": 1,  # Save checkpoint every epoch
        "val": True,  # Enable validation
        "plots": True,  # Disable training plots to avoid KeyError
        "cache": False,  # Don't cache images (to save memory)
        "workers": 2,  # Fewer workers for test
        "project": str(output_dir),  # Save to test models directory
        "name": "test_detector",  # Experiment name
        "exist_ok": True,  # Overwrite existing experiment
        "verbose": True,  # Verbose training output
        "amp": False,  # Disable AMP to fix MPS validation issues
        "seed": 42,  # Reproducible results
    }

    print(f"\nTest training configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")

    print(f"\nStarting test training...")
    print("=" * 60)

    # Start training
    try:
        results = model.train(**train_args)

        print("\n" + "=" * 60)
        print("TEST TRAINING COMPLETED!")
        print("=" * 60)

        # Find the best model
        best_model_path = output_dir / "test_detector" / "weights" / "best.pt"
        last_model_path = output_dir / "test_detector" / "weights" / "last.pt"

        if best_model_path.exists():
            print(f"\n✅ Test model saved: {best_model_path}")
        if last_model_path.exists():
            print(f"📁 Last model saved: {last_model_path}")

        print(f"\n📊 Check training plots in: {output_dir}/test_detector/")
        print(f"🎯 If this works, try the full dataset!")

        return results

    except Exception as e:
        print(f"Test training failed: {e}")
        raise


def main():
    """Main entry point"""
    try:
        print("=" * 60)
        print("TEST TRAINING - BUILDING NUMBER DETECTION")
        print("=" * 60)

        # Test training with small dataset
        results = test_train_detection_model()

        print(f"\n🎉 Test training completed successfully!")
        print(f"💡 Next: If successful, run full training with complete dataset")

    except KeyboardInterrupt:
        print(f"\n🛑 Test training interrupted by user")
    except Exception as e:
        print(f"\n❌ Test training failed: {e}")
        raise


if __name__ == "__main__":
    main()
