# Model Training Directory

This directory contains all components for training the building number detection model using YOLOv8.

## Directory Structure

```
model_training/
├── data/                    # Training data
│   └── dataset/            # YOLO format dataset (ready for training)
│       ├── images/
│       │   ├── train/      # Training images (912 images)
│       │   └── val/        # Validation images (228 images)
│       ├── labels/
│       │   ├── train/      # Training labels (912 files)
│       │   └── val/        # Validation labels (228 files)
│       └── dataset.yaml    # YOLO dataset configuration
├── models/                 # Trained models and outputs
│   └── building_number_detector/  # Training outputs from YOLOv8
├── scripts/                # Training and dataset preparation scripts
│   ├── train_detector.py           # Main training script
│   ├── add_training_images.py      # Add new images to existing dataset
│   ├── create_yolo_dataset.py      # Create YOLO dataset from raw data
│   ├── copy_training_images.py     # Copy images from validation results
│   └── create_bbox_annotations.py  # Generate bounding box annotations
├── archive/                # Archived intermediate data (for reference)
│   ├── positive/           # Original positive samples (940 images)
│   ├── negative/           # Original negative samples (3405 images)
│   └── annotations/        # Original YOLO format annotations (940 files)
└── README.md               # This file
```

## Usage

### Training the Model

To train the building number detection model:

```bash
cd model_training/scripts
python train_detector.py
```

The script will:
- Use the dataset from `../data/dataset/`
- Save trained models to `../models/building_number_detector/`
- Generate training plots and metrics

### Adding New Training Images

To add a new batch of images to the existing dataset:

```bash
cd model_training/scripts
python add_training_images.py -i /path/to/new/images
```

This script will:
- Detect numbers in new images using EasyOCR
- Classify images as positive (containing numbers) or negative samples
- Create YOLO format annotations for detected numbers
- Add images to the existing dataset with proper train/val split
- Clean up cache files for fresh training

### Dataset Statistics

**Current Dataset (in `data/dataset/`):**
- **Training**: 912 images, 912 labels
- **Validation**: 228 images, 228 labels
- **Classes**: 1 (building numbers)
- **Format**: YOLOv8 compatible
- **Total Size**: ~94MB

**Archive Data (reference only):**
- **Positive samples**: 940 images (~75MB)
- **Negative samples**: 3405 images (~265MB)
- **Annotations**: 940 YOLO format files (~3.7MB)
- **Archive Total**: ~344MB

## Key Changes

This structure represents a cleanup from the previous organization:

### What Changed:
1. **Consolidated Dataset**: Single `data/dataset/` location instead of scattered intermediate directories
2. **Organized Scripts**: All training scripts moved to `scripts/` directory
3. **Archived Intermediates**: Original processing directories moved to `archive/` for reference
4. **Updated Paths**: All scripts updated to work with the new relative paths
5. **Removed Duplicates**: Eliminated ~169MB of duplicate image data

### Benefits:
- **344MB space savings** by removing duplicate data
- **Clearer organization** with single dataset source of truth
- **Easier maintenance** with all scripts in one location
- **Better documentation** of the training pipeline
- **Simpler workflow** for adding new training data

## Training Pipeline

The complete training pipeline follows these steps:

1. **Data Collection**: Gather building/door images
2. **OCR Detection**: Use `building_number_detector.py` to detect numbers
3. **Organization**: Use `organize_by_digits.py` to sort by detected digits
4. **Sample Preparation**: Use `copy_training_images.py` to create positive/negative samples
5. **Annotation**: Use `create_bbox_annotations.py` to generate bounding boxes
6. **Dataset Creation**: Use `create_yolo_dataset.py` to create YOLO format dataset
7. **Training**: Use `train_detector.py` to train the YOLOv8 model
8. **Expansion**: Use `add_training_images.py` to add new images incrementally

## Model Configuration

- **Architecture**: YOLOv8 Nano (yolov8n.pt)
- **Image Size**: 640x640
- **Classes**: 1 (building number)
- **Training Parameters**:
  - Epochs: 50
  - Batch Size: 16
  - Patience: 10 (early stopping)
  - Device: MPS (Apple Silicon) / CUDA / CPU

## Output Files

After training, the following files will be generated in `models/building_number_detector/`:

- `weights/best.pt` - Best performing model weights
- `weights/last.pt` - Final epoch weights
- `results.png` - Training metrics plots
- `confusion_matrix.png` - Model performance matrix
- Various curve plots (P, R, F1, PR)

## Requirements

- Python 3.8+
- ultralytics (YOLOv8)
- opencv-python
- easyocr
- Other dependencies from project requirements

## Notes

- The archive directory contains the original intermediate data for reference
- All paths in scripts are relative to the script location
- Cache files are automatically cleaned when adding new data
- The dataset maintains a 80/20 train/validation split