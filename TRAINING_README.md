# YOLO Chess Detection Training

Training scripts for detecting chess pieces using the latest YOLO models with GPU acceleration.

## Quick Start

All training/inference utilities are now unified in a single entrypoint: `stages.py`.

### 1. Check System Requirements
```bash
python stages.py --stage 8
```
This will verify your GPU, PyTorch, and dataset setup.

### 2. Install Dependencies
```bash
pip install ultralytics torch torchvision opencv-python
```

For GPU support (NVIDIA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Train the Model
```bash
python stages.py --stage 11 --model yolo11s.pt --epochs 30 --imgsz 416 --batch 4 --workers 2
```
Or use YOLOv8 version:
```bash
python stages.py --stage 12 --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16 --workers 8
```

Prepare dataset (if needed):
```bash
python stages.py --stage 10 --labels-subdir labels_nlp_2026-02-10-11-44-39
```

### 4. Run Inference
```bash
python stages.py --stage 15 --weights runs/detect/train_chess/weights/best.pt --image preprocessed_frames/frame_001.jpg
```

Video inference:
```bash
python stages.py --stage 16 --weights runs/detect/train/weights/best.pt --source videos/chess.mp4 --out runs/detect/predict_custom --no-show
```

Resume training from checkpoint:
```bash
python stages.py --stage 13 --checkpoint runs/detect/train_chess/weights/last.pt
```

Export trained model:
```bash
python stages.py --stage 14 --weights runs/detect/train_chess/weights/best.pt --export-formats onnx,torchscript,tflite
```

## Model Options

### YOLOv11 Models (Latest - 2024/2025)
- **yolo11n.pt** - Nano (fastest, lowest accuracy)
- **yolo11s.pt** - Small (recommended for most cases)
- **yolo11m.pt** - Medium (good accuracy, moderate speed)
- **yolo11l.pt** - Large (high accuracy, slower)
- **yolo11x.pt** - Extra Large (highest accuracy, slowest)

### YOLOv8 Models (Stable)
- **yolov8n.pt** - Nano
- **yolov8s.pt** - Small 
- **yolov8m.pt** - Medium
- **yolov8l.pt** - Large
- **yolov8x.pt** - Extra Large

## Batch Size Guidelines

Based on GPU VRAM:
- **12+ GB VRAM**: batch_size = 32-64
- **8-12 GB VRAM**: batch_size = 16-32
- **4-8 GB VRAM**: batch_size = 8-16
- **< 4 GB VRAM**: batch_size = 4-8
- **CPU only**: batch_size = 2-4

## Training Parameters

Key parameters for stage 11/12 in `stages.py`:

```python
EPOCHS = 100          # Number of training epochs
IMG_SIZE = 640        # Input image size (640, 1280)
BATCH_SIZE = 16       # Batch size (adjust for your GPU)
optimizer = 'AdamW'   # Optimizer (AdamW, Adam, SGD)
lr0 = 0.01           # Initial learning rate
```

## Performance Optimizations

1. **Automatic Mixed Precision (AMP)**: Enabled by default
   - Faster training with minimal accuracy loss
   - Reduces memory usage

2. **Image Caching**: Enabled by default
   - Caches images in RAM for faster loading
   - Requires sufficient RAM

3. **Multi-worker Data Loading**: 8 workers by default
   - Adjust based on CPU cores

4. **Early Stopping**: Patience = 50 epochs
   - Stops training if no improvement

## Output Files

After training, find results in:
```
runs/detect/train_chess/
├── weights/
│   ├── best.pt        # Best model (use this for inference)
│   └── last.pt        # Last epoch checkpoint
├── results.csv        # Training metrics
├── results.png        # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── PR_curve.png
```

## Dataset Structure

```
datasets/chess_yolo/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Classes (15 classes in FEN notation)

0. cb - Chessboard
1. player1
2. player2
3. p - Black Pawn
4. r - Black Rook
5. n - Black Knight
6. b - Black Bishop
7. q - Black Queen
8. k - Black King
9. P - White Pawn
10. R - White Rook
11. N - White Knight
12. B - White Bishop
13. Q - White Queen
14. K - White King

## Tips for Best Results

1. **Use GPU**: Training is 10-50x faster on GPU
2. **More data**: Add more annotated images if possible
3. **Augmentation**: Enabled by default to improve generalization
4. **Model size**: Start with 'yolo11s.pt' for best balance
5. **Epochs**: 100 epochs is usually sufficient; use early stopping
6. **Learning rate**: Default 0.01 works well; lower if unstable
7. **Image size**: 640 is standard; increase to 1280 for better accuracy

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `BATCH_SIZE = 8`
- Use smaller model: `yolo11n.pt`
- Reduce image size: `IMG_SIZE = 416`

### Slow Training
- Enable GPU if available
- Increase batch size if GPU allows
- Use smaller model for faster iteration
- Enable caching: `cache=True`

### Poor Accuracy
- Train longer: increase `EPOCHS`
- Use larger model: `yolo11m.pt` or `yolo11l.pt`
- Check dataset quality and annotations
- Adjust augmentation parameters
- Add more training data

## Validation Metrics

- **mAP50**: Mean Average Precision at IoU=0.50 (higher is better)
- **mAP50-95**: mAP at IoU thresholds from 0.50 to 0.95 (more strict)
- **Precision**: How many detections are correct
- **Recall**: How many ground truth objects are detected

Good results: mAP50 > 0.90, mAP50-95 > 0.70

## License

This project uses Ultralytics YOLO which is licensed under AGPL-3.0.
