# Bangla Chess Commentary: CV + NLP Pipeline

A practical computer-vision pipeline that analyzes chess videos, detects the board and pieces, and prepares the structure needed for Bangla commentary. All runnable steps are unified in a single entrypoint: `stages.py`.

> NLP commentary generation is **not implemented yet**. The CV pipeline and training utilities are in place; the language module is the next milestone.

## Why This Project

- Turn raw chess videos into structured game data.
- Detect boards and pieces frame-by-frame with YOLO.
- Build the foundation for Bangla commentary from CV outputs.

## Quick Start

All commands run from the root directory.

### 1. Check System and Dependencies

```bash
python stages.py --stage 8
```

### 2. Install Dependencies

```bash
pip install ultralytics torch torchvision opencv-python
```

For NVIDIA GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Prepare Dataset (if needed)

```bash
python stages.py --stage 10 --labels-subdir labels_nlp_2026-02-10-11-44-39
```

### 4. Train a Model

YOLO11:

```bash
python stages.py --stage 11 --model yolo11s.pt --epochs 30 --imgsz 416 --batch 4 --workers 2
```

YOLOv8:

```bash
python stages.py --stage 12 --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16 --workers 8
```

### 5. Run Inference

Single image:

```bash
python stages.py --stage 15 --weights runs/detect/train_chess/weights/best.pt --image preprocessed_frames/frame_001.jpg
```

Video:

```bash
python stages.py --stage 16 --weights runs/detect/train/weights/best.pt --source videos/chess.mp4 --out runs/detect/predict_custom --no-show
```

Resume training:

```bash
python stages.py --stage 13 --checkpoint runs/detect/train_chess/weights/last.pt
```

Export model:

```bash
python stages.py --stage 14 --weights runs/detect/train_chess/weights/best.pt --export-formats onnx,torchscript,tflite
```

## Pipeline Stages (Single File)

Use `python stages.py --list` to see all stages. Core stages include:

1. Frame extraction from video (1 FPS)
2. Board detection (multi-strategy detector)
3. Piece recognition and board annotation
4. FEN encoding demo
5. Move reconstruction demo
6. Stockfish engine analysis demo
7. Feature engineering demo
8. System and dependency check
10. Prepare YOLO dataset
11. Train YOLO11
12. Train YOLOv8
13. Resume training
14. Export model
15. Image inference
16. Video inference

## Current Results (From Latest Run)

From `output/detect/train_chess/results.csv` (30 epochs):

- Early epochs show small non-zero metrics (epoch 1 mAP50 about 0.138).
- Many later epochs report zero precision/recall/mAP and `nan` validation losses.

This suggests the training run is not converging reliably. Likely causes include label/data issues, class imbalance, or incorrect dataset formatting. Re-check stage 10 and verify that label files match images.

## Outputs and Artifacts

- Training outputs: `output/detect/train_chess/`
- Example plots: `results.png`, `BoxPR_curve.png`, `confusion_matrix.png`

## Dataset Layout

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

## Class List (15 Classes)

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

## Model Options

YOLO11:
- yolo11n.pt (nano)
- yolo11s.pt (small, recommended)
- yolo11m.pt (medium)
- yolo11l.pt (large)
- yolo11x.pt (extra large)

YOLOv8:
- yolov8n.pt (nano)
- yolov8s.pt (small)
- yolov8m.pt (medium)
- yolov8l.pt (large)
- yolov8x.pt (extra large)

## Troubleshooting

Out of memory:
- Lower `--batch`
- Use smaller model
- Reduce `--imgsz`

Slow training:
- Use GPU if available
- Increase batch size within VRAM limits
- Enable caching if memory allows

Poor accuracy:
- Verify label integrity
- Increase epochs
- Add more data

## NLP Status (Not Implemented Yet)

The Bangla commentary module is planned but not implemented. Next steps:

- Convert CV outputs (FEN, move list, evaluations) into structured prompts
- Train or fine-tune a Bangla language model
- Integrate NLP into the stage pipeline

## Notes

- Large model/video artifacts are excluded via `.gitignore`.
- Use stage 8 to validate GPU, PyTorch, and Ultralytics before training.

## License

This project uses Ultralytics YOLO which is licensed under AGPL-3.0.
>>>>>>> 19498f8 (read.md)
