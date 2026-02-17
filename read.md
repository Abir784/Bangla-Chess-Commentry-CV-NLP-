# Bangla Chess Commentary CV/NLP Pipeline

This project builds a computer-vision pipeline to analyze chess videos, detect the board and pieces, and (eventually) generate Bangla commentary. All runnable steps are unified in a single entrypoint: `stages.py`.

## Project Overview

Goal: detect chessboards and pieces from video frames, reconstruct board state, and prepare for commentary generation. The CV pipeline is implemented; the NLP commentary module is **not implemented yet**.

## Pipeline Stages (Single File)

Use `python stages.py --list` to view all stages. Core stages include:

1. Frame extraction from video (1 FPS)
2. Board detection using a multi-strategy detector
3. Piece recognition and board annotation
4. FEN encoding demo
5. Move reconstruction demo
6. Stockfish engine analysis demo
7. Feature engineering demo
8. System and dependency check
10. Prepare YOLO dataset (train/val split)
11. Train YOLO11
12. Train YOLOv8
13. Resume training
14. Export model
15. Image inference
16. Video inference

## Key Data Locations

- Frames: `preprocessed_frames/`
- Annotations: `Annotations/`
- Dataset: `datasets/chess_yolo/`
- Training outputs: `output/detect/train_chess/`

## Results Found (Current)

From `output/detect/train_chess/results.csv` (30 epochs):

- Early epoch metrics show small non-zero values (e.g., epoch 1: mAP50 about 0.138).
- Most later epochs report zero precision/recall/mAP, and validation losses become `nan`.

This indicates the current training run is not converging reliably. Likely causes include label/data issues, class imbalance, or incorrect dataset formatting. Re-check the dataset creation step (stage 10) and verify that label files match images.

## Example Outputs

The training run produced standard YOLO artifacts:

- `output/detect/train_chess/results.png`
- `output/detect/train_chess/BoxPR_curve.png`
- `output/detect/train_chess/confusion_matrix.png`

## NLP Status

The NLP commentary generation is **not implemented yet**. Planned work includes:

- Converting CV outputs (FEN, move list, engine evals) into structured textual prompts
- Training or fine-tuning a Bangla language model for commentary
- Integrating the NLP module into the stage pipeline

## Notes

- Large model/video artifacts are excluded from git via `.gitignore`.
- Use stage 8 to validate GPU, PyTorch, and Ultralytics before training.
