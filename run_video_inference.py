import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO on a video and visualize results.")
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--source", default="videos/chess.mp4", help="Path to input video")
    parser.add_argument("--out", default="runs/detect/predict_custom", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=3000, help="Stop after this many frames")
    parser.add_argument("--no-show", action="store_true", help="Disable live window display")
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.55,
        help="Scale factor for the preview window (e.g., 0.75 for 75%)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    source_path = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    results_iter = model.predict(
        source=str(source_path),
        stream=True,
        conf=args.conf,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        save=False,
    )

    writer = None
    video_out_path = out_dir / f"{source_path.stem}_pred.mp4"

    try:
        for idx, result in enumerate(results_iter, start=1):
            if args.max_frames > 0 and idx > args.max_frames:
                break
            annotated = result.plot()
            if writer is None:
                height, width = annotated.shape[:2]
                fps = result.speed.get("fps", 25) or 25
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
            writer.write(annotated)
            if not args.no_show:
                if args.display_scale != 1.0:
                    disp_w = max(1, int(annotated.shape[1] * args.display_scale))
                    disp_h = max(1, int(annotated.shape[0] * args.display_scale))
                    preview = cv2.resize(annotated, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                else:
                    preview = annotated
                cv2.imshow("YOLO Prediction", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print(f"Saved video: {video_out_path}")


if __name__ == "__main__":
    main()
