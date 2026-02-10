from pathlib import Path
import random
import shutil

workspace = Path(r"F:\BanglaNLP")
images_dir = workspace / "preprocessed_frames"
labels_dir = workspace / "Annotations" / "labels_nlp_2026-02-10-11-44-39"
out_dir = workspace / "datasets" / "chess_yolo"

images_train = out_dir / "images" / "train"
images_val = out_dir / "images" / "val"
labels_train = out_dir / "labels" / "train"
labels_val = out_dir / "labels" / "val"

for p in [images_train, images_val, labels_train, labels_val]:
    p.mkdir(parents=True, exist_ok=True)

labels_path = workspace / "labels.txt"
labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]

pairs = []
for label_file in labels_dir.glob("*.txt"):
    image_file = images_dir / (label_file.stem + ".jpg")
    if image_file.exists():
        pairs.append((image_file, label_file))

pairs.sort(key=lambda x: x[0].name)
random.seed(42)
random.shuffle(pairs)

split_idx = int(len(pairs) * 0.8)
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

for image_file, label_file in train_pairs:
    shutil.copy2(image_file, images_train / image_file.name)
    shutil.copy2(label_file, labels_train / label_file.name)

for image_file, label_file in val_pairs:
    shutil.copy2(image_file, images_val / image_file.name)
    shutil.copy2(label_file, labels_val / label_file.name)

yaml_lines = [
    f"path: {out_dir.as_posix()}",
    "train: images/train",
    "val: images/val",
    "names:",
]
for name in labels:
    yaml_lines.append(f"  - {name}")

(out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

print(f"images: {len(pairs)}")
print(f"train: {len(train_pairs)}")
print(f"val: {len(val_pairs)}")
print(f"data.yaml: {out_dir / 'data.yaml'}")
