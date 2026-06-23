import random
import shutil
from pathlib import Path

from PIL import Image
from ultralytics.utils.downloads import download
from ultralytics.utils import ASSETS_URL, TQDM


def visdrone2yolo(dataset_dir: Path, split: str, source_name: str):
    """Convert VisDrone annotations to YOLO format."""
    source_dir = dataset_dir / source_name
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move images into YOLO-style structure
    source_images_dir = source_dir / "images"
    if source_images_dir.exists():
        for img in source_images_dir.glob("*.jpg"):
            target_img = images_dir / img.name
            if not target_img.exists():
                shutil.move(str(img), str(target_img))

    # Convert annotations
    annotation_dir = source_dir / "annotations"

    for annotation_file in TQDM(annotation_dir.glob("*.txt"), desc=f"Converting {split}"):
        image_path = images_dir / annotation_file.with_suffix(".jpg").name

        if not image_path.exists():
            continue

        img_width, img_height = Image.open(image_path).size
        dw, dh = 1.0 / img_width, 1.0 / img_height

        yolo_lines = []

        with open(annotation_file, encoding="utf-8") as file:
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if len(row) < 6:
                    continue

                # row format:
                # bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
                if row[4] == "0":
                    continue

                x, y, w, h = map(int, row[:4])
                cls = int(row[5]) - 1

                x_center = (x + w / 2) * dw
                y_center = (y + h / 2) * dh
                w_norm = w * dw
                h_norm = h * dh

                yolo_lines.append(
                    f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                )

        output_label_path = labels_dir / annotation_file.name
        output_label_path.write_text("".join(yolo_lines), encoding="utf-8")


def copy_random_sample(
    dataset_dir: Path,
    split: str = "train",
    sample_size: int = 10,
    output_images_dir: Path = Path("data/sample_images"),
    output_labels_dir: Path = Path("data/sample_labels"),
):
    """Copy random image-label pairs into project sample folders."""
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg"))

    valid_pairs = []

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists() and label_path.stat().st_size > 0:
            valid_pairs.append((image_path, label_path))

    if len(valid_pairs) < sample_size:
        raise ValueError(f"Only {len(valid_pairs)} valid image-label pairs found.")

    selected_pairs = random.sample(valid_pairs, sample_size)

    for image_path, label_path in selected_pairs:
        shutil.copy2(image_path, output_images_dir / image_path.name)
        shutil.copy2(label_path, output_labels_dir / label_path.name)

    print(f"Copied {sample_size} random image-label pairs.")
    print(f"Images saved to: {output_images_dir}")
    print(f"Labels saved to: {output_labels_dir}")


def main():
    dataset_dir = Path("datasets/VisDrone")

    urls = [
        f"{ASSETS_URL}/VisDrone2019-DET-train.zip",
        f"{ASSETS_URL}/VisDrone2019-DET-val.zip",
        f"{ASSETS_URL}/VisDrone2019-DET-test-dev.zip",
    ]

    print("Downloading VisDrone...")
    download(urls, dir=dataset_dir, threads=4)

    splits = {
        "VisDrone2019-DET-train": "train",
        "VisDrone2019-DET-val": "val",
        "VisDrone2019-DET-test-dev": "test",
    }

    print("Converting VisDrone annotations to YOLO format...")
    for source_folder, split in splits.items():
        source_path = dataset_dir / source_folder

        if source_path.exists():
            visdrone2yolo(dataset_dir, split, source_folder)
            shutil.rmtree(source_path)

    print("Copying random sample to project folders...")
    copy_random_sample(
        dataset_dir=dataset_dir,
        split="train",
        sample_size=10,
        output_images_dir=Path("data/sample_images"),
        output_labels_dir=Path("data/sample_labels"),
    )


if __name__ == "__main__":
    main()