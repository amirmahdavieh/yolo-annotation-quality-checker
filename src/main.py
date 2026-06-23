import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from yolo_loader import load_yolo_labels, yolo_to_xyxy
from validators import validate_annotation, find_duplicate_boxes
from statistics import generate_summary
from visualizer import draw_annotations


SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def find_images(images_dir: Path):
    images = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        images.extend(images_dir.glob(f"*{ext}"))
    return sorted(images)


def check_dataset(images_dir: Path, labels_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual_examples"
    visual_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []

    image_paths = find_images(images_dir)

    for image_path in tqdm(image_paths, desc="Checking images"):
        image = cv2.imread(str(image_path))

        if image is None:
            report_rows.append({
                "image": image_path.name,
                "label_file": "",
                "line_number": "",
                "class_id": "",
                "issue": "image_read_error",
                "details": "OpenCV could not read the image",
            })
            continue

        image_height, image_width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            report_rows.append({
                "image": image_path.name,
                "label_file": label_path.name,
                "line_number": "",
                "class_id": "",
                "issue": "missing_label_file",
                "details": "No matching YOLO label file found",
            })
            continue

        if label_path.read_text(encoding="utf-8").strip() == "":
            report_rows.append({
                "image": image_path.name,
                "label_file": label_path.name,
                "line_number": "",
                "class_id": "",
                "issue": "empty_label_file",
                "details": "Label file exists but contains no annotations",
            })
            continue

        annotations = load_yolo_labels(label_path)
        annotations_with_boxes = []

        for ann in annotations:
            if ann.get("parse_error"):
                report_rows.append({
                    "image": image_path.name,
                    "label_file": label_path.name,
                    "line_number": ann.get("line_number", ""),
                    "class_id": "",
                    "issue": ann["parse_error"],
                    "details": ann.get("raw_line", ""),
                })
                continue

            xyxy = yolo_to_xyxy(ann, image_width, image_height)
            issues = validate_annotation(ann, image_width, image_height, xyxy)

            annotations_with_boxes.append({
                **ann,
                "xyxy": xyxy,
                "issues": issues,
            })

            for issue in issues:
                report_rows.append({
                    "image": image_path.name,
                    "label_file": label_path.name,
                    "line_number": ann["line_number"],
                    "class_id": ann["class_id"],
                    "issue": issue,
                    "details": f"xyxy={xyxy}",
                })

        duplicates = find_duplicate_boxes(annotations_with_boxes)

        for duplicate in duplicates:
            report_rows.append({
                "image": image_path.name,
                "label_file": label_path.name,
                "line_number": f"{duplicate['box_a_line']} and {duplicate['box_b_line']}",
                "class_id": duplicate["class_id"],
                "issue": duplicate["issue"],
                "details": f"IoU={duplicate['iou']}",
            })

        draw_annotations(
            image_path=image_path,
            annotations_with_boxes=annotations_with_boxes,
            output_path=visual_dir / f"{image_path.stem}_checked.jpg",
        )

    report_df = pd.DataFrame(report_rows)
    report_path = output_dir / "report.csv"
    report_df.to_csv(report_path, index=False)

    summary = generate_summary(report_rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print(f"\nReport saved to: {report_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Visual examples saved to: {visual_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Annotation Quality Checker")
    parser.add_argument("--images", type=str, default="data/sample_images", help="Path to image directory")
    parser.add_argument("--labels", type=str, default="data/sample_labels", help="Path to YOLO label directory")
    parser.add_argument("--output", type=str, default="reports", help="Path to output report directory")

    args = parser.parse_args()

    check_dataset(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
