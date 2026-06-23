from pathlib import Path
from typing import List, Dict, Tuple


def load_yolo_labels(label_path: Path) -> List[Dict]:
    """
    Loads a YOLO label file.

    Expected format per line:
    class_id x_center y_center width height

    Returns a list of dictionaries.
    """
    annotations = []

    if not label_path.exists():
        return annotations

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    for line_number, line in enumerate(lines, start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            annotations.append({
                "line_number": line_number,
                "raw_line": line,
                "parse_error": "expected_5_values",
            })
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            annotations.append({
                "line_number": line_number,
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "raw_line": line,
                "parse_error": None,
            })

        except ValueError:
            annotations.append({
                "line_number": line_number,
                "raw_line": line,
                "parse_error": "invalid_number_format",
            })

    return annotations


def yolo_to_xyxy(annotation: Dict, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    """
    Converts YOLO normalized coordinates to pixel coordinates.

    YOLO format:
    x_center, y_center, width, height

    Output:
    x1, y1, x2, y2
    """
    x_center = annotation["x_center"] * image_width
    y_center = annotation["y_center"] * image_height
    box_width = annotation["width"] * image_width
    box_height = annotation["height"] * image_height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    return x1, y1, x2, y2
