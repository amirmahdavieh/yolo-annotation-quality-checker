from typing import Dict, List, Tuple


def calculate_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """
    Calculates Intersection over Union between two bounding boxes.

    Box format:
    x1, y1, x2, y2
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def validate_annotation(annotation: Dict, image_width: int, image_height: int, xyxy_box: Tuple[int, int, int, int]) -> List[str]:
    """
    Validates one YOLO annotation and returns a list of detected issues.
    """
    issues = []

    if annotation.get("parse_error"):
        return [annotation["parse_error"]]

    x = annotation["x_center"]
    y = annotation["y_center"]
    w = annotation["width"]
    h = annotation["height"]

    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        issues.append("invalid_normalized_values")

    x1, y1, x2, y2 = xyxy_box

    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        issues.append("box_outside_image")

    box_area_ratio = w * h

    if box_area_ratio < 0.0001:
        issues.append("suspicious_small_box")

    if box_area_ratio > 0.80:
        issues.append("suspicious_large_box")

    if not issues:
        issues.append("valid")

    return issues


def find_duplicate_boxes(annotations_with_boxes: List[Dict], iou_threshold: float = 0.90) -> List[Dict]:
    """
    Detects possible duplicate boxes.

    Duplicate condition:
    - same class
    - IoU greater than threshold
    """
    duplicates = []

    for i in range(len(annotations_with_boxes)):
        for j in range(i + 1, len(annotations_with_boxes)):
            ann_a = annotations_with_boxes[i]
            ann_b = annotations_with_boxes[j]

            if ann_a["class_id"] != ann_b["class_id"]:
                continue

            iou = calculate_iou(ann_a["xyxy"], ann_b["xyxy"])

            if iou >= iou_threshold:
                duplicates.append({
                    "box_a_line": ann_a["line_number"],
                    "box_b_line": ann_b["line_number"],
                    "class_id": ann_a["class_id"],
                    "iou": round(iou, 4),
                    "issue": "possible_duplicate_box",
                })

    return duplicates
