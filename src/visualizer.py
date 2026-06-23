from pathlib import Path
from typing import List, Dict
import cv2


def draw_annotations(image_path: Path, annotations_with_boxes: List[Dict], output_path: Path) -> None:
    """
    Draws annotations and detected issues on an image.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        return

    for item in annotations_with_boxes:
        x1, y1, x2, y2 = item["xyxy"]
        issues = item.get("issues", ["valid"])
        class_id = item.get("class_id", "unknown")

        if issues == ["valid"]:
            color = (0, 180, 0)
            label = f"class {class_id}: valid"
        else:
            color = (0, 0, 255)
            label = f"class {class_id}: {','.join(issues)}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text_y = max(y1 - 8, 15)
        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
