from collections import Counter
from typing import List, Dict


def generate_summary(report_rows: List[Dict]) -> Dict:
    """
    Generates summary statistics from all validation report rows.
    """
    issue_counter = Counter(row["issue"] for row in report_rows)
    class_counter = Counter()

    images = set()
    valid_annotations = 0
    total_annotations = 0

    for row in report_rows:
        images.add(row["image"])

        if row.get("class_id") not in [None, ""]:
            class_counter[str(row["class_id"])] += 1
            total_annotations += 1

        if row["issue"] == "valid":
            valid_annotations += 1

    return {
        "total_images_checked": len(images),
        "total_annotation_rows": total_annotations,
        "valid_annotations": valid_annotations,
        "issues": dict(issue_counter),
        "class_distribution": dict(class_counter),
    }
