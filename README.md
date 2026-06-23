# YOLO Annotation Quality Checker

A Python-based quality assurance tool for YOLO object detection datasets.

This project checks annotation files before model training and detects common labeling problems such as:

- invalid YOLO coordinates
- bounding boxes outside image boundaries
- suspiciously small or large boxes
- duplicate bounding boxes
- missing label files
- empty label files
- class distribution imbalance

The tool also generates:

- `report.csv`
- `summary.json`
- visual images with bounding boxes and warnings

## Why this project matters

Object detection models such as YOLO strongly depend on annotation quality.  
Bad labels can reduce model performance, create false detections, and make training unstable.

This tool helps inspect dataset quality before training.

## Project structure

```text
yolo-annotation-quality-checker/
│
├── data/
│   ├── sample_images/
│   └── sample_labels/
│
├── src/
│   ├── main.py
│   ├── yolo_loader.py
│   ├── validators.py
│   ├── statistics.py
│   └── visualizer.py
│
├── reports/
│   └── visual_examples/
│
├── requirements.txt
└── README.md
```

## YOLO label format

Each `.txt` label file must contain one object per line:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.50 0.50 0.30 0.25
```

All coordinates are normalized between `0` and `1`.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/yolo-annotation-quality-checker.git
cd yolo-annotation-quality-checker
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python src/main.py --images data/sample_images --labels data/sample_labels --output reports
```

## Example output

After running, the project creates:

```text
reports/report.csv
reports/summary.json
reports/visual_examples/
```

## Example issues detected

| Issue | Meaning |
|---|---|
| `missing_label_file` | Image has no matching `.txt` label file |
| `invalid_normalized_values` | YOLO coordinates are outside the valid range |
| `box_outside_image` | Bounding box extends outside the image |
| `suspicious_small_box` | Bounding box is extremely small |
| `suspicious_large_box` | Bounding box covers too much of the image |
| `possible_duplicate_box` | Two boxes probably describe the same object |

## CV bullet point

```latex
\textbf{YOLO Annotation Quality Checker}
\begin{itemize}
    \item Developed a Python-based annotation quality assurance tool for YOLO object detection datasets, validating bounding-box consistency, detecting duplicate and invalid labels, and generating visual reports for dataset review.
\end{itemize}
```

## Technologies

Python, OpenCV, Pandas, NumPy, YOLO annotation format, dataset validation, annotation quality assurance.
