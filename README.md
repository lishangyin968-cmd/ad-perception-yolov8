# Autonomous Driving Perception System Based on YOLOv8

A comprehensive computer vision project for **urban intersection perception** based on YOLOv8.

This project simulates a simplified **autonomous driving perception module**, detecting common traffic participants such as:

- Car
- Bus
- Motorcycle
- Bicycle

The project covers the **complete pipeline from model training to deployment**, including:

- Model training and evaluation
- Model comparison experiments
- Visualization and analysis
- Real-time video inference
- Performance benchmarking (FPS)
- ONNX deployment

---

# 1. Project Overview

In autonomous driving systems, the perception module is responsible for detecting and understanding the surrounding environment.

This project implements a **vision-based object detection system** using YOLOv8 for urban traffic scenes.

The system can detect multiple types of traffic participants and supports real-time inference and deployment.

The project evolves through three stages:

| Stage | Description |
|-----|-----|
| Baseline Version | Basic YOLOv8 training and inference |
| Experiment Version | Model comparison and analysis |
| Engineering Version | Real-time detection and deployment |

---

# 2. Project Structure

```

ad-perception-yolov8/
├── assets/
│   ├── videos/               # Demo videos
│   ├── images/               # Test images
│   └── outputs/              # Detection results
│
├── configs/
│   └── data.yaml             # Dataset configuration
│
├── data/
│   └── datasets/
│       └── intersection_dataset/
│           ├── train/
│           │   ├── images/
│           │   └── labels/
│           ├── valid/
│           ├── test/
│           └── README.roboflow.txt
│
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── visualize.py          # Visualization tools
│   └── utils.py
│
├── deployment/
│   ├── predict_video.py      # Real-time video detection
│   ├── benchmark_fps.py      # FPS benchmark
│   ├── export_onnx.py        # Export model to ONNX
│   └── infer_onnx.py         # ONNX inference validation
│
├── experiments/
│   └── results_table.md      # Experiment comparison
│
├── weights/
│   ├── best_v8s_640.pt
│   └── best_v8s_640.onnx
│
├── runs/                     # YOLO training logs
├── requirements.txt
└── README.md

````

---

# 3. Installation

Clone the repository:

```bash
git clone https://github.com/yourname/ad-perception-yolov8.git
cd ad-perception-yolov8
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

* Python 3.9+
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* ONNX Runtime

---

# 4. Dataset

The project uses a **traffic intersection dataset** annotated in YOLO format.

Dataset structure:

```
intersection_dataset/
├── train/
│   ├── images
│   └── labels
├── valid/
├── test/
```

Object classes:

| Class ID | Category   |
| -------- | ---------- |
| 0        | bicycle    |
| 1        | bus        |
| 2        | car        |
| 3        | motorcycle |

---

# 5. Model Training

Train YOLOv8 model:

```
yolo train model=yolov8n.pt data=configs/data.yaml imgsz=640 epochs=30 batch=8
```

Training results will be saved in:

```
runs/detect/
```

Metrics include:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95

---

# 6. Model Evaluation

Evaluate trained model:

```
python scripts/evaluate.py
```

The evaluation includes:

* Precision / Recall
* Confusion Matrix
* PR Curve
* F1 Curve

---

# 7. Experiment Comparison

This project conducts several experiments to analyze model performance.

### Model Comparison

| Model   | Input Size | mAP@0.5   | Notes                                         |
| ------- | ---------- | --------- | --------------------------------------------- |
| YOLOv8n | 640        | 0.551     | baseline                                      |
| YOLOv8n | 960        | 0.474     | higher resolution did not improve performance |
| YOLOv8s | 640        | **0.801** | best result                                   |

Key observations:

* Increasing resolution does not always improve performance.
* YOLOv8s significantly improves detection accuracy compared with YOLOv8n.
* Major gains appear in **car** and **motorcycle** detection.

---

# 8. Visualization and Analysis

The training process generates several useful diagnostic plots:

* Training loss curves
* Precision / Recall curves
* F1 confidence curve
* Confusion matrix
* Dataset distribution

Example outputs include:

```
runs/detect/train/
 ├── results.png
 ├── PR_curve.png
 ├── confusion_matrix.png
 └── labels.jpg
```

These visualizations help analyze model performance and dataset balance.

---

# 9. Real-Time Video Detection

The project supports real-time video inference.

Run:

```
python deployment/predict_video.py
```

This script:

* Reads a video file
* Performs frame-by-frame detection
* Displays results in real-time
* Saves annotated output video

Example output:

```
assets/outputs/demo_results/output_detected.mp4
```

---

# 10. FPS Benchmark

To evaluate inference speed:

```
python deployment/benchmark_fps.py
```

Benchmark result on a test image:

| Metric          | Value       |
| --------------- | ----------- |
| Average Latency | **7.10 ms** |
| FPS             | **140.78**  |

This indicates the model can achieve **real-time performance** under 640×640 input resolution.

---

# 11. ONNX Deployment

The trained model can be exported to ONNX for cross-platform deployment.

Export ONNX model:

```
python deployment/export_onnx.py
```

Validate ONNX inference:

```
python deployment/infer_onnx.py
```

ONNX output example:

| Item         | Result           |
| ------------ | ---------------- |
| Input Shape  | (1, 3, 640, 640) |
| Output Shape | (1, 8, 8400)     |
| Status       | Success          |

---

# 12. Deployment Pipeline

The deployment architecture follows:

```
PyTorch (.pt)
   ↓
ONNX (.onnx)
   ↓
TensorRT (.engine)   [future work]
```

Completed steps:

* PyTorch inference
* FPS benchmark
* ONNX export
* ONNX runtime inference

Planned improvements:

* TensorRT acceleration
* Multi-thread inference
* Edge deployment

---

# 13. Key Features

This project demonstrates:

* End-to-end object detection pipeline
* Model comparison experiments
* Performance benchmarking
* Real-time video inference
* Deployment-friendly model export

---

# 14. Future Work

Potential improvements:

* TensorRT acceleration
* Multi-camera perception
* Object tracking
* Integration with autonomous driving frameworks
