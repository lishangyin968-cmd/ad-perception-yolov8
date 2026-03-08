# Experiment Results

This file summarizes the model comparison experiments conducted in this project.

The experiments evaluate the impact of:

- Model size
- Input resolution
- Detection performance

All experiments are conducted on:

GPU: RTX 5060 Laptop GPU  
Framework: YOLOv8 (Ultralytics)

---

# Model Comparison

| Model | Input Size | mAP@0.5 | Notes |
|------|------|------|------|
| YOLOv8n | 640 | 0.551 | Baseline experiment |
| YOLOv8n | 960 | 0.474 | Higher resolution did not improve overall performance |
| YOLOv8s | 640 | **0.801** | Best overall performance |

---

# Observations

### 1 Model Capacity

YOLOv8s significantly improves detection accuracy compared with YOLOv8n.

The improvement is especially visible in:

- car detection
- motorcycle detection

---

### 2 Input Resolution

Increasing input resolution from **640 → 960** did not improve overall performance.

Possible reasons include:

- dataset size limitation
- model capacity mismatch
- class imbalance

---

### 3 Dataset Imbalance

The dataset shows a long-tail distribution:

| Class | Instances |
|------|------|
| bicycle | 16 |
| bus | 47 |
| car | 613 |
| motorcycle | 1770 |

This imbalance affects the detection performance of small classes such as bicycle.

---

# Conclusion

YOLOv8s with **640 input resolution** provides the best trade-off between:

- accuracy
- inference speed
- deployment feasibility.