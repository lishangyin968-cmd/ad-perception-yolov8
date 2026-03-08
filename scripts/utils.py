import os
import cv2
import numpy as np
from ultralytics import YOLO


def load_model(model_path):
    """Load YOLOv8 model."""
    return YOLO(model_path)


def draw_boxes(image, results):
    """Draw bounding boxes on image."""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            # Get class name
            class_name = result.names[cls]

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def save_results(image, output_path):
    """Save image with detections."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def process_video(input_path, output_path, model, conf=0.25):
    """Process video and save results."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create output video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, conf=conf)

        # Draw boxes
        frame_with_boxes = draw_boxes(frame, results)

        # Write frame
        out.write(frame_with_boxes)

    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")