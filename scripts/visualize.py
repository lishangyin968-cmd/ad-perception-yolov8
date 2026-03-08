import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import load_model, draw_boxes


def visualize_dataset(dataset_path, num_samples=5):
    """Visualize dataset samples with annotations."""
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    # Get image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Select samples
    samples = image_files[:num_samples]

    plt.figure(figsize=(15, 10))
    for i, img_file in enumerate(samples):
        # Load image
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()

            # Draw annotations
            for label in labels:
                parts = label.strip().split()
                cls = int(parts[0])
                x_center = float(parts[1]) * img.shape[1]
                y_center = float(parts[2]) * img.shape[0]
                width = float(parts[3]) * img.shape[1]
                height = float(parts[4]) * img.shape[0]

                # Calculate coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(img_file)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('dataset_visualization.png')
    plt.show()


def visualize_predictions(model_path, image_path):
    """Visualize model predictions."""
    # Load model
    model = load_model(model_path)

    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(img)

    # Draw boxes
    img_with_boxes = draw_boxes(img.copy(), results)

    # Plot results
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_with_boxes)
    plt.title('Predictions')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    plt.show()


if __name__ == "__main__":
    # Visualize dataset
    visualize_dataset('data/datasets/intersection_dataset/train')

    # Visualize predictions
    # visualize_predictions('runs/detect/train/weights/best.pt', 'bus.jpg')