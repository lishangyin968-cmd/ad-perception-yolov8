from ultralytics import YOLO
import argparse
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Evaluation Script")
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', help='Model path')
    parser.add_argument('--data', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'datasets', 'intersection_dataset', 'data.yaml'), help='Dataset configuration file')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device to run on (cpu, cuda:0, cuda:1, etc.)')
    parser.add_argument('--name', type=str, default='val', help='Evaluation name')
    return parser.parse_args()


def main():
    """Main function for evaluation."""
    args = parse_args()

    # Load model
    model = YOLO(args.model)

    # Evaluate model
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name
    )

    # Print evaluation metrics
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.precision:.4f}")
    print(f"Recall: {results.box.recall:.4f}")


if __name__ == "__main__":
    main()