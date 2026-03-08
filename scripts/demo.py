from ultralytics import YOLO
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Demo")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model path or name')
    parser.add_argument('--source', type=str, default='assets/sample.mp4',
                        help='Source path (image, video, directory, webcam)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', default=True, help='Save results')
    parser.add_argument('--device', type=str, default='', help='Device to run on (cpu, cuda:0, cuda:1, etc.)')
    return parser.parse_args()


def main():
    """Main function for object detection demo."""
    args = parse_args()

    # Load model
    model = YOLO(args.model)

    # Run inference
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        device=args.device
    )

    print("Inference finished.")


if __name__ == "__main__":
    main()