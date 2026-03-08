from ultralytics import YOLO
import argparse
import os
import sys
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument('--model', type=str, default='yolov8n.yaml', help='Model configuration or pretrained model')
    parser.add_argument('--data', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'datasets', 'intersection_dataset', 'data.yaml'), help='Dataset configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device to run on (cpu, cuda:0, cuda:1, etc.)')
    parser.add_argument('--name', type=str, default='train', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--workers', type=int, default=8,help='number of data loading workers (0 = disable multiprocessing)')
    return parser.parse_args()


def main():
    """Main function for training."""
    args = parse_args()

    # Load model
    model = YOLO(args.model)

    # Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        resume=args.resume,
        workers = args.workers
    )

    print("Training completed.")


if __name__ == "__main__":
    main()