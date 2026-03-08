from ultralytics import YOLO


def main():
    model = YOLO("weights/best_v8s_640.pt")
    model.export(format="onnx", imgsz=640)

    print("ONNX export finished.")


if __name__ == "__main__":
    main()