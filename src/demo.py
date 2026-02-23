from ultralytics import YOLO

def main():
    # 使用官方预训练模型
    model = YOLO("yolov8n.pt")

    results = model.predict(
        source="assets/sample.mp4",  # 放一个小视频
        conf=0.25,
        save=True
    )

    print("Inference finished.")

if __name__ == "__main__":
    main()