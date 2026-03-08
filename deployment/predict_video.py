import cv2
from ultralytics import YOLO
from pathlib import Path


def main():
    model_path = "weights/best_v8s_640.pt"
    video_path = "assets/videos/demo_video.mp4"
    output_path = "assets/outputs/demo_results/output_detected.mp4"

    Path("assets/outputs/demo_results").mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, verbose=False)
        annotated_frame = results[0].plot()

        writer.write(annotated_frame)
        cv2.imshow("YOLOv8 Video Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()