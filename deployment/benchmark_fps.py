import time
import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "weights" / "best_v8s_640.pt"
    image_dir = project_root / "assets" / "images"

    warmup_runs = 20
    benchmark_runs = 100
    # 自动查找图片（jpg / jpeg / png）
    images = list(image_dir.glob("*.jpg")) + \
             list(image_dir.glob("*.jpeg")) + \
             list(image_dir.glob("*.png"))
    if len(images) == 0:
        raise RuntimeError(f"No images found in {image_dir}")

    image_path = str(images[0])
    print(f"Using benchmark image: {image_path}")
    #加载模型
    model = YOLO(model_path)
    #读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    # -------------------------
    # 预热（避免第一次推理慢）
    # -------------------------
    for _ in range(warmup_runs):
        _ = model(image, conf=0.3, verbose=False)

    # -------------------------
    # 开始测试
    # -------------------------
    start_time = time.time()
    for _ in range(benchmark_runs):
        _ = model(image, conf=0.3, verbose=False)
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = total_time / benchmark_runs
    fps = 1.0 / avg_latency

    print("\n===== FPS Benchmark Result =====")
    print(f"Benchmark image: {image_path}")
    print(f"Runs: {benchmark_runs}")
    print(f"Average latency: {avg_latency * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()