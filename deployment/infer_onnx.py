import cv2
import numpy as np
import onnxruntime as ort


def preprocess(image, input_size=640):
    original_h, original_w = image.shape[:2]
    image_resized = cv2.resize(image, (input_size, input_size))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_input = image_rgb.astype(np.float32) / 255.0
    image_input = np.transpose(image_input, (2, 0, 1))
    image_input = np.expand_dims(image_input, axis=0)
    return image_input, original_w, original_h


def main():
    onnx_path = "weights/best_v8s_640.onnx"
    image_path = "assets/images/test.jpeg"

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    image_input, original_w, original_h = preprocess(image)

    outputs = session.run(None, {input_name: image_input})

    print("ONNX inference success.")
    print(f"Input shape: {image_input.shape}")
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")


if __name__ == "__main__":
    main()