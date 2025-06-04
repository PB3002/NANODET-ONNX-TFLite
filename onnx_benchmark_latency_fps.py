import argparse
import os
import time
import glob
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='ONNX inference benchmark for NanoDet')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input image directory')
    parser.add_argument('--output_path', type=str, default='outputs', help='Path to save output images')
    parser.add_argument('--score_thr', type=float, default=0.5, help='Score threshold for detection')
    parser.add_argument('--input_shape', type=str, default='320,320', help='Model input shape as height,width')
    return parser.parse_args()

def preprocess_image(image_path, input_shape):
    img = cv2.imread(image_path)
    h, w = input_shape
    img_resized = cv2.resize(img, (w, h))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)  # HWC to CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
    return img, img_input

def postprocess(outputs, score_thr, input_shape, img_shape):
    boxes, scores, classes = outputs  # Adjust based on actual model output
    mask = scores > score_thr
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    # Scale boxes back to original image size
    h, w = img_shape[:2]
    input_h, input_w = input_shape
    boxes[:, [0, 2]] *= w / input_w
    boxes[:, [1, 3]] *= h / input_h
    return boxes, scores, classes

def draw_boxes(img, boxes, scores, classes):
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'Class {int(cls)}: {score:.2f}'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def main():
    args = parse_args()
    input_shape = tuple(map(int, args.input_shape.split(',')))

    # Initialize ONNX session
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Get list of images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_path, ext)))

    if not image_paths:
        print(f"No images found in {args.input_path}")
        return

    # Benchmarking variables
    latencies = []
    total_images = len(image_paths)

    print(f"Processing {total_images} images...")

    for idx, image_path in enumerate(image_paths):
        start_time = time.time()

        # Preprocess
        img, img_input = preprocess_image(image_path, input_shape)

        # Inference
        outputs = session.run(None, {input_name: img_input})[0]
        # Note: Adjust output parsing based on NanoDet's actual output format
        boxes, scores, classes = postprocess(outputs, args.score_thr, input_shape, img.shape)

        # Draw boxes
        img = draw_boxes(img, boxes, scores, classes)

        # Save output
        output_filename = os.path.join(args.output_path, f"output_{idx}_{os.path.basename(image_path)}")
        cv2.imwrite(output_filename, img)

        # Calculate latency
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)

        print(f"Processed {image_path}: Latency = {latency:.2f} ms")

    # Calculate metrics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    # Print benchmark results
    print("\nBenchmark Results:")
    print(f"Total Images: {total_images}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Latency Std Dev: {std_latency:.2f} ms")
    print(f"Average FPS: {fps:.2f}")
    print(f"Output images saved to: {args.output_path}")

if __name__ == '__main__':
    main()