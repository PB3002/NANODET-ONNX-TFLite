import cv2
import os
import time
import onnxruntime
import argparse
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NanoDet:
    def __init__(self, model_path, score_thr, input_shape):
        self.score_thr = score_thr
        self.input_shape = input_shape  # Should be a tuple, e.g., (320, 320)
        self.session = onnxruntime.InferenceSession(model_path)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        
        self.coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.palette = np.array([[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.coco_names))], dtype=np.uint8)

    def preprocess(self, img, input_shape):
        img_h, img_w = img.shape[:2]
        net_h, net_w = input_shape
        scale = min(net_h / img_h, net_w / img_w)
        
        resize_w = int(round(img_w * scale))
        resize_h = int(round(img_h * scale))
        
        resized_img = cv2.resize(img, (resize_w, resize_h))
        
        padded_img = np.full((net_h, net_w, 3), 114, dtype=np.uint8)
        padded_img[:resize_h, :resize_w] = resized_img
        
        # Normalize and transpose
        padded_img = padded_img.astype(np.float32) / 255.0
        # HWC to CHW
        padded_img = padded_img.transpose((2, 0, 1))
        # Add batch dimension
        input_tensor = np.expand_dims(padded_img, axis=0)
        return input_tensor, scale

    def postprocess(self, preds, ratio, score_thr):
        # Assuming preds is a list of arrays or a single array from the model output
        # This part is highly dependent on the specific output format of NanoDet ONNX model
        # The original script seems to expect preds[0] to contain detections:
        # [batch_id, x_min, y_min, x_max, y_max, score, class_id]
        # This might need adjustment based on your exact NanoDet model variant.
        
        # If preds is a list, take the first element which usually contains the detections
        if isinstance(preds, list) and len(preds) > 0:
            detections = preds[0] 
        elif isinstance(preds, np.ndarray):
            detections = preds
        else:
            logging.warning("Unexpected prediction output format.")
            return []

        if detections.ndim == 3 and detections.shape[0] == 1: # (1, num_dets, 7)
            detections = detections[0]
        elif detections.ndim != 2 or detections.shape[1] != 7: # (num_dets, 7)
             logging.warning(f"Detections array shape {detections.shape} is not as expected (N, 7). Skipping postprocessing for this image.")
             return []


        results = []
        for det in detections:
            # Assuming format: [x_min, y_min, x_max, y_max, class_id, score] (common for some nanodet versions)
            # or [batch_idx, x_min, y_min, x_max, y_max, score, class_id] from the original example.
            # Adjust indices if your model output is different.
            # Based on typical ONNX outputs for object detection:
            # If shape is (1, N, 7) for batch_idx, x1,y1,x2,y2,score,class
            # If shape is (1, N, 6) for x1,y1,x2,y2,score,class

            # Let's assume the original script's structure where `preds` from session.run might be a list,
            # and `preds[0]` contains the actual detection data with shape (num_detections, 7)
            # where columns are [batch_idx, xmin, ymin, xmax, ymax, score, class_id]
            # Or, if it's just (num_detections, 6) for [xmin, ymin, xmax, ymax, score, class_id]

            # Let's try to be robust for common variations:
            if len(det) == 7: # batch_id, x1, y1, x2, y2, score, class_id
                score = det[5]
                class_id = int(det[6])
                x1, y1, x2, y2 = det[1:5]
            elif len(det) == 6: # x1, y1, x2, y2, score, class_id
                score = det[4]
                class_id = int(det[5])
                x1, y1, x2, y2 = det[0:4]
            else:
                logging.warning(f"Unexpected detection length: {len(det)}. Skipping.")
                continue

            if score >= score_thr:
                # Adjust box coordinates by the ratio
                x1 /= ratio
                y1 /= ratio
                x2 /= ratio
                y2 /= ratio
                results.append([int(x1), int(y1), int(x2), int(y2), score, class_id])
        return results
        
    def draw_bboxes(self, img, bboxes, palette, names, score_thr):
        for bbox in bboxes:
            x1, y1, x2, y2, score, class_id = bbox
            if score < score_thr:
                continue
            
            label = f'{names[class_id]}: {score:.2f}'
            color = palette[class_id]
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color.tolist(), 2)
            
            # Put label text above the rectangle
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - text_size[1] - baseline), (x1 + text_size[0], y1), color.tolist(), -1)
            cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return img

def get_args():
    parser = argparse.ArgumentParser(description="Benchmark NanoDet ONNX model FPS and Latency.")
    parser.add_argument('--model', required=True, type=str, help="Path to the ONNX model file.")
    parser.add_argument('--dataset_path', required=True, type=str, help="Path to the directory containing images.")
    parser.add_argument('--output_path', type=str, default=None, help="Optional: Path to save processed images. If not provided, images are not saved.")
    parser.add_argument('--score_thr', type=float, default=0.5, help="Score threshold for detections.")
    parser.add_argument('--input_shape', type=str, default="320,320", help="Model input shape as 'height,width' (e.g., '320,320').")
    return parser.parse_args()

def main():
    args = get_args()

    try:
        input_shape_tuple = tuple(map(int, args.input_shape.split(',')))
        if len(input_shape_tuple) != 2:
            raise ValueError("Input shape must be two integers (height,width).")
    except ValueError as e:
        logging.error(f"Invalid input_shape format: {args.input_shape}. Error: {e}")
        return

    try:
        predictor = NanoDet(args.model, args.score_thr, input_shape_tuple)
    except Exception as e:
        logging.error(f"Failed to initialize NanoDet predictor: {e}")
        return

    if not os.path.isdir(args.dataset_path):
        logging.error(f"Dataset path '{args.dataset_path}' not found or is not a directory.")
        return

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        logging.info(f"Output images will be saved to: {args.output_path}")

    image_files = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    for item in os.listdir(args.dataset_path):
        if os.path.splitext(item.lower())[1] in supported_extensions:
            image_files.append(os.path.join(args.dataset_path, item))

    if not image_files:
        logging.info("No image files found in the dataset directory.")
        return

    total_inference_time_ms = 0
    total_core_processing_time_ms = 0 # Preprocess + Inference + Postprocess
    images_processed_count = 0

    logging.info(f"Starting benchmark on {len(image_files)} images...")

    for i, img_path in enumerate(image_files):
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                logging.warning(f"Could not read image: {img_path}. Skipping.")
                continue
            
            # --- Core Processing Timing Start ---
            core_processing_start_time = time.perf_counter()

            input_tensor, ratio = predictor.preprocess(frame, predictor.input_shape)
            
            inference_start_time = time.perf_counter()
            ort_inputs = {predictor.input_name: input_tensor}
            ort_outs = predictor.session.run(None, ort_inputs)
            inference_end_time = time.perf_counter()
            
            results = predictor.postprocess(ort_outs, ratio, args.score_thr)
            
            core_processing_end_time = time.perf_counter()
            # --- Core Processing Timing End ---

            current_inference_latency_ms = (inference_end_time - inference_start_time) * 1000
            current_core_processing_time_ms = (core_processing_end_time - core_processing_start_time) * 1000

            total_inference_time_ms += current_inference_latency_ms
            total_core_processing_time_ms += current_core_processing_time_ms
            images_processed_count += 1
            
            logging.info(f"Processed {os.path.basename(img_path)} ({i+1}/{len(image_files)}): "
                         f"Infer Latency: {current_inference_latency_ms:.2f} ms, "
                         f"Core Proc Time: {current_core_processing_time_ms:.2f} ms")

            if args.output_path:
                # Draw bboxes on a copy if you need 'frame' for something else,
                # or directly on 'frame' if it's the final step for this image.
                output_frame = frame.copy() 
                predictor.draw_bboxes(output_frame, results, predictor.palette, predictor.coco_names, args.score_thr)
                output_file_path = os.path.join(args.output_path, os.path.basename(img_path))
                cv2.imwrite(output_file_path, output_frame)

        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}", exc_info=True) # Log stack trace

    if images_processed_count == 0:
        logging.info("No images were successfully processed.")
        return

    avg_inference_latency_ms = total_inference_time_ms / images_processed_count
    avg_core_processing_time_ms = total_core_processing_time_ms / images_processed_count
    avg_fps = 1000.0 / avg_core_processing_time_ms if avg_core_processing_time_ms > 0 else 0 # FPS based on core processing time

    print("\n--- Benchmark Results ---")
    print(f"Total images processed: {images_processed_count}")
    print(f"Average inference latency: {avg_inference_latency_ms:.2f} ms")
    print(f"Average core processing time per image (preprocess + inference + postprocess): {avg_core_processing_time_ms:.2f} ms")
    print(f"Average FPS (based on core processing time): {avg_fps:.2f}")
    print("-------------------------\n")

if __name__ == '__main__':
    main()