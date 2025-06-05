
import argparse
import os
import glob
import time

import cv2
import numpy as np
import onnxruntime
from loguru import logger

# Assuming utils.py is in the same directory or accessible in the Python path
# Using the ORIGINAL utils.py from the repository
from utils.utils import image_preprocess, post_process, visualize

# Default COCO class names (80 classes) - Used if model detection fails or for reference
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Assumed reg_max for NanoDet-Plus models
# This might need adjustment for different model types
DEFAULT_REG_MAX = 7

def make_parser():
    """Creates an argument parser for the benchmarking script."""
    # Changed argument name and default behavior
    parser = argparse.ArgumentParser("ONNX Model Benchmark with Class Filtering (Default: Person)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the directory containing input images.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='outputs_benchmark',
        help="Path to the directory to save optional output images.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshold to filter detection results for visualization.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="320,320",
        help="Specify the input shape for inference (height,width).",
    )
    parser.add_argument(
        "--classes", # Renamed from --filter_classes
        nargs='+', # Allows one or more class names
        type=str,
        default=None, # Default handled in main block now
        help="Optional list of class names to filter results for (e.g., person car bus). If not provided, defaults to 'person'.",
    )
    return parser

def detect_model_parameters(interpreter, input_shape, reg_max):
    """Runs a dummy inference to detect output shape and infer num_classes."""
    logger.info("Attempting to detect model parameters (num_classes)...")
    try:
        # Create dummy input
        dummy_input = np.zeros((1, 3, input_shape[0], input_shape[1]), dtype=np.float32)
        ort_inputs = {interpreter.get_inputs()[0].name: dummy_input}
        
        # Run inference
        output = interpreter.run(None, ort_inputs)
        output_shape = output[0].shape
        logger.info(f"Detected model output shape: {output_shape}")

        # Infer num_classes: Assumes output format [batch, num_preds, num_classes + (reg_max + 1) * 4]
        # This calculation is specific to NanoDet-Plus output structure
        last_dim = output_shape[-1]
        num_classes = last_dim - (reg_max + 1) * 4
        
        if num_classes <= 0:
            logger.error(f"Could not determine num_classes from output shape {output_shape} and reg_max={reg_max}. Last dim size: {last_dim}. Expected format: num_classes + (reg_max + 1) * 4.")
            return None, None

        logger.info(f"Auto-detected num_classes: {num_classes}")
        return num_classes, reg_max

    except Exception as e:
        logger.error(f"Failed during model parameter detection: {e}")
        logger.exception("Detailed traceback:")
        return None, None

def get_class_names(num_classes):
    """Provides class names based on the detected number of classes."""
    if num_classes == 1:
        logger.info("Using default class name 'object' for single-class model.")
        return ['object']
    elif num_classes == 80:
        logger.info("Using COCO class names for 80-class model.")
        return COCO_CLASS_NAMES
    else:
        logger.warning(f"Using generic class names (class_0, class_1, ...) for {num_classes}-class model.")
        return [f'class_{i}' for i in range(num_classes)]

def filter_results_by_class(results, filter_indices):
    """Filters the detection results dictionary to keep only specified class indices."""
    if filter_indices is None or not results:
        # If filter_indices is None, it means we want all classes (no filtering argument provided initially)
        # However, the logic now defaults to 'person' if no arg is given, so this case might not be hit
        # unless the default logic changes. Keeping it for robustness.
        return results

    filtered_results = {}
    for batch_idx, batch_data in results.items():
        filtered_batch_data = {}
        for class_idx, detections in batch_data.items():
            if class_idx in filter_indices:
                filtered_batch_data[class_idx] = detections
        if filtered_batch_data: # Only add batch if it has filtered detections
            filtered_results[batch_idx] = filtered_batch_data
    return filtered_results

def benchmark_inference(interpreter, image_path, input_shape, score_threshold, num_classes, reg_max, class_names, filter_class_indices=None):
    """
    Runs inference on a single image, measures latency, and optionally filters results.

    Args:
        interpreter: The ONNX Runtime InferenceSession.
        image_path: Path to the input image file.
        input_shape: Tuple representing the model input shape (height, width).
        score_threshold: The score threshold for filtering detections.
        num_classes: Number of classes the model predicts.
        reg_max: The reg_max value for the model.
        class_names: List of class names corresponding to model output.
        filter_class_indices: Optional set of class indices to keep.

    Returns:
        A tuple containing:
        - latency_preprocess_ms: Preprocessing time in milliseconds.
        - latency_inference_ms: Model inference time in milliseconds.
        - latency_postprocess_ms: Postprocessing time in milliseconds.
        - processed_image: The image with detections visualized (or None if error).
    """
    latency_preprocess_ms = None
    latency_inference_ms = None
    latency_postprocess_ms = None
    result_image = None

    try:
        origin_img = cv2.imread(image_path)
        if origin_img is None:
            logger.warning(f"Could not read image: {image_path}")
            return latency_preprocess_ms, latency_inference_ms, latency_postprocess_ms, result_image

        # --- Preprocessing --- 
        t_start_preprocess = time.perf_counter()
        img = image_preprocess(origin_img, input_shape)
        ort_inputs = {interpreter.get_inputs()[0].name: img[None, :, :, :]}
        t_end_preprocess = time.perf_counter()
        latency_preprocess_ms = (t_end_preprocess - t_start_preprocess) * 1000

        # --- Inference --- 
        t_start_inference = time.perf_counter()
        output = interpreter.run(None, ort_inputs)
        t_end_inference = time.perf_counter()
        latency_inference_ms = (t_end_inference - t_start_inference) * 1000

        # --- Postprocessing --- 
        t_start_postprocess = time.perf_counter()
        processed_results = None
        try:
            # Use detected num_classes and reg_max
            results = post_process(output[0], num_classes, reg_max, input_shape)

            # Filter results if filter_class_indices is provided
            # Note: filter_class_indices will always be set (defaulting to 'person' index if needed)
            filtered_results = filter_results_by_class(results, filter_class_indices)

            if filtered_results and 0 in filtered_results and filtered_results[0]: # Check if filtered results exist for batch 0
                 logger.debug(f"Visualizing filtered detections for {len(filtered_results[0])} classes.")
                 result_image = visualize(filtered_results[0], origin_img.copy(), class_names, score_threshold)
            else:
                 logger.debug(f"No detections found after filtering or postprocessing returned empty for {image_path}")
                 result_image = origin_img.copy() # Return original image if no detections
        except Exception as e:
            logger.error(f"Postprocessing or Filtering failed for {image_path}: {e}")
            logger.exception("Detailed traceback:") # Log full traceback
            logger.error(f"Output shape was: {output[0].shape}")
            result_image = None # Indicate error during postprocessing
        t_end_postprocess = time.perf_counter()
        latency_postprocess_ms = (t_end_postprocess - t_start_postprocess) * 1000

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        logger.exception("Detailed traceback:") # Log full traceback
        latency_preprocess_ms = latency_preprocess_ms if 't_start_preprocess' in locals() else None
        latency_inference_ms = latency_inference_ms if 't_start_inference' in locals() else None
        latency_postprocess_ms = latency_postprocess_ms if 't_start_postprocess' in locals() else None
        result_image = None

    return latency_preprocess_ms, latency_inference_ms, latency_postprocess_ms, result_image

if __name__ == '__main__':
    args = make_parser().parse_args()
    args.input_shape = tuple(map(int, args.input_shape.split(',')))
    reg_max = DEFAULT_REG_MAX

    # --- Handle --classes argument default --- 
    if args.classes is None:
        logger.info("No --classes specified, defaulting to ['person'].")
        args.classes = ['person']

    logger.info(f"Loading ONNX model from: {args.model}")
    try:
        interpreter = onnxruntime.InferenceSession(args.model, providers=['CPUExecutionProvider'])
        logger.info("Model loaded successfully using CPUExecutionProvider.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        exit()

    # --- Auto-detect model parameters --- 
    num_classes, detected_reg_max = detect_model_parameters(interpreter, args.input_shape, reg_max)
    if num_classes is None:
        logger.error("Exiting due to failure in detecting model parameters.")
        exit()
    
    # --- Get class names based on detected number --- 
    class_names = get_class_names(num_classes)
    class_name_to_id = {name: i for i, name in enumerate(class_names)}

    # --- Process --classes argument using detected class names --- 
    filter_class_indices = set()
    invalid_classes = []
    logger.info(f"Attempting to filter for: {args.classes}")
    for name in args.classes:
        if name in class_name_to_id:
            filter_class_indices.add(class_name_to_id[name])
        else:
            invalid_classes.append(name)
    if invalid_classes:
        logger.warning(f"Invalid class names provided for filtering: {', '.join(invalid_classes)}. Ignoring them.")
        logger.warning(f"Valid class names for this model are: {', '.join(class_names)}")
    if not filter_class_indices:
        logger.error("No valid classes specified for filtering. Cannot proceed.")
        logger.error(f"Please provide valid class names from: {', '.join(class_names)}")
        exit() # Exit if no valid classes remain after filtering
    else:
         logger.info(f"Successfully set filter for classes: {[class_names[i] for i in filter_class_indices]} (Indices: {filter_class_indices})")

    # --- Dataset and Output Path Setup --- 
    if not os.path.isdir(args.dataset_path):
        logger.error(f"Dataset path not found or is not a directory: {args.dataset_path}")
        exit()
    os.makedirs(args.output_path, exist_ok=True)

    # --- Find Images --- 
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.dataset_path, ext)))
    if not image_files:
        logger.error(f"No images found in directory: {args.dataset_path}")
        exit()
    logger.info(f"Found {len(image_files)} images in {args.dataset_path}")

    # --- Benchmarking --- 
    total_preprocess_time_ms = 0
    total_inference_time_ms = 0
    total_postprocess_time_ms = 0
    processed_image_count = 0
    successful_processing_count = 0

    # Optional: Warm-up run
    logger.info("Performing warm-up inference...")
    _ = benchmark_inference(interpreter, image_files[0], args.input_shape, args.score_thr, num_classes, reg_max, class_names, filter_class_indices)
    logger.info("Warm-up complete.")

    logger.info("Starting benchmark...")
    for image_path in image_files:
        logger.info(f"Processing: {os.path.basename(image_path)}")
        lat_pre, lat_inf, lat_post, result_img = benchmark_inference(interpreter, image_path, args.input_shape, args.score_thr, num_classes, reg_max, class_names, filter_class_indices)

        processed_image_count += 1
        if lat_pre is not None and lat_inf is not None and lat_post is not None:
            total_preprocess_time_ms += lat_pre
            total_inference_time_ms += lat_inf
            total_postprocess_time_ms += lat_post
            successful_processing_count += 1

            if result_img is not None and args.output_path:
                output_filename = os.path.join(args.output_path, os.path.basename(image_path))
                try:
                    cv2.imwrite(output_filename, result_img)
                    logger.debug(f"Saved output image to {output_filename}")
                except Exception as e:
                    logger.warning(f"Could not save output image {output_filename}: {e}")
        else:
             logger.warning(f"Skipping benchmark calculation for image due to processing error: {image_path}")

    # --- Results --- 
    if successful_processing_count > 0:
        avg_latency_preprocess_ms = total_preprocess_time_ms / successful_processing_count
        avg_latency_inference_ms = total_inference_time_ms / successful_processing_count
        avg_latency_postprocess_ms = total_postprocess_time_ms / successful_processing_count
        avg_latency_total_ms = avg_latency_preprocess_ms + avg_latency_inference_ms + avg_latency_postprocess_ms

        fps_inference = 1000 / avg_latency_inference_ms if avg_latency_inference_ms > 0 else 0
        fps_total = 1000 / avg_latency_total_ms if avg_latency_total_ms > 0 else 0

        logger.info("--- Benchmark Results ---")
        logger.info(f"Model: {os.path.basename(args.model)}")
        logger.info(f"Detected Classes: {num_classes}")
        logger.info(f"Total Images Found: {len(image_files)}")
        logger.info(f"Images Processed (Attempted): {processed_image_count}")
        logger.info(f"Images Successfully Benchmarked: {successful_processing_count}")
        logger.info(f"Input Shape: {args.input_shape}")
        logger.info(f"Score Threshold: {args.score_thr}")
        # Always report the classes being filtered, even if default
        logger.info(f"Filtered Classes: {[class_names[i] for i in filter_class_indices]}")
        logger.info(f"Average Preprocessing Latency: {avg_latency_preprocess_ms:.2f} ms")
        logger.info(f"Average Inference Latency: {avg_latency_inference_ms:.2f} ms")
        logger.info(f"Average Postprocessing Latency: {avg_latency_postprocess_ms:.2f} ms")
        logger.info(f"Average Total Latency (Pre + Inf + Post): {avg_latency_total_ms:.2f} ms")
        logger.info(f"Inference FPS: {fps_inference:.2f}")
        logger.info(f"End-to-End FPS (Pre + Inf + Post): {fps_total:.2f}")
        logger.info("--- ----------------- ---")
    else:
        logger.error("No images were processed successfully. Cannot calculate benchmark results.")

    logger.info(f"Benchmark finished. Output images (if any) saved to: {args.output_path}")

