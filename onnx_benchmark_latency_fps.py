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
from utils.utils_fixed import image_preprocess, post_process, visualize

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
DEFAULT_REG_MAX = 7

def make_parser():
    """Creates an argument parser for the benchmarking script."""
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
        "--classes",
        nargs='+',
        type=str,
        default=None,
        help="Optional list of class names to filter results for (e.g., person car bus). If not provided, defaults to 'person'. If model is single-class, providing one name here will override the default 'object' name.",
    )
    return parser

def detect_model_parameters(interpreter, input_shape, reg_max):
    """Runs a dummy inference to detect output shape and infer num_classes."""
    logger.info("Attempting to detect model parameters (num_classes)...")
    try:
        dummy_input = np.zeros((1, 3, input_shape[0], input_shape[1]), dtype=np.float32)
        ort_inputs = {interpreter.get_inputs()[0].name: dummy_input}
        output = interpreter.run(None, ort_inputs)
        output_shape = output[0].shape
        logger.info(f"Detected model output shape: {output_shape}")
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

def get_default_class_names(num_classes):
    """Provides default class names (COCO or generic) based on the detected number of classes."""
    # This function is now only for default names when user doesn't override single-class name
    if num_classes == 1:
        # This case should ideally be handled by the override logic in main
        # but keep a fallback name here.
        logger.info("Using default class name 'object' for single-class model (fallback)." )
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
        return results
    filtered_results = {}
    for batch_idx, batch_data in results.items():
        filtered_batch_data = {}
        for class_idx, detections in batch_data.items():
            if class_idx in filter_indices:
                filtered_batch_data[class_idx] = detections
        if filtered_batch_data:
            filtered_results[batch_idx] = filtered_batch_data
    return filtered_results

def benchmark_inference(interpreter, image_path, input_shape, score_threshold, num_classes, reg_max, class_names, filter_class_indices=None):
    """
    Runs inference on a single image, measures latency, and optionally filters results.
    Args as defined in main block.
    Returns tuple: (latency_preprocess_ms, latency_inference_ms, latency_postprocess_ms, processed_image)
    """
    latency_preprocess_ms, latency_inference_ms, latency_postprocess_ms, result_image = None, None, None, None
    try:
        origin_img = cv2.imread(image_path)
        if origin_img is None:
            logger.warning(f"Could not read image: {image_path}")
            return latency_preprocess_ms, latency_inference_ms, latency_postprocess_ms, result_image

        t_start_preprocess = time.perf_counter()
        img = image_preprocess(origin_img, input_shape)
        ort_inputs = {interpreter.get_inputs()[0].name: img[None, :, :, :]}
        latency_preprocess_ms = (time.perf_counter() - t_start_preprocess) * 1000

        t_start_inference = time.perf_counter()
        output = interpreter.run(None, ort_inputs)
        latency_inference_ms = (time.perf_counter() - t_start_inference) * 1000

        t_start_postprocess = time.perf_counter()
        try:
            # Use the provided num_classes, reg_max, and class_names
            results = post_process(output[0], num_classes, reg_max, input_shape)
            filtered_results = filter_results_by_class(results, filter_class_indices)

            if filtered_results and 0 in filtered_results and filtered_results[0]:
                 logger.debug(f"Visualizing filtered detections for {len(filtered_results[0])} classes.")
                 # Pass the potentially overridden class_names list to visualize
                 # Pass the input_shape to ensure proper scaling
                 result_image = visualize(filtered_results[0], origin_img.copy(), class_names, score_threshold, model_input_size=input_shape)
            else:
                 logger.debug(f"No detections found after filtering or postprocessing returned empty for {image_path}")
                 result_image = origin_img.copy()
        except Exception as e:
            logger.error(f"Postprocessing or Filtering failed for {image_path}: {e}")
            logger.exception("Detailed traceback:")
            logger.error(f"Output shape was: {output[0].shape}")
            result_image = None
        latency_postprocess_ms = (time.perf_counter() - t_start_postprocess) * 1000

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        logger.exception("Detailed traceback:")
        latency_preprocess_ms = latency_preprocess_ms if 't_start_preprocess' in locals() else None
        latency_inference_ms = latency_inference_ms if 't_start_inference' in locals() else None
        latency_postprocess_ms = latency_postprocess_ms if 't_start_postprocess' in locals() else None
        result_image = None

    return latency_preprocess_ms, latency_inference_ms, latency_postprocess_ms, result_image

if __name__ == '__main__':
    args = make_parser().parse_args()
    args.input_shape = tuple(map(int, args.input_shape.split(',')))
    reg_max = DEFAULT_REG_MAX

    logger.info(f"Loading ONNX model from: {args.model}")
    try:
        interpreter = onnxruntime.InferenceSession(args.model, providers=['CPUExecutionProvider'])
        logger.info("Model loaded successfully using CPUExecutionProvider.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        exit()

    num_classes, detected_reg_max = detect_model_parameters(interpreter, args.input_shape, reg_max)
    if num_classes is None:
        logger.error("Exiting due to failure in detecting model parameters.")
        exit()

    # --- Determine Class Names (handle single-class override) --- 
    user_provided_classes = args.classes
    user_provided_single_class_name = None
    if user_provided_classes and len(user_provided_classes) == 1:
        user_provided_single_class_name = user_provided_classes[0]

    if num_classes == 1 and user_provided_single_class_name:
        logger.info(f"Detected single-class model. Using user-provided name: '{user_provided_single_class_name}'")
        class_names = [user_provided_single_class_name]
    else:
        # Use default logic (COCO or generic) if multi-class or user didn't provide a single specific name
        class_names = get_default_class_names(num_classes)
        if num_classes == 1 and user_provided_classes and len(user_provided_classes) > 1:
             logger.warning(f"User provided multiple class names ({user_provided_classes}) but model is single-class. Using default name '{class_names[0]}'.")
        elif num_classes == 1 and not user_provided_classes:
             logger.info(f"Detected single-class model and no classes specified. Using default name '{class_names[0]}'.")

    class_name_to_id = {name: i for i, name in enumerate(class_names)}

    # --- Determine classes to filter for --- 
    target_classes_for_filtering = []
    if args.classes is None: # No --classes arg provided
        default_target = 'person'
        if default_target in class_name_to_id:
             logger.info(f"No --classes specified, defaulting to ['{default_target}'].")
             target_classes_for_filtering = [default_target]
        elif num_classes == 1:
             # If default 'person' isn't valid, but it's a single class model, use its assigned name
             single_class_name = class_names[0]
             logger.info(f"No --classes specified, default '{default_target}' invalid for this model. Defaulting to the single class: ['{single_class_name}'].")
             target_classes_for_filtering = [single_class_name]
        else:
             logger.error(f"No --classes specified, and default '{default_target}' is not valid for this multi-class model. Cannot determine default filter.")
             logger.error(f"Please specify classes using --classes. Valid names: {', '.join(class_names)}")
             exit()
    else: # --classes arg was provided
        target_classes_for_filtering = args.classes

    # --- Validate and get indices for the target filter classes --- 
    filter_class_indices = set()
    final_filter_classes_names = []
    invalid_classes = []
    valid_classes_found = False
    logger.info(f"Attempting to filter for target classes: {target_classes_for_filtering}")
    for name in target_classes_for_filtering:
        if name in class_name_to_id:
            filter_class_indices.add(class_name_to_id[name])
            final_filter_classes_names.append(name)
            valid_classes_found = True
        else:
            invalid_classes.append(name)

    if invalid_classes:
        logger.warning(f"Invalid class names provided for filtering: {', '.join(invalid_classes)}. Ignoring them.")
        logger.warning(f"Valid class names for this model are: {', '.join(class_names)}")

    if not valid_classes_found:
        logger.error("No valid classes specified or found for filtering. Cannot proceed.")
        logger.error(f"Please provide valid class names from: {', '.join(class_names)}")
        exit()
    else:
         logger.info(f"Successfully set filter for classes: {final_filter_classes_names} (Indices: {filter_class_indices})")

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
    total_preprocess_time_ms, total_inference_time_ms, total_postprocess_time_ms = 0, 0, 0
    processed_image_count, successful_processing_count = 0, 0

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
        logger.info(f"Detected Classes: {num_classes} ({', '.join(class_names)})") # Show assigned names
        logger.info(f"Total Images Found: {len(image_files)}")
        logger.info(f"Images Processed (Attempted): {processed_image_count}")
        logger.info(f"Images Successfully Benchmarked: {successful_processing_count}")
        logger.info(f"Input Shape: {args.input_shape}")
        logger.info(f"Score Threshold: {args.score_thr}")
        logger.info(f"Filtered Classes: {final_filter_classes_names}") # Report the names used for filtering
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
