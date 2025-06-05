'''
Benchmark script for NanoDet ONNX models to calculate latency and FPS on an image dataset.

This script takes an ONNX model and a directory of images as input, runs inference
on each image, and calculates the average inference latency and FPS.
'''

import argparse
import os
import glob
import time

import cv2
import numpy as np
import onnxruntime
from loguru import logger

# Assuming utils.py is in the same directory or accessible in the Python path
from utils.utils import image_preprocess, post_process, visualize

# Define class names if needed for visualization or post-processing
# Adjust this list based on the specific NanoDet model being used
class_names = ['person'] # Example for a person detector

def make_parser():
    """Creates an argument parser for the benchmarking script."""
    parser = argparse.ArgumentParser("ONNX Model Benchmark")
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
    return parser

def benchmark_inference(interpreter, image_path, input_shape):
    """
    Runs inference on a single image and measures latency.

    Args:
        interpreter: The ONNX Runtime InferenceSession.
        image_path: Path to the input image file.
        input_shape: Tuple representing the model input shape (height, width).

    Returns:
        A tuple containing:
        - latency_preprocess_ms: Preprocessing time in milliseconds.
        - latency_inference_ms: Model inference time in milliseconds.
        - latency_postprocess_ms: Postprocessing time in milliseconds.
        - processed_image: The image with detections visualized (or None if error).
    """
    try:
        origin_img = cv2.imread(image_path)
        if origin_img is None:
            logger.warning(f"Could not read image: {image_path}")
            return None, None, None, None

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
        # Assuming post_process and visualize are defined in utils.utils
        # Note: Post-processing might depend on model output structure.
        # Adjust `num_classes` and `num_coords` if necessary.
        # The original script used len(class_names) and 7 (likely for bbox + score + class_id + something else?)
        # Let's stick to that for now, but it might need adjustment.
        results = post_process(output[0], len(class_names), 7, input_shape)
        result_image = visualize(results[0], origin_img.copy(), class_names, args.score_thr)
        t_end_postprocess = time.perf_counter()
        latency_postprocess_ms = (t_end_postprocess - t_start_postprocess) * 1000

        return latency_preprocess_ms, latency_inference_ms, latency_postprocess_ms, result_image

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None, None, None, None

if __name__ == '__main__':
    args = make_parser().parse_args()
    args.input_shape = tuple(map(int, args.input_shape.split(',')))

    logger.info(f"Loading ONNX model from: {args.model}")
    try:
        interpreter = onnxruntime.InferenceSession(args.model)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        exit()

    if not os.path.isdir(args.dataset_path):
        logger.error(f"Dataset path not found or is not a directory: {args.dataset_path}")
        exit()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Find image files (adjust extensions if needed)
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.dataset_path, ext)))

    if not image_files:
        logger.error(f"No images found in directory: {args.dataset_path}")
        exit()

    logger.info(f"Found {len(image_files)} images in {args.dataset_path}")

    total_preprocess_time_ms = 0
    total_inference_time_ms = 0
    total_postprocess_time_ms = 0
    processed_image_count = 0

    # Optional: Warm-up run
    logger.info("Performing warm-up inference...")
    _ = benchmark_inference(interpreter, image_files[0], args.input_shape)
    logger.info("Warm-up complete.")

    logger.info("Starting benchmark...")
    for image_path in image_files:
        logger.debug(f"Processing: {os.path.basename(image_path)}")
        lat_pre, lat_inf, lat_post, result_img = benchmark_inference(interpreter, image_path, args.input_shape)

        if lat_inf is not None: # Check if inference was successful
            total_preprocess_time_ms += lat_pre
            total_inference_time_ms += lat_inf
            total_postprocess_time_ms += lat_post
            processed_image_count += 1

            # Optionally save the output image
            if result_img is not None and args.output_path:
                output_filename = os.path.join(args.output_path, os.path.basename(image_path))
                try:
                    cv2.imwrite(output_filename, result_img)
                except Exception as e:
                    logger.warning(f"Could not save output image {output_filename}: {e}")
        else:
             logger.warning(f"Skipping image due to processing error: {image_path}")


    if processed_image_count > 0:
        avg_latency_preprocess_ms = total_preprocess_time_ms / processed_image_count
        avg_latency_inference_ms = total_inference_time_ms / processed_image_count
        avg_latency_postprocess_ms = total_postprocess_time_ms / processed_image_count
        avg_latency_total_ms = avg_latency_preprocess_ms + avg_latency_inference_ms + avg_latency_postprocess_ms

        fps_inference = 1000 / avg_latency_inference_ms if avg_latency_inference_ms > 0 else 0
        fps_total = 1000 / avg_latency_total_ms if avg_latency_total_ms > 0 else 0

        logger.info("--- Benchmark Results ---")
        logger.info(f"Processed Images: {processed_image_count}/{len(image_files)}")
        logger.info(f"Input Shape: {args.input_shape}")
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

