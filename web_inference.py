"""
Edge Impulse OCR Web Interface

This module provides a Flask-based web server for real-time OCR inference
using Edge Impulse models. It supports model cascading with a detector model
(to find text regions) and a predictor/recognizer model (to read the text).

Compatible with Arduino App Lab as a standalone Flask app (no bricks).

Designed to run on:
- Arduino UNO Q (linux-aarch64 models, port 5001 for App Lab)
- macOS ARM64 (mac-arm64 models)
- Other Linux platforms

Usage:
    # On Arduino UNO Q (auto-detects models, auto-starts on the default RTSP stream):
    python3 python/main.py

    # Use a specific RTSP stream as the video source:
    python3 web_inference.py --rtsp-url rtsp://x.x.x.x:554/mjpeg/1

    # Or with explicit model paths:
    python3 web_inference.py \
        --detect-file ./models/linux-aarch64/detect-v3-640-480-i8.eim \
        --predict-file ./models/linux-aarch64/recognizer-320-48-f32.eim \
        --dict-file source_models/rec_en_dict.txt

    # Via Arduino App Lab CLI:
    arduino-app-cli app start .

Access the web UI at:
    - http://localhost:5001
    - http://<device-ip>:5001
"""

import argparse
import base64
import glob
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
from edge_impulse_linux.image import ImageImpulseRunner
from flask import Flask, Response, jsonify, render_template, request

# =============================================================================
# CONFIGURATION
# =============================================================================

# Color for bounding boxes (BGR format for OpenCV)
BOX_COLOR = (0, 255, 0)  # Green

# Default web server settings
DEFAULT_HOST = '0.0.0.0'  # Listen on all interfaces for network access
DEFAULT_PORT = 5001       # Port 5001 for App Lab compatibility (port 7000 is reserved)

# Default video source: RTSP network camera stream.
# Override with --rtsp-url, or select a wired camera in the web UI.
DEFAULT_RTSP_URL = 'rtsp://10.0.0.124:554/mjpeg/1'


# =============================================================================
# LOGGING UTILITY
# =============================================================================

def _log(msg: str, *, enabled: bool = True):
    """
    Print a log message with flush to ensure immediate output.

    Args:
        msg: The message to print
        enabled: If False, the message is suppressed
    """
    if not enabled:
        return
    print(msg, flush=True)


# =============================================================================
# CAMERA DETECTION
# =============================================================================

def detect_cameras(max_cameras: int = 10) -> list[dict]:
    """
    Detect available camera devices.

    On Linux, uses v4l2 to get device info and filters out encoder/decoder devices.
    Falls back to OpenCV index probing on other platforms.

    Args:
        max_cameras: Maximum number of camera indices to check

    Returns:
        List of dictionaries with 'index' and 'name' for each camera
    """
    cameras = []

    # On Linux, check /dev/video* devices with v4l2 info
    if os.path.exists('/dev'):
        video_devices = sorted(glob.glob('/dev/video*'))

        for device in video_devices:
            try:
                index = int(device.replace('/dev/video', ''))

                # Get device info from v4l2
                device_name = None
                device_caps = None
                is_capture_device = False

                try:
                    import subprocess

                    # Get device info
                    result = subprocess.run(
                        ['v4l2-ctl', '--device', device, '--info'],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        output = result.stdout
                        for line in output.split('\n'):
                            if 'Card type' in line:
                                device_name = line.split(':', 1)[1].strip()
                            if 'Device Caps' in line or 'Capabilities' in line:
                                device_caps = line

                        # Check if it's a video capture device (not encoder/decoder)
                        is_capture_device = 'Video Capture' in output

                        # Skip encoder/decoder devices
                        if 'encoder' in (device_name or '').lower() or 'decoder' in (device_name or '').lower():
                            continue
                        if 'venus' in (device_name or '').lower():
                            continue
                except Exception:
                    # If v4l2-ctl fails, try to open anyway
                    is_capture_device = True

                if not is_capture_device:
                    continue

                # Try to open and read from the camera
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    ret, frame = cap.read()
                    cap.release()

                    if ret and frame is not None and frame.size > 0:
                        # Check if it's likely an IR camera
                        is_ir = False
                        camera_type = "RGB"

                        if len(frame.shape) == 3:
                            avg_brightness = frame.mean()
                            b, g, r = cv2.split(frame)
                            color_diff = abs(float(b.mean()) - float(r.mean()))

                            # IR cameras are usually grayscale or very dark
                            if avg_brightness < 10:
                                is_ir = True
                                camera_type = "IR"
                            elif color_diff < 3 and avg_brightness < 80:
                                is_ir = True
                                camera_type = "IR"

                        # Build descriptive name
                        if device_name:
                            # For Logitech BRIO, identify the stream type
                            if 'BRIO' in device_name or 'Logitech' in device_name:
                                if is_ir:
                                    name = f"{device_name} IR ({device})"
                                else:
                                    name = f"{device_name} ({device})"
                            else:
                                name = f"{device_name} ({device})"
                        else:
                            name = f"Camera {index} ({device})"

                        if is_ir:
                            name += " [IR]"

                        cameras.append({
                            'index': index,
                            'name': name,
                            'is_ir': is_ir,
                            'device_name': device_name or ''
                        })
            except (ValueError, Exception):
                continue

        # Sort: RGB cameras first, then by index
        cameras.sort(key=lambda x: (x.get('is_ir', False), x['index']))

    # Fallback: try indices directly (for macOS and other platforms)
    if not cameras:
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        cameras.append({
                            'index': i,
                            'name': f'Camera {i}'
                        })
            except Exception:
                continue

    return cameras


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

class _InitTimeout(Exception):
    """Exception raised when model initialization times out."""
    pass


def _init_runner_with_timeout(runner: ImageImpulseRunner, *, seconds: int, debug: bool):
    """
    Initialize an Edge Impulse runner with a timeout.

    The edge_impulse_linux.runner can wait forever for the IPC socket if the EIM
    never creates it (e.g., wrong architecture, missing dependencies). This function
    puts a hard cap on initialization time.

    Args:
        runner: The ImageImpulseRunner instance to initialize
        seconds: Maximum time to wait for initialization (0 = no timeout)
        debug: Enable verbose debug output from the runner

    Returns:
        Model info dictionary from the runner

    Raises:
        _InitTimeout: If initialization exceeds the timeout
    """
    if seconds <= 0:
        return runner.init(debug=debug)

    def _on_alarm(_signum, _frame):
        raise _InitTimeout(f'EIM init timed out after {seconds}s')

    prev = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _on_alarm)
    try:
        signal.alarm(int(seconds))
        return runner.init(debug=debug)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# =============================================================================
# OCR DICTIONARY AND PREDICTION PARSING
# =============================================================================

def parse_paddle_ocr_dictionary(file_path: str) -> list[str]:
    """
    Load the character dictionary file for the OCR predictor.

    The dictionary maps output indices to characters. Each line in the file
    represents one character in the vocabulary.

    Args:
        file_path: Path to the dictionary text file

    Returns:
        List of characters in the dictionary
    """
    return Path(file_path).read_text(encoding='utf-8').split('\n')


def parse_paddle_ocr_predictor(pred_resp: dict, dictionary: list[str]):
    """
    Parse the raw predictor output into readable text.

    The predictor outputs a tensor of probabilities for each character position.
    This function applies CTC (Connectionist Temporal Classification) decoding
    to convert probabilities to text.

    Args:
        pred_resp: Raw response from the predictor model
        dictionary: Character dictionary for decoding

    Returns:
        Dictionary with 'output' (text) and 'avgConfidence', or None if no text

    Raises:
        RuntimeError: If the predictor output is malformed
    """
    result = pred_resp.get('result', {})
    freeform = result.get('freeform')

    if freeform is None:
        raise RuntimeError('Predictor model did not return freeform results')
    if not isinstance(freeform, list) or len(freeform) == 0:
        raise RuntimeError('Predictor freeform results are empty')

    tensor = freeform[0]
    if not isinstance(tensor, list):
        raise RuntimeError('Predictor freeform[0] was not a list')

    # Calculate the number of time steps (columns) in the output
    # Each column has (dictionary_size + 1) values (includes blank label)
    cols = len(tensor) / (len(dictionary) + 1)
    if cols % 1 != 0:
        raise RuntimeError(
            'Invalid output shape (probably incorrect dictionary?). '
            f'Dict size={(len(dictionary) + 1)}, output tensor length={len(tensor)}, '
            f'detected columns={cols} (expected to be an integer)'
        )

    # Reshape flat tensor into 2D array [time_steps, vocab_size]
    reshaped: list[list[float]] = []
    step = len(dictionary) + 1
    for ix in range(0, len(tensor), step):
        reshaped.append(tensor[ix:ix + step])

    # CTC decoding: take argmax at each time step, skip blank labels (index 0)
    all_confidences = 0.0
    predicted_chars_length = 0
    output = ''

    for col in reshaped:
        hi_val = 0.0
        hi_ix = -1
        for ix, v in enumerate(col):
            if v > hi_val:
                hi_val = float(v)
                hi_ix = ix

        # Index 0 is the CTC blank label - skip it
        if hi_ix != 0:
            output += dictionary[hi_ix - 1]
            all_confidences += hi_val
            predicted_chars_length += 1

    if predicted_chars_length == 0:
        return None

    return {
        'output': output,
        'avgConfidence': all_confidences / predicted_chars_length,
    }


# =============================================================================
# BOUNDING BOX COORDINATE MAPPING
# =============================================================================

def _map_prediction_to_original_image(
    resize_mode: str,
    box_norm: dict,
    img: dict,
):
    """
    Map normalized bounding box coordinates back to original image coordinates.

    Different resize modes affect how coordinates are transformed:
    - squash: Simple scaling (aspect ratio not preserved)
    - fit-short: Scale to fit shorter dimension, crop excess
    - fit-long: Scale to fit longer dimension, pad with letterbox

    Args:
        resize_mode: The resize mode used by the model ('squash', 'fit-short', 'fit-long')
        box_norm: Normalized box coordinates (x, y, width, height in 0-1 range)
        img: Image dimensions (clientWidth/Height = original, referenceWidth/Height = model input)

    Returns:
        Dictionary with mapped x, y, width, height in original image coordinates
    """
    x = 0.0
    y = 0.0
    width = 0.0
    height = 0.0

    if resize_mode == 'squash':
        # Simple scaling - just multiply by scale factors
        scale_x = img['clientWidth'] / img['referenceWidth']
        scale_y = img['clientHeight'] / img['referenceHeight']
        x = max(0.0, min(box_norm['x'] * img['referenceWidth'] * scale_x, img['clientWidth']))
        y = max(0.0, min(box_norm['y'] * img['referenceHeight'] * scale_y, img['clientHeight']))
        width = min(box_norm['width'] * img['referenceWidth'] * scale_x, img['clientWidth'] - x)
        height = min(box_norm['height'] * img['referenceHeight'] * scale_y, img['clientHeight'] - y)

    elif resize_mode == 'fit-short':
        # Scale to fit shorter dimension, accounting for cropping
        reference_aspect = img['referenceWidth'] / img['referenceHeight']
        client_aspect = img['clientWidth'] / img['clientHeight']

        scale = 1.0
        offset_x = 0.0
        offset_y = 0.0
        if client_aspect > reference_aspect:
            scale = img['clientHeight'] / img['referenceHeight']
            crop_width = img['clientWidth'] - img['referenceWidth'] * scale
            offset_x = crop_width / 2.0
        else:
            scale = img['clientWidth'] / img['referenceWidth']
            crop_height = img['clientHeight'] - img['referenceHeight'] * scale
            offset_y = crop_height / 2.0

        x = max(0.0, min(box_norm['x'] * img['referenceWidth'] * scale + offset_x, img['clientWidth']))
        y = max(0.0, min(box_norm['y'] * img['referenceHeight'] * scale + offset_y, img['clientHeight']))
        width = min(box_norm['width'] * img['referenceWidth'] * scale, img['clientWidth'] - x)
        height = min(box_norm['height'] * img['referenceHeight'] * scale, img['clientHeight'] - y)

    elif resize_mode == 'fit-long':
        # Scale to fit longer dimension, accounting for letterboxing
        reference_aspect = img['referenceWidth'] / img['referenceHeight']
        client_aspect = img['clientWidth'] / img['clientHeight']
        if client_aspect > reference_aspect:
            scale = img['clientWidth'] / img['referenceWidth']
            pad_height = img['clientHeight'] - img['referenceHeight'] * scale
            pad_y = pad_height / 2.0
            x = max(0.0, min(box_norm['x'] * img['referenceWidth'] * scale, img['clientWidth']))
            y = max(0.0, min(box_norm['y'] * img['referenceHeight'] * scale + pad_y, img['clientHeight']))
            width = min(box_norm['width'] * img['referenceWidth'] * scale, img['clientWidth'] - x)
            height = min(box_norm['height'] * img['referenceHeight'] * scale, img['clientHeight'] - y)
        else:
            scale = img['clientHeight'] / img['referenceHeight']
            pad_width = img['clientWidth'] - img['referenceWidth'] * scale
            pad_x = pad_width / 2.0
            x = max(0.0, min(box_norm['x'] * img['referenceWidth'] * scale + pad_x, img['clientWidth']))
            y = max(0.0, min(box_norm['y'] * img['referenceHeight'] * scale, img['clientHeight']))
            width = min(box_norm['width'] * img['referenceWidth'] * scale, img['clientWidth'] - x)
            height = min(box_norm['height'] * img['referenceHeight'] * scale, img['clientHeight'] - y)
    else:
        # Unknown resize mode - return zero box
        return {'x': 0.0, 'y': 0.0, 'width': 0.0, 'height': 0.0}

    return {'x': x, 'y': y, 'width': width, 'height': height}


def map_bounding_box_back_to_original_image(
    detector_model_info: dict,
    resized_info: dict,
    bb: dict,
):
    """
    Transform a bounding box from model coordinates to original image coordinates.

    This is necessary because the model operates on resized/cropped images,
    but we want to draw boxes on the original camera frame.

    Args:
        detector_model_info: Model metadata containing resize mode
        resized_info: Dictionary with originalWidth/Height and newWidth/Height
        bb: Bounding box from model output (x, y, width, height, label, value)

    Returns:
        Transformed bounding box with integer coordinates
    """
    # Get resize mode from model parameters
    resize_mode = detector_model_info.get('model_parameters', {}).get('image_resize_mode')

    # Normalize resize mode names to match our implementation
    if resize_mode == 'fit-shortest':
        resize_mode = 'fit-short'
    elif resize_mode == 'fit-longest':
        resize_mode = 'fit-long'
    elif resize_mode in (None, '', 'not-reported'):
        resize_mode = 'squash'

    # Map coordinates
    mapped = _map_prediction_to_original_image(
        resize_mode,
        {
            'x': bb['x'] / resized_info['newWidth'],
            'y': bb['y'] / resized_info['newHeight'],
            'width': bb['width'] / resized_info['newWidth'],
            'height': bb['height'] / resized_info['newHeight'],
        },
        {
            'clientWidth': resized_info['originalWidth'],
            'clientHeight': resized_info['originalHeight'],
            'referenceWidth': resized_info['newWidth'],
            'referenceHeight': resized_info['newHeight'],
        },
    )

    # Convert to integers and clamp to image bounds
    x = int(mapped['x'] // 1)
    y = int(mapped['y'] // 1)
    width = int(-(-mapped['width'] // 1))  # Ceiling division
    height = int(-(-mapped['height'] // 1))

    # Ensure box stays within image bounds
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + width > resized_info['originalWidth']:
        width = resized_info['originalWidth'] - x
    if y + height > resized_info['originalHeight']:
        height = resized_info['originalHeight'] - y

    return {
        'label': bb.get('label', ''),
        'value': bb.get('value', 0.0),
        'x': x,
        'y': y,
        'width': width,
        'height': height,
    }


# =============================================================================
# OCR INFERENCE ENGINE
# =============================================================================

class OCREngine:
    """
    OCR inference engine using Edge Impulse models.

    This class manages the model cascading pipeline:
    1. Detector model finds text regions in the image
    2. Predictor model recognizes text in each detected region

    The engine runs inference in a background thread to maintain
    smooth video streaming.
    """

    def __init__(
        self,
        detect_file: str,
        predict_file: str,
        dict_file: str,
        min_box_area: int = 20,
        init_timeout_s: int = 30,
        debug: bool = False,
    ):
        """
        Initialize the OCR engine.

        Args:
            detect_file: Path to the detector .eim model file
            predict_file: Path to the predictor/recognizer .eim model file
            dict_file: Path to the character dictionary file
            min_box_area: Minimum bounding box area to process (filters noise)
            init_timeout_s: Timeout for model initialization
            debug: Enable verbose debug output
        """
        self.detect_file = detect_file
        self.predict_file = predict_file
        self.dict_file = dict_file
        self.min_box_area = min_box_area
        self.init_timeout_s = init_timeout_s
        self.debug = debug

        # Model runners (initialized later)
        self.detector_runner: Optional[ImageImpulseRunner] = None
        self.predictor_runner: Optional[ImageImpulseRunner] = None
        self.detector_model_info: Optional[dict] = None
        self.predictor_model_info: Optional[dict] = None
        self.predictor_dict: Optional[list[str]] = None

        # Models loaded flag
        self._models_loaded = False

        # Video source (set when starting): int = camera device index, str = RTSP/HTTP URL
        self.source = None
        self.is_stream_source = False
        self.cap: Optional[cv2.VideoCapture] = None

        # Thread-safe state for sharing between inference and web threads
        self._lock = threading.Lock()
        self._current_frame: Optional[bytes] = None  # JPEG-encoded frame
        self._current_results: list[dict] = []
        self._frame_count = 0
        self._detector_ms = 0
        self._predictor_ms = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0

        # Control flags
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None

    def load_models(self):
        """
        Load the ML models (but don't start camera yet).

        This is called once at startup to pre-load the models.
        """
        if self._models_loaded:
            return

        _log('Loading character dictionary...')
        self.predictor_dict = parse_paddle_ocr_dictionary(self.dict_file)
        _log(f'  Dictionary loaded: {len(self.predictor_dict)} characters')

        _log('Initializing detector model...')
        self.detector_runner = ImageImpulseRunner(self.detect_file)
        try:
            self.detector_model_info = _init_runner_with_timeout(
                self.detector_runner,
                seconds=self.init_timeout_s,
                debug=self.debug,
            )
        except Exception as ex:
            _log(f'Failed to initialize detector model: {ex}')
            raise

        _log('Initializing predictor model...')
        self.predictor_runner = ImageImpulseRunner(self.predict_file)
        try:
            self.predictor_model_info = _init_runner_with_timeout(
                self.predictor_runner,
                seconds=self.init_timeout_s,
                debug=self.debug,
            )
        except Exception as ex:
            _log(f'Failed to initialize predictor model: {ex}')
            raise

        # Log model information
        det_proj = self.detector_model_info.get('project', {})
        pred_proj = self.predictor_model_info.get('project', {})
        _log('Models loaded:')
        _log(f"  Detector:  {det_proj.get('owner')} / {det_proj.get('name')} (v{det_proj.get('deploy_version')})")
        _log(f"  Predictor: {pred_proj.get('owner')} / {pred_proj.get('name')} (v{pred_proj.get('deploy_version')})")

        self._models_loaded = True

    def get_model_info(self) -> tuple[str, str]:
        """
        Get human-readable model information.

        Returns:
            Tuple of (detector_info, predictor_info) strings
        """
        if not self.detector_model_info or not self.predictor_model_info:
            return ('Not loaded', 'Not loaded')

        det_proj = self.detector_model_info.get('project', {})
        pred_proj = self.predictor_model_info.get('project', {})

        det_info = f"{det_proj.get('name', 'Unknown')} v{det_proj.get('deploy_version', '?')}"
        pred_info = f"{pred_proj.get('name', 'Unknown')} v{pred_proj.get('deploy_version', '?')}"

        return (det_info, pred_info)

    def _open_capture(self) -> bool:
        """
        Open (or reopen) the video capture for the configured source.

        Supports wired camera device indices (int) and RTSP/HTTP stream URLs (str).

        Returns:
            True if the capture opened successfully, False otherwise
        """
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        source = self.source
        self.is_stream_source = isinstance(source, str) and source.startswith(
            ('rtsp://', 'http://', 'https://')
        )

        if self.is_stream_source:
            _log(f'Opening stream {source}...')
            # FFmpeg backend handles RTSP/HTTP network streams
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            # Minimize buffering for lower latency on network streams
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            camera_device = int(source)
            _log(f'Opening camera device {camera_device}...')
            # Use V4L2 backend on Linux for better camera control
            if os.path.exists('/dev/video' + str(camera_device)):
                self.cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
            else:
                self.cap = cv2.VideoCapture(camera_device)

            # Request 720p resolution (camera will use closest supported)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Set MJPEG format if available (faster than raw)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        return self.cap is not None and self.cap.isOpened()

    def start(self, source=None) -> dict:
        """
        Start the inference with the specified video source.

        Args:
            source: Camera device index (int) or RTSP/HTTP stream URL (str).
                    If None, falls back to camera device 0.

        Returns:
            Dictionary with 'success' boolean and optional 'error' message
        """
        if self._running:
            return {'success': False, 'error': 'Already running'}

        if not self._models_loaded:
            return {'success': False, 'error': 'Models not loaded'}

        if source is None:
            source = 0
        self.source = source

        if not self._open_capture():
            return {'success': False, 'error': f'Cannot open video source: {source}'}

        # Test read a frame (network streams can take a moment to connect)
        ret = False
        for _ in range(50):  # up to ~5s
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)
        if not ret:
            self.cap.release()
            self.cap = None
            return {'success': False, 'error': f'Video source opened but cannot read frames: {source}'}

        # Reset state
        with self._lock:
            self._current_frame = None
            self._current_results = []
            self._frame_count = 0
            self._detector_ms = 0
            self._predictor_ms = 0
            self._fps = 0.0
            self._last_fps_time = time.time()
            self._fps_frame_count = 0

        # Start inference thread
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
        _log('Inference thread started')

        det_info, pred_info = self.get_model_info()
        return {
            'success': True,
            'detector_info': det_info,
            'predictor_info': pred_info
        }

    def stop(self):
        """Stop the inference thread and release camera."""
        self._running = False

        if self._inference_thread:
            self._inference_thread.join(timeout=5.0)
            self._inference_thread = None

        # Release camera
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        _log('Inference stopped')

    def shutdown(self):
        """Full shutdown - stop inference and release models."""
        self.stop()

        # Stop model runners
        if self.detector_runner:
            try:
                self.detector_runner.stop()
            except Exception:
                pass

        if self.predictor_runner:
            try:
                self.predictor_runner.stop()
            except Exception:
                pass

        _log('OCR engine shut down')

    def is_running(self) -> bool:
        """Check if inference is currently running."""
        return self._running

    def _inference_loop(self):
        """
        Main inference loop running in a background thread.

        Continuously captures frames, runs detection and recognition,
        and updates the shared state for the web interface.
        """
        consecutive_read_failures = 0
        while self._running:
            try:
                # Capture frame from camera / stream
                ok, frame_bgr = self.cap.read()
                if not ok:
                    consecutive_read_failures += 1
                    # Network streams can drop - attempt to reconnect after repeated failures
                    if self.is_stream_source and consecutive_read_failures >= 30:
                        _log('Stream read failed repeatedly - attempting to reconnect...')
                        consecutive_read_failures = 0
                        if not self._open_capture():
                            _log('Reconnect failed, will keep retrying...')
                    time.sleep(0.01)
                    continue

                consecutive_read_failures = 0

                start_ms = int(time.time() * 1000)

                # Convert BGR to RGB for Edge Impulse
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # =========================================================
                # Stage 1: Text Detection
                # =========================================================
                features, _ = self.detector_runner.get_features_from_image_auto_studio_settings(frame_rgb)
                det_resp = self.detector_runner.classify(features)
                detector_time_ms = int(time.time() * 1000) - start_ms

                # Get image dimensions for coordinate mapping
                original_h, original_w = frame_rgb.shape[:2]
                detector_w = self.detector_model_info['model_parameters']['image_input_width']
                detector_h = self.detector_model_info['model_parameters']['image_input_height']
                original_info = {
                    'originalWidth': int(original_w),
                    'originalHeight': int(original_h),
                    'newWidth': int(detector_w),
                    'newHeight': int(detector_h),
                }

                # =========================================================
                # Stage 2: Text Recognition (per detected region)
                # =========================================================
                predictor_start_ms = int(time.time() * 1000)
                results: list[dict] = []
                overlay_items: list[dict] = []

                bbs = det_resp.get('result', {}).get('bounding_boxes') or []
                if isinstance(bbs, list):
                    for orig_bb in bbs:
                        if not isinstance(orig_bb, dict):
                            continue

                        # Map bounding box to original image coordinates
                        bb = map_bounding_box_back_to_original_image(
                            self.detector_model_info, original_info, orig_bb
                        )

                        # Filter out invalid or too-small boxes
                        if bb['width'] <= 0 or bb['height'] <= 0:
                            continue
                        if bb['width'] * bb['height'] < self.min_box_area:
                            continue

                        # Crop the detected text region
                        x, y, w, h = bb['x'], bb['y'], bb['width'], bb['height']
                        crop_rgb = frame_rgb[y:y + h, x:x + w]
                        if crop_rgb.size == 0:
                            continue

                        # Run text recognition on the cropped region
                        pred_features, _ = self.predictor_runner.get_features_from_image_auto_studio_settings(crop_rgb)
                        pred_resp = self.predictor_runner.classify(pred_features)
                        predicted = parse_paddle_ocr_predictor(pred_resp, self.predictor_dict)

                        if not predicted:
                            continue

                        text = str(predicted['output'])
                        conf = float(predicted['avgConfidence'])

                        results.append({
                            'text': text,
                            'confidence': conf,
                            'x': int(x),
                            'y': int(y),
                            'width': int(w),
                            'height': int(h),
                        })

                        overlay_items.append({
                            'x': int(x),
                            'y': int(y),
                            'w': int(w),
                            'h': int(h),
                            'text': text,
                            'conf': conf,
                        })

                predictor_time_ms = int(time.time() * 1000) - predictor_start_ms

                # =========================================================
                # Draw overlays on the frame
                # =========================================================
                display_frame = frame_bgr.copy()

                for item in overlay_items:
                    x1 = max(0, item['x'])
                    y1 = max(0, item['y'])
                    x2 = max(0, item['x'] + item['w'])
                    y2 = max(0, item['y'] + item['h'])

                    # Draw green bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

                    # Draw label with background
                    label = f"{item['text']} ({item['conf']:.2f})"
                    (font_w, font_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    bg_x2 = min(display_frame.shape[1] - 1, x1 + font_w + 6)
                    bg_y1 = max(0, y1 - font_h - baseline - 6)
                    cv2.rectangle(display_frame, (x1, bg_y1), (bg_x2, y1), BOX_COLOR, -1)
                    cv2.putText(
                        display_frame,
                        label,
                        (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Draw status overlay in top-left corner
                status = f"Frame {self._frame_count + 1} | Det: {detector_time_ms}ms | Pred: {predictor_time_ms}ms"
                cv2.putText(
                    display_frame,
                    status,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Encode frame as JPEG for web streaming
                _, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # =========================================================
                # Update shared state (thread-safe)
                # =========================================================
                with self._lock:
                    self._current_frame = jpeg.tobytes()
                    self._current_results = results
                    self._frame_count += 1
                    self._detector_ms = detector_time_ms
                    self._predictor_ms = predictor_time_ms

                    # Calculate FPS
                    self._fps_frame_count += 1
                    now = time.time()
                    elapsed = now - self._last_fps_time
                    if elapsed >= 1.0:
                        self._fps = self._fps_frame_count / elapsed
                        self._fps_frame_count = 0
                        self._last_fps_time = now

            except Exception as ex:
                _log(f'Inference error: {ex}')
                time.sleep(0.1)

    def get_frame(self) -> Optional[bytes]:
        """Get the current JPEG-encoded frame (thread-safe)."""
        with self._lock:
            return self._current_frame

    def get_status(self) -> dict:
        """Get current inference status (thread-safe)."""
        with self._lock:
            det_info, pred_info = self.get_model_info()
            return {
                'running': self._running,
                'frame_count': self._frame_count,
                'detector_ms': self._detector_ms,
                'predictor_ms': self._predictor_ms,
                'fps': self._fps,
                'results': self._current_results.copy(),
                'detector_info': det_info,
                'predictor_info': pred_info,
            }


# =============================================================================
# FLASK WEB APPLICATION
# =============================================================================

# Global OCR engine instance (initialized in main)
ocr_engine: Optional[OCREngine] = None

# Create Flask app
app = Flask(__name__)


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/cameras')
def list_cameras():
    """
    List available camera devices.

    Returns JSON with list of cameras, each having 'index' and 'name'.
    """
    cameras = detect_cameras()
    return jsonify({'cameras': cameras})


@app.route('/config')
def get_config():
    """
    Return client configuration for the web UI.

    Includes the default RTSP URL (if set) so the UI can prefill it.
    """
    return jsonify({'default_rtsp_url': DEFAULT_RTSP_URL or ''})


@app.route('/start', methods=['POST'])
def start_inference():
    """
    Start the inference engine with the specified video source.

    Expects JSON body with either:
      - 'source': RTSP/HTTP stream URL (string), or
      - 'camera': wired camera device index (integer)
    Returns JSON with 'success' boolean and optional 'error' message.
    """
    data = request.get_json() or {}
    # 'source' may be an RTSP/HTTP URL (str); 'camera' is a wired camera index (int)
    source = data.get('source')
    if source in (None, ''):
        source = data.get('camera', 0)

    result = ocr_engine.start(source=source)
    return jsonify(result)


@app.route('/stop', methods=['POST'])
def stop_inference():
    """
    Stop the inference engine.

    Returns JSON with 'success' boolean.
    """
    ocr_engine.stop()
    return jsonify({'success': True})


@app.route('/video_feed')
def video_feed():
    """
    MJPEG video stream endpoint.

    This endpoint streams JPEG frames continuously using multipart/x-mixed-replace
    content type, which browsers render as a live video stream.
    """
    def generate():
        while ocr_engine.is_running():
            frame = ocr_engine.get_frame()
            if frame:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                )
            else:
                time.sleep(0.01)

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    """
    JSON endpoint for current inference status.

    Returns frame count, timing info, FPS, running state, and detected text results.
    """
    return jsonify(ocr_engine.get_status())


# =============================================================================
# MODEL AUTO-DETECTION FOR ARDUINO APP LAB
# =============================================================================

import platform


def _detect_platform() -> str:
    """
    Detect the current platform for model selection.

    Returns:
        'linux-aarch64', 'mac-arm64', or 'unknown'
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == 'linux' and machine == 'aarch64':
        return 'linux-aarch64'
    elif system == 'darwin' and machine == 'arm64':
        return 'mac-arm64'
    elif system == 'linux' and machine == 'x86_64':
        return 'linux-x86_64'
    return 'unknown'


def _find_model_file(base_dir: str, pattern_keywords: list[str]) -> str | None:
    """
    Search for a model file matching keywords in the given directory tree.

    Args:
        base_dir: Root directory to search
        pattern_keywords: Keywords to look for in filenames (e.g., ['detect', 'detector'])

    Returns:
        Path to the first matching .eim file, or None
    """
    import glob as glob_mod
    for eim_file in sorted(glob_mod.glob(os.path.join(base_dir, '**', '*.eim'), recursive=True)):
        fname = os.path.basename(eim_file).lower()
        if any(kw in fname for kw in pattern_keywords):
            return eim_file
    return None


def _find_dict_file(base_dir: str) -> str | None:
    """
    Search for a character dictionary file in the project.

    Args:
        base_dir: Root directory to search

    Returns:
        Path to the dictionary file, or None
    """
    # Check common locations
    candidates = [
        os.path.join(base_dir, 'source_models', 'rec_en_dict.txt'),
        os.path.join(base_dir, 'models', 'rec_en_dict.txt'),
        os.path.join(base_dir, 'assets', 'rec_en_dict.txt'),
        os.path.join(base_dir, 'rec_en_dict.txt'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fallback: search recursively
    import glob as glob_mod
    results = glob_mod.glob(os.path.join(base_dir, '**', 'rec_en_dict.txt'), recursive=True)
    return results[0] if results else None


def auto_detect_models(base_dir: str) -> dict:
    """
    Auto-detect model files based on the current platform.

    Searches for detector and recognizer .eim files in platform-specific
    model directories. This enables zero-config startup on Arduino UNO Q.

    Args:
        base_dir: Project root directory

    Returns:
        Dictionary with 'detect_file', 'predict_file', 'dict_file' paths
        (values are None if not found)
    """
    plat = _detect_platform()
    _log(f'Detected platform: {plat}')

    # Define search order for model directories
    model_dirs = []
    if plat == 'linux-aarch64':
        model_dirs = [
            os.path.join(base_dir, 'models', 'arduino-uno-q'),
            os.path.join(base_dir, 'models', 'linux-aarch64'),
            os.path.join(base_dir, 'models'),
        ]
    elif plat == 'mac-arm64':
        model_dirs = [
            os.path.join(base_dir, 'models', 'mac-arm64'),
            os.path.join(base_dir, 'models'),
        ]
    else:
        model_dirs = [os.path.join(base_dir, 'models')]

    detect_file = None
    predict_file = None

    # Search each directory in priority order
    for model_dir in model_dirs:
        if not os.path.isdir(model_dir):
            continue

        if detect_file is None:
            detect_file = _find_model_file(model_dir, ['detect', 'detector'])
        if predict_file is None:
            predict_file = _find_model_file(model_dir, ['recogni', 'predict', 'paddleocr'])

        if detect_file and predict_file:
            break

    dict_file = _find_dict_file(base_dir)

    if detect_file:
        _log(f'  Detector model: {detect_file}')
    else:
        _log('  Detector model: NOT FOUND')

    if predict_file:
        _log(f'  Predictor model: {predict_file}')
    else:
        _log('  Predictor model: NOT FOUND')

    if dict_file:
        _log(f'  Dictionary: {dict_file}')
    else:
        _log('  Dictionary: NOT FOUND')

    return {
        'detect_file': detect_file,
        'predict_file': predict_file,
        'dict_file': dict_file,
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the web inference server."""
    global ocr_engine, DEFAULT_RTSP_URL

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Edge Impulse OCR Web Interface (Arduino App Lab compatible)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect models (Arduino UNO Q / App Lab):
  python3 python/main.py

  # Or run directly with auto-detection:
  python3 web_inference.py

  # Explicit model paths (Arduino UNO Q):
  python3 web_inference.py \\
    --detect-file ./models/arduino-uno-q/detector-linux-aarch64.eim \\
    --predict-file ./models/arduino-uno-q/recognizer-linux-aarch64.eim \\
    --dict-file source_models/rec_en_dict.txt

  # Explicit model paths (macOS ARM64):
  python3.10 web_inference.py \\
    --detect-file ./models/mac-arm64/detect-v3-640-480-f32.eim \\
    --predict-file ./models/mac-arm64/recognizer-320-48-f32.eim \\
    --dict-file source_models/rec_en_dict.txt

  # Use a specific RTSP stream as the video source:
  python3 web_inference.py --rtsp-url rtsp://10.0.0.124:554/mjpeg/1

  # Via Arduino App Lab CLI:
  arduino-app-cli app start .

Access the web UI at http://localhost:5001 or http://<device-ip>:5001
Inference auto-starts on the default RTSP stream; you can switch to a
wired camera (dropdown) or another RTSP URL from the UI.
        """
    )

    # Model file arguments (now optional - auto-detected if not provided)
    parser.add_argument(
        '--detect-file',
        default=None,
        help='Path to detector .eim model file (auto-detected if omitted)'
    )
    parser.add_argument(
        '--predict-file',
        default=None,
        help='Path to predictor/recognizer .eim model file (auto-detected if omitted)'
    )
    parser.add_argument(
        '--dict-file',
        default=None,
        help='Path to character dictionary file (auto-detected if omitted)'
    )

    # Video source
    parser.add_argument(
        '--rtsp-url',
        default=DEFAULT_RTSP_URL,
        help=f'RTSP/HTTP stream URL to use as the video source and auto-start with '
             f'(default: {DEFAULT_RTSP_URL}). Pass an empty string to disable; '
             f'wired cameras can still be selected in the web UI.'
    )

    # Inference settings
    parser.add_argument(
        '--min-box-area',
        default='20',
        help='Minimum bounding box area to process (default: 20)'
    )
    parser.add_argument(
        '--init-timeout-s',
        default='30',
        help='Timeout in seconds for model initialization (default: 30)'
    )

    # Web server settings
    parser.add_argument(
        '--host',
        default=DEFAULT_HOST,
        help=f'Host to bind the web server to (default: {DEFAULT_HOST})'
    )
    parser.add_argument(
        '--port',
        default=str(DEFAULT_PORT),
        type=int,
        help=f'Port for the web server (default: {DEFAULT_PORT})'
    )

    # Debug options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable Flask debug mode and verbose EIM output'
    )

    args = parser.parse_args()

    # Allow the CLI to override the default RTSP URL (also served by /config)
    if args.rtsp_url is not None:
        DEFAULT_RTSP_URL = args.rtsp_url

    # Auto-detect models if not explicitly provided
    # Resolve base_dir relative to where the script is run from
    base_dir = os.getcwd()
    # If running from python/ subdirectory, go up one level
    if os.path.basename(base_dir) == 'python':
        base_dir = os.path.dirname(base_dir)

    detect_file = args.detect_file
    predict_file = args.predict_file
    dict_file = args.dict_file

    if not detect_file or not predict_file or not dict_file:
        _log('Auto-detecting models...')
        detected = auto_detect_models(base_dir)

        if not detect_file:
            detect_file = detected['detect_file']
        if not predict_file:
            predict_file = detected['predict_file']
        if not dict_file:
            dict_file = detected['dict_file']

    # Validate required files
    if not detect_file:
        _log('ERROR: No detector model found. Provide --detect-file or place .eim files in models/')
        sys.exit(1)
    if not predict_file:
        _log('ERROR: No predictor model found. Provide --predict-file or place .eim files in models/')
        sys.exit(1)
    if not dict_file:
        _log('ERROR: No dictionary file found. Provide --dict-file or place rec_en_dict.txt in source_models/')
        sys.exit(1)

    # Parse numeric arguments
    min_box_area = int(args.min_box_area) if str(args.min_box_area).isdigit() else 20
    init_timeout_s = int(args.init_timeout_s) if str(args.init_timeout_s).isdigit() else 30

    # Create the OCR engine (models loaded, camera selected via UI)
    ocr_engine = OCREngine(
        detect_file=detect_file,
        predict_file=predict_file,
        dict_file=dict_file,
        min_box_area=min_box_area,
        init_timeout_s=init_timeout_s,
        debug=args.debug,
    )

    try:
        # Pre-load models at startup
        ocr_engine.load_models()

        # Auto-start inference on the configured RTSP stream (if any)
        if DEFAULT_RTSP_URL:
            _log(f'Auto-starting inference with stream: {DEFAULT_RTSP_URL}')
            start_result = ocr_engine.start(source=DEFAULT_RTSP_URL)
            if not start_result.get('success'):
                _log(f'WARNING: auto-start failed ({start_result.get("error")}). '
                     f'You can still start manually from the web UI.')

        _log('')
        _log('=' * 60)
        _log('Edge Impulse OCR Web Interface')
        _log('=' * 60)
        _log(f'Web UI available at:')
        _log(f'  - http://localhost:{args.port}')
        _log(f'  - http://<device-ip>:{args.port}')
        _log('')
        _log('Inference auto-starts on the default RTSP stream.')
        _log('You can switch to a wired camera or another RTSP URL from the UI.')
        _log('Press Ctrl+C to stop the server')
        _log('=' * 60)
        _log('')

        # Start Flask server
        # threaded=True allows handling multiple requests simultaneously
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reloader=False,  # Disable reloader to prevent double initialization
        )

    except KeyboardInterrupt:
        _log('\nShutting down...')
    finally:
        if ocr_engine:
            ocr_engine.shutdown()


if __name__ == '__main__':
    main()
