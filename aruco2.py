# -----------------------------------------------------------------------------------------
# PHASE 3: STANDALONE MODE & UI REFINEMENT
# -----------------------------------------------------------------------------------------
# ARCHITECT: Harsh Kamlesh Chhajed
# DATE: 2026-01-03
# -----------------------------------------------------------------------------------------
# CHANGE LOG:
# 1. WEBSOCKET REMOVAL: The system is now fully offline/local.
#    - Removed AsyncWebSocketServer, asyncio, websockets.
#    - Removed "Ghost Robot" feedback loop.
# 2. UI CLARITY: "End" point is now conditional.
#    - It is HIDDEN until the user leaves Start AND hits Checkpoint 1.
#    - Prevents visual overlap on closed-loop shapes (Square, Circle).
# -----------------------------------------------------------------------------------------
# GPU ACCELERATION (RTX 4060):
# - CUDA Pre-processing remains active.
# -----------------------------------------------------------------------------------------

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import signal
import sys
import serial
import threading
import serial.tools.list_ports
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
import math
from scipy.spatial.transform import Rotation as R
import logging
import traceback
import random

# ----------------------------
# Configuration & Constants
# ----------------------------
ENABLE_CLAHE = False
ENABLE_SHARPENING = False


@dataclass
class AppConfig:
    # Robot Constants
    HOME_POS: np.ndarray = np.array([0.05, 0.0, 0.20])

    # GOLDEN FEATURE: The Robot's "Front Facing" Orientation
    HOME_QUAT: np.ndarray = np.array([0.0, 0.675, 0.0, 0.738])

    # Smart Transformation Mapping
    USER_FRAME_MAPPING: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

    # ---------------------------------------------------------
    # FILTER CONFIGURATION
    # ---------------------------------------------------------
    # OneEuro Position (mm scale)
    POS_FILTER_MIN_CUTOFF: float = 0.05
    POS_FILTER_BETA: float = 0.1

    # OneEuro Rotation (Quaternion scale)
    ROT_FILTER_MIN_CUTOFF: float = 0.02
    ROT_FILTER_BETA: float = 0.02

    FILTER_DERIVATIVE_CUTOFF: float = 1.0

    # EMA Configuration
    EMA_ALPHA: float = 0.35

    # App Settings
    SERIAL_BAUD: int = 115200

    # ArUco Configuration
    # Users will attach this marker ID to the back of the phone.
    TRACK_MARKER_ID: int = 0
    MARKER_SIZE_M: float = 0.05  # 5cm physical size (for reference)

    # Camera Config (Native HD)
    DEPTH_WIDTH: int = 1024
    DEPTH_HEIGHT: int = 768
    COLOR_WIDTH: int = 1280
    COLOR_HEIGHT: int = 720
    FPS: int = 30

    # Logic
    ORIGIN_SET_DURATION: float = 3.0
    MAX_DEPTH_DROPOUT_FRAMES: int = 10

    # UI Constants
    SQUARE_SIZE: int = 75
    PANEL_HEIGHT: int = 400
    BG_COLOR: Tuple[int, int, int] = (45, 45, 45)
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
    HIGHLIGHT_COLOR: Tuple[int, int, int] = (0, 255, 255)
    WARNING_COLOR: Tuple[int, int, int] = (0, 165, 255)
    OK_COLOR: Tuple[int, int, int] = (0, 210, 0)
    MARKER_BORDER_COLOR: Tuple[int, int, int] = (0, 255, 0)

    # Logic Thresholds
    LAG_THRESHOLD: float = 0.05

    # User Study Config
    SHAPE_COLOR: Tuple[int, int, int] = (0, 255, 0)
    SHAPE_THICKNESS: int = 9
    SHAPE_ALPHA: float = 0.5
    SHAPE_SCALE: float = 0.3
    SHAPE_OFFSET_Y: int = 110
    SAFETY_MIN_X: float = 0.180
    START_ZONE_RADIUS: int = 30
    CHECKPOINT_RADIUS: int = 30
    TRACING_DURATION: float = 60.0
    COMPLETION_DELAY: float = 2.0


config = AppConfig()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ----------------------------
# Type Definitions
# ----------------------------
@dataclass
class RobotPose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


# ----------------------------
# Math & Physics Engine
# ----------------------------
class OneEuroFilter:
    """
    Optimized filtering algorithm for HRI.
    """

    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = np.array(x0, dtype=float)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class EMAFilter:
    def __init__(self, x0, alpha=0.5):
        self.alpha = alpha
        self.x_prev = np.array(x0, dtype=float)

    def __call__(self, x):
        x_hat = self.alpha * x + (1 - self.alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class MathUtils:
    @staticmethod
    def normalize_quat(q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / norm

    @staticmethod
    def get_relative_pose(
        hand_pos_cam: np.ndarray,
        origin_pos: np.ndarray,
        current_imu_quat: np.ndarray,
        origin_imu_quat: np.ndarray,
    ) -> RobotPose:
        # [-z, x, -y] conversion from Camera to Robot frame
        robot_frame_pos = np.array(
            [-hand_pos_cam[2], hand_pos_cam[0], -hand_pos_cam[1]]
        )
        rel_pos = robot_frame_pos - origin_pos
        final_pos = config.HOME_POS + rel_pos

        r_curr = R.from_quat(current_imu_quat)
        r_origin = R.from_quat(origin_imu_quat)
        r_home = R.from_quat(config.HOME_QUAT)
        r_mapping = R.from_quat(config.USER_FRAME_MAPPING)

        r_change_local = r_origin.inv() * r_curr
        r_final = r_home * r_mapping * r_change_local
        final_q = r_final.as_quat()

        return RobotPose(
            x=float(final_pos[0]),
            y=float(final_pos[1]),
            z=float(final_pos[2]),
            qx=float(final_q[0]),
            qy=float(final_q[1]),
            qz=float(final_q[2]),
            qw=float(final_q[3]),
        )


# ----------------------------
# GPU / Image Processing
# ----------------------------
class ImageProcessor:
    def __init__(self):
        self.use_cuda = False
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                self.use_cuda = True
                logging.info(f"CUDA enabled. Found {count} device(s).")
                self.gpu_mat = cv2.cuda_GpuMat()

                # GPU Objects
                self.cuda_clahe = cv2.cuda.createCLAHE(
                    clipLimit=2.0, tileGridSize=(8, 8)
                )

                # Sharpen Kernel for CUDA
                kernel = np.array(
                    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
                )
                self.cuda_sharpen_filter = cv2.cuda.createLinearFilter(
                    cv2.CV_8UC3, cv2.CV_8UC3, kernel
                )
            else:
                logging.info("CUDA not found. Using CPU for image processing.")
        except AttributeError:
            logging.info("cv2.cuda module not found. Using CPU.")
        except Exception as e:
            logging.warning(f"Error initializing CUDA: {e}. Using CPU.")

        # CPU Fallback objects
        self.cpu_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.cpu_kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def preprocess(self, color_img: np.ndarray) -> np.ndarray:
        """Handles CLAHE and Sharpening, preferentially on GPU."""
        if not (ENABLE_CLAHE or ENABLE_SHARPENING):
            return color_img

        if self.use_cuda:
            try:
                self.gpu_mat.upload(color_img)

                if ENABLE_CLAHE:
                    gpu_lab = cv2.cuda.cvtColor(self.gpu_mat, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.cuda.split(gpu_lab)
                    l = self.cuda_clahe.apply(l, cv2.cuda_Stream_Null())
                    gpu_lab = cv2.cuda.merge((l, a, b))
                    self.gpu_mat = cv2.cuda.cvtColor(gpu_lab, cv2.COLOR_LAB2BGR)

                if ENABLE_SHARPENING:
                    self.gpu_mat = self.cuda_sharpen_filter.apply(self.gpu_mat)

                return self.gpu_mat.download()
            except Exception as e:
                logging.warning(f"CUDA Preprocess failed: {e}. Falling back to CPU.")

        # CPU Fallback
        res_img = color_img.copy()
        if ENABLE_CLAHE:
            try:
                lab = cv2.cvtColor(res_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = self.cpu_clahe.apply(l)
                lab = cv2.merge((l, a, b))
                res_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception:
                pass

        if ENABLE_SHARPENING:
            try:
                res_img = cv2.filter2D(res_img, -1, self.cpu_kernel_sharpen)
            except Exception:
                pass
        return res_img

    def flip(self, img: np.ndarray) -> np.ndarray:
        if self.use_cuda:
            try:
                self.gpu_mat.upload(img)
                gpu_flipped = cv2.cuda.flip(self.gpu_mat, 1)
                return gpu_flipped.download()
            except Exception:
                return cv2.flip(img, 1)
        else:
            return cv2.flip(img, 1)


# ----------------------------
# Hardware Interfaces
# ----------------------------
class IMUReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.latest_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.lock = threading.Lock()
        self.ser: Optional[serial.Serial] = None

    def _find_port(self) -> Optional[str]:
        for port in serial.tools.list_ports.comports():
            if (port.vid == 0x10C4 and port.pid == 0xEA60) or (
                "CP210x" in port.description
                or "CH340" in port.description
                or "USB Serial" in port.description
            ):
                return port.device
        return None

    def run(self):
        port = self._find_port()
        if not port:
            logging.warning("No ESP32/IMU found on USB.")
            return

        try:
            self.ser = serial.Serial(port, config.SERIAL_BAUD, timeout=0.1)
            logging.info(f"Connected to IMU on {port}")
            self.ser.reset_input_buffer()
        except Exception as e:
            logging.error(f"Failed to open serial: {e}")
            return

        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    lines = (
                        self.ser.read_all().decode("utf-8", errors="ignore").split("\n")
                    )
                    if not lines:
                        time.sleep(0.005)
                        continue

                    for line in reversed(lines):
                        line = line.strip()
                        if not line or line.startswith("MODE"):
                            continue
                        parts = line.split(",")
                        if len(parts) >= 5:
                            try:
                                q = np.array([float(p) for p in parts[1:5]])
                                if np.any(np.abs(q) > 1.05):
                                    continue
                                with self.lock:
                                    self.latest_quat = q
                                break
                            except ValueError:
                                continue
                time.sleep(0.002)
            except Exception as e:
                logging.error(f"IMU Read Error: {e}")
                time.sleep(1.0)

    def get_quat(self) -> np.ndarray:
        with self.lock:
            return self.latest_quat.copy()

    def start_stream(self):
        if self.ser:
            try:
                self.ser.write(b"start\n")
            except Exception as e:
                logging.error(f"Serial write failed: {e}")

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()


# ----------------------------
# Vision Pipeline (ARUCO OPTIMIZED)
# ----------------------------
class VisionSystem:
    def __init__(self, img_processor):
        self.processor = img_processor

        # 1. Initialize ArUco Detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # 2. Initialize RealSense
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.depth,
            config.DEPTH_WIDTH,
            config.DEPTH_HEIGHT,
            rs.format.z16,
            config.FPS,
        )
        self.rs_config.enable_stream(
            rs.stream.color,
            config.COLOR_WIDTH,
            config.COLOR_HEIGHT,
            rs.format.bgr8,
            config.FPS,
        )

        self.profile = None
        self.intrinsics = None
        self.depth_scale = 0.001
        self.align = rs.align(rs.stream.color)

        self.frame_lock = threading.Lock()
        self.latest_color_img = None
        self.latest_depth_img = None
        self.latest_ts = 0.0
        self.running = False
        self.capture_thread = None

    def start(self):
        self.profile = self.pipeline.start(self.rs_config)
        device = self.profile.get_device()
        self.depth_scale = device.first_depth_sensor().get_depth_scale()

        self.intrinsics = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                capture_time = time.time()
                aligned_frames = self.align.process(frames)

                c_frame = aligned_frames.get_color_frame()
                d_frame = aligned_frames.get_depth_frame()

                if not c_frame or not d_frame:
                    continue

                color_arr = np.asanyarray(c_frame.get_data())
                depth_arr = np.asanyarray(d_frame.get_data())

                # GPU Accelerated Preprocessing
                processed_color = self.processor.preprocess(color_arr)
                processed_depth = depth_arr

                with self.frame_lock:
                    self.latest_color_img = processed_color
                    self.latest_depth_img = processed_depth
                    self.latest_ts = capture_time

            except Exception as e:
                logging.error(f"Frame capture error: {e}")
                time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        self.pipeline.stop()

    def get_frames(self):
        with self.frame_lock:
            if self.latest_color_img is not None and self.latest_depth_img is not None:
                return (
                    self.latest_color_img.copy(),
                    self.latest_depth_img.copy(),
                    self.latest_ts,
                )
        return None, None, 0.0

    def get_depth_vectorized(
        self, depth_image, x: int, y: int, radius: int = 3
    ) -> float:
        h, w = depth_image.shape[:2]
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(w, x + radius + 1), min(h, y + radius + 1)
        roi = depth_image[y1:y2, x1:x2]
        valid_depths = roi[roi > 0]
        if valid_depths.size == 0:
            return 0.0
        return float(np.median(valid_depths)) * self.depth_scale

    def detect_aruco(
        self, image: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], List[Any]]:
        """
        Detects markers and returns the centroid of the target ID and the raw corners for visualization.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        target_center = None
        if ids is not None:
            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                if marker_id == config.TRACK_MARKER_ID:
                    c = corners[i][0]
                    cx = float(np.mean(c[:, 0]))
                    cy = float(np.mean(c[:, 1]))
                    target_center = (cx, cy)
                    break

        return target_center, corners, ids


# ----------------------------
# Helper Functions (Rendering)
# ----------------------------
def get_shape_targets(
    shape_id: int, w: int, h: int
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    cx = w // 2
    cy = (h // 2) + config.SHAPE_OFFSET_Y
    base_size = int(h * config.SHAPE_SCALE)
    start_pt, end_pt, refs = None, None, []

    if shape_id == 1:  # Square
        half = base_size // 2
        start_pt = (cx - half, cy - half)
        end_pt = start_pt
        refs = [(cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
    elif shape_id == 2:  # Circle
        radius = base_size // 2
        start_pt = (cx, cy - radius)
        end_pt = start_pt
        refs = [(cx + radius, cy), (cx, cy + radius), (cx - radius, cy)]
    elif shape_id == 3:  # Letter 'S'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = (base_size / 22.0) * 0.8
        thickness_text = config.SHAPE_THICKNESS + 5
        text = "S"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness_text)
        start_pt = (int(cx + tw * 0.25), int(cy - th * 0.35))
        end_pt = (int(cx - tw * 0.3), int(cy + th * 0.35))
        refs = [(cx, cy)]
    elif shape_id == 4:  # Triangle
        radius = base_size // 2
        start_pt = (cx, cy - radius)
        end_pt = start_pt
        refs = [(cx + radius, cy + radius), (cx - radius, cy + radius)]
    elif shape_id == 5:  # Rectangle (Added)
        half_h = base_size // 2
        half_w = int(base_size * 0.75)  # 50% wider than square
        start_pt = (cx - half_w, cy - half_h)
        end_pt = start_pt
        refs = [
            (cx + half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
        ]

    return start_pt, end_pt, refs


def draw_reference_shape(image: np.ndarray, shape_id: int) -> np.ndarray:
    if shape_id == 0:
        return image
    overlay = image.copy()
    h, w = image.shape[:2]
    cx = w // 2
    cy = (h // 2) + config.SHAPE_OFFSET_Y
    base_size = int(h * config.SHAPE_SCALE)
    thickness = config.SHAPE_THICKNESS

    if shape_id == 1:
        half = base_size // 2
        cv2.rectangle(
            overlay,
            (cx - half, cy - half),
            (cx + half, cy + half),
            config.SHAPE_COLOR,
            thickness,
        )
    elif shape_id == 2:
        radius = base_size // 2
        cv2.circle(overlay, (cx, cy), radius, config.SHAPE_COLOR, thickness)
    elif shape_id == 3:
        text = "S"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = (base_size / 22.0) * 0.8
        thickness_text = thickness + 5
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness_text)
        cv2.putText(
            overlay,
            text,
            (cx - tw // 2, cy + th // 2),
            font,
            font_scale,
            config.SHAPE_COLOR,
            thickness_text,
        )
    elif shape_id == 4:
        radius = base_size // 2
        p1, p2, p3 = (
            (cx, cy - radius),
            (cx + radius, cy + radius),
            (cx - radius, cy + radius),
        )
        cv2.line(overlay, p1, p2, config.SHAPE_COLOR, thickness)
        cv2.line(overlay, p2, p3, config.SHAPE_COLOR, thickness)
        cv2.line(overlay, p3, p1, config.SHAPE_COLOR, thickness)
    elif shape_id == 5:
        half_h = base_size // 2
        half_w = int(base_size * 0.75)
        cv2.rectangle(
            overlay,
            (cx - half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            config.SHAPE_COLOR,
            thickness,
        )

    return cv2.addWeighted(
        overlay, config.SHAPE_ALPHA, image, 1 - config.SHAPE_ALPHA, 0
    )


# ----------------------------
# Main Application
# ----------------------------
def main():
    img_processor = ImageProcessor()
    imu = IMUReader()
    imu.start()

    # WEBSOCKET REMOVED HERE

    vision = VisionSystem(img_processor)
    try:
        vision.start()
        logging.info("ArUco Vision System initialized.")
    except RuntimeError as e:
        logging.critical(f"Camera failed: {e}")
        return

    origin_pos_robot_frame = None
    origin_quat = np.array([0.0, 0.0, 0.0, 1.0])
    origin_set_start = None
    origin_screen_coords = None
    manual_imu_calibration_done = False

    # Filters
    pos_filter = None
    rot_filter = None

    is_first_detection = True
    orientation_test_mode = False
    active_filter_mode = "EURO"
    filter_needs_reset = False

    last_valid_target_cam = None
    depth_dropout_counter = 0

    # State Machine
    STATE_IDLE = 0
    STATE_WAITING_START = 1
    STATE_TRACING = 2
    STATE_COMPLETED = 3

    study_state = STATE_IDLE
    current_shape_id = 0
    tracing_start_time = None
    completion_wait_start = None
    has_left_start_zone = False
    last_frozen_pose = None
    visited_checkpoints = set()

    title = f"CV Feed: ArUco Tracking (ID {config.TRACK_MARKER_ID})"
    logging.info("System initialized. Starting main loop...")

    try:
        while True:
            try:
                loop_start = time.time()

                # ---------------------------------------------------------
                # DATA ACQUISITION
                # ---------------------------------------------------------
                color_img, depth_img, frame_ts = vision.get_frames()
                if color_img is None or depth_img is None:
                    time.sleep(0.001)
                    continue

                # Sync IMU read
                current_quat_raw = imu.get_quat()
                if frame_ts == 0.0:
                    frame_ts = loop_start

                # ---------------------------------------------------------
                # PROCESSING BLOCK (ARUCO)
                # ---------------------------------------------------------
                h, w, _ = color_img.shape

                # Detect markers
                marker_centroid, corners, ids = vision.detect_aruco(color_img)

                current_target_cam = None
                raw_px, raw_py = 0, 0
                tracking_status = "Scanning"

                if marker_centroid:
                    cx, cy = marker_centroid
                    raw_px, raw_py = int(cx), int(cy)

                    # Sample Depth at Centroid
                    dist = vision.get_depth_vectorized(depth_img, raw_px, raw_py)

                    if dist > 0:
                        current_target_cam = np.array(
                            rs.rs2_deproject_pixel_to_point(
                                vision.intrinsics, [raw_px, raw_py], dist
                            )
                        )
                        last_valid_target_cam = current_target_cam
                        depth_dropout_counter = 0
                        tracking_status = f"Tracking ID {config.TRACK_MARKER_ID}"
                    elif (
                        last_valid_target_cam is not None
                        and depth_dropout_counter < config.MAX_DEPTH_DROPOUT_FRAMES
                    ):
                        # LKV Logic (Last Known Value)
                        current_target_cam = last_valid_target_cam
                        depth_dropout_counter += 1
                        tracking_status = "Holding (Depth Gap)"
                else:
                    # LKV Logic for lost visual tracking
                    if (
                        last_valid_target_cam is not None
                        and depth_dropout_counter < config.MAX_DEPTH_DROPOUT_FRAMES
                    ):
                        current_target_cam = last_valid_target_cam
                        depth_dropout_counter += 1
                        tracking_status = "Holding (Occluded)"
                    else:
                        tracking_status = "LOST"

                # Visual Coordinates (Mirror X for intuitive feedback)
                visual_px = w - raw_px if raw_px > 0 else 0
                visual_py = raw_py

                # ---------------------------------------------------------
                # FILTERING
                # ---------------------------------------------------------
                tracked_pos_cam = None

                if current_target_cam is not None:
                    if is_first_detection or pos_filter is None or filter_needs_reset:
                        if active_filter_mode == "EURO":
                            pos_filter = OneEuroFilter(
                                frame_ts,
                                current_target_cam,
                                min_cutoff=config.POS_FILTER_MIN_CUTOFF,
                                beta=config.POS_FILTER_BETA,
                                d_cutoff=config.FILTER_DERIVATIVE_CUTOFF,
                            )
                        else:
                            pos_filter = EMAFilter(
                                current_target_cam, alpha=config.EMA_ALPHA
                            )
                        tracked_pos_cam = current_target_cam
                        is_first_detection = False
                    else:
                        if active_filter_mode == "EURO":
                            tracked_pos_cam = pos_filter(frame_ts, current_target_cam)
                        else:
                            tracked_pos_cam = pos_filter(current_target_cam)

                elif is_first_detection is False:
                    pos_filter = None  # Reset filter on total loss

                # Rotation Filtering (IMU)
                if rot_filter is None or filter_needs_reset:
                    if active_filter_mode == "EURO":
                        rot_filter = OneEuroFilter(
                            frame_ts,
                            current_quat_raw,
                            min_cutoff=config.ROT_FILTER_MIN_CUTOFF,
                            beta=config.ROT_FILTER_BETA,
                        )
                    else:
                        rot_filter = EMAFilter(current_quat_raw, alpha=config.EMA_ALPHA)
                    current_quat = current_quat_raw
                    filter_needs_reset = False
                else:
                    if active_filter_mode == "EURO":
                        current_quat = rot_filter(frame_ts, current_quat_raw)
                    else:
                        current_quat = rot_filter(current_quat_raw)
                    current_quat = MathUtils.normalize_quat(current_quat)

                # ---------------------------------------------------------
                # ORIGIN & POSE LOGIC
                # ---------------------------------------------------------
                sq_x1, sq_y1 = (
                    w - config.SQUARE_SIZE
                ) // 2, h - config.SQUARE_SIZE - 20
                in_box = (sq_x1 < visual_px < sq_x1 + config.SQUARE_SIZE) and (
                    sq_y1 < visual_py < sq_y1 + config.SQUARE_SIZE
                )

                if (
                    in_box
                    and origin_pos_robot_frame is None
                    and current_target_cam is not None
                ):
                    if origin_set_start is None:
                        origin_set_start = time.time()
                    elif time.time() - origin_set_start > config.ORIGIN_SET_DURATION:
                        origin_pos_robot_frame = np.array(
                            [
                                -tracked_pos_cam[2],
                                tracked_pos_cam[0],
                                -tracked_pos_cam[1],
                            ]
                        )
                        if not manual_imu_calibration_done:
                            origin_quat = current_quat
                        origin_screen_coords = (visual_px, visual_py)
                        imu.start_stream()
                        logging.info("Origin Set via ArUco!")
                else:
                    if origin_pos_robot_frame is None:
                        origin_set_start = None

                output_pose = None
                if origin_pos_robot_frame is not None and tracked_pos_cam is not None:
                    output_pose = MathUtils.get_relative_pose(
                        tracked_pos_cam,
                        origin_pos_robot_frame,
                        current_quat,
                        origin_quat,
                    )
                    if orientation_test_mode:
                        output_pose.x = float(config.HOME_POS[0])
                        output_pose.y = float(config.HOME_POS[1])
                        output_pose.z = float(config.HOME_POS[2])
                    if current_shape_id == 4:
                        output_pose.x = 0.250

                # Safety Check
                safety_violation = False
                if output_pose and current_shape_id != 0:
                    if current_shape_id != 4 and output_pose.x < config.SAFETY_MIN_X:
                        safety_violation = True

                # ---------------------------------------------------------
                # STUDY LOGIC
                # ---------------------------------------------------------
                start_pt, end_pt, refs = get_shape_targets(current_shape_id, w, h)

                if study_state == STATE_WAITING_START:
                    visited_checkpoints.clear()
                    completion_wait_start = None
                    if not safety_violation and start_pt and visual_px > 0:
                        dist_to_start = np.hypot(
                            visual_px - start_pt[0], visual_py - start_pt[1]
                        )
                        if dist_to_start < config.START_ZONE_RADIUS:
                            study_state = STATE_TRACING
                            tracing_start_time = time.time()
                            has_left_start_zone = False
                            logging.info("Study: Tracing Started")

                elif study_state == STATE_TRACING:
                    if tracing_start_time and visual_px > 0:
                        dist_to_start = np.hypot(
                            visual_px - start_pt[0], visual_py - start_pt[1]
                        )
                        if dist_to_start > config.START_ZONE_RADIUS * 1.5:
                            has_left_start_zone = True

                        for idx, ref_pt in enumerate(refs):
                            if idx not in visited_checkpoints:
                                dist_ref = np.hypot(
                                    visual_px - ref_pt[0], visual_py - ref_pt[1]
                                )
                                if dist_ref < config.CHECKPOINT_RADIUS:
                                    visited_checkpoints.add(idx)

                        is_complete = False
                        if has_left_start_zone:
                            if len(visited_checkpoints) == len(refs):
                                dist_to_end = np.hypot(
                                    visual_px - end_pt[0], visual_py - end_pt[1]
                                )
                                if dist_to_end < config.START_ZONE_RADIUS:
                                    is_complete = True

                        if is_complete:
                            if completion_wait_start is None:
                                completion_wait_start = time.time()
                            if (
                                time.time() - completion_wait_start
                                >= config.COMPLETION_DELAY
                            ):
                                study_state = STATE_COMPLETED
                                last_frozen_pose = output_pose
                                logging.info("Study: Task Completed")
                        else:
                            completion_wait_start = None

                elif study_state == STATE_COMPLETED:
                    if last_frozen_pose:
                        output_pose = last_frozen_pose

                # ----------------------------
                # UI Rendering
                # ----------------------------
                # 1. Background & Markers
                display_img = img_processor.flip(color_img)

                # Draw ArUco Markers (Before Shapes)
                if ids is not None:
                    temp_img = color_img.copy()
                    cv2.aruco.drawDetectedMarkers(
                        temp_img, corners, ids, config.MARKER_BORDER_COLOR
                    )
                    display_img = img_processor.flip(temp_img)

                dashboard = np.full(
                    (h + config.PANEL_HEIGHT, w, 3), config.BG_COLOR, dtype=np.uint8
                )
                display_img = draw_reference_shape(display_img, current_shape_id)

                # 2. Origin Box
                if origin_pos_robot_frame is None:
                    cv2.rectangle(
                        display_img,
                        (sq_x1, sq_y1),
                        (sq_x1 + config.SQUARE_SIZE, sq_y1 + config.SQUARE_SIZE),
                        (0, 200, 0),
                        2,
                    )

                if origin_screen_coords is not None:
                    cv2.drawMarker(
                        display_img,
                        origin_screen_coords,
                        (0, 255, 255),
                        cv2.MARKER_CROSS,
                        25,
                        2,
                    )

                # 3. User Cursor (The Marker Centroid)
                if visual_px > 0:
                    cv2.drawMarker(
                        display_img,
                        (visual_px, visual_py),
                        (0, 255, 255),
                        cv2.MARKER_CROSS,
                        20,
                        2,
                    )
                    cv2.rectangle(
                        display_img,
                        (visual_px - 10, visual_py - 10),
                        (visual_px + 10, visual_py + 10),
                        config.HIGHLIGHT_COLOR,
                        2,
                    )

                # 4. GHOST ROBOT REMOVED (WebSocket feature)

                # 5. Study UI Overlays (Start/End/Timer)
                if current_shape_id != 0:
                    # DRAW START
                    if start_pt:
                        cv2.circle(display_img, start_pt, 12, (0, 0, 255), -1)
                        cv2.putText(
                            display_img,
                            "START",
                            (start_pt[0] + 18, start_pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                    # DRAW END (CONDITIONAL)
                    # Show END only if we left start zone AND hit at least 1 checkpoint
                    if end_pt:
                        if has_left_start_zone and len(visited_checkpoints) > 0:
                            color_end = (
                                (0, 200, 0)
                                if completion_wait_start is None
                                else (255, 255, 255)
                            )
                            cv2.circle(display_img, end_pt, 12, color_end, -1)
                            cv2.putText(
                                display_img,
                                "END",
                                (end_pt[0] + 18, end_pt[1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color_end,
                                2,
                            )

                    # DRAW CHECKPOINTS
                    for idx, pt in enumerate(refs):
                        col = (
                            (0, 255, 0) if idx in visited_checkpoints else (0, 255, 255)
                        )
                        cv2.circle(display_img, pt, 10, col, -1)

                # 6. Dashboard Text
                dashboard[0:h, 0:w] = display_img
                y = h + 40

                # Title
                cv2.putText(
                    dashboard,
                    title,
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    config.TEXT_COLOR,
                    2,
                )

                # Tracking Info
                cv2.putText(
                    dashboard,
                    f"Status: {tracking_status}",
                    (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (
                        config.HIGHLIGHT_COLOR
                        if "Tracking" in tracking_status
                        else config.WARNING_COLOR
                    ),
                    2,
                )

                # Shape/State
                state_names = ["IDLE", "WAIT START", "TRACING", "COMPLETED"]
                cv2.putText(
                    dashboard,
                    f"Task State: {state_names[study_state]}",
                    (w // 2, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    config.TEXT_COLOR,
                    2,
                )

                cv2.putText(
                    dashboard,
                    f"Controls: 1-4 (Shapes) | C (Tare) | Q (Exit)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    config.TEXT_COLOR,
                    2,
                )
                y += 30

                # WEBSOCKET STATUS REMOVED

                if output_pose:
                    cv2.putText(
                        dashboard,
                        f"Pose X/Y/Z: {output_pose.x:.3f}, {output_pose.y:.3f}, {output_pose.z:.3f}",
                        (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        config.TEXT_COLOR,
                        2,
                    )
                    y += 30

                # Latency
                lat_ms = (time.time() - loop_start) * 1000
                cv2.putText(
                    dashboard,
                    f"Latency: {lat_ms:.1f} ms",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    config.TEXT_COLOR,
                    2,
                )

                # Timer Overlays (on main image)
                if study_state == STATE_TRACING:
                    elapsed = time.time() - tracing_start_time
                    remaining = max(0.0, config.TRACING_DURATION - elapsed)
                    if completion_wait_start:
                        hold_rem = max(
                            0.0,
                            config.COMPLETION_DELAY
                            - (time.time() - completion_wait_start),
                        )
                        cv2.putText(
                            display_img,
                            f"HOLD: {hold_rem:.1f}s",
                            (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 255, 255),
                            4,
                        )

                    cv2.putText(
                        display_img,
                        f"TIME: {remaining:.1f}s",
                        (w - 300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        4,
                    )

                dashboard[0:h, 0:w] = display_img

                # WEBSOCKET DATA TRANSMISSION REMOVED

                cv2.imshow(title, dashboard)

                # ---------------------------------------------------------
                # INPUT HANDLING
                # ---------------------------------------------------------
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("c"):
                    origin_quat = current_quat
                    manual_imu_calibration_done = True
                    logging.info(f"IMU Tare Complete. Front: {config.HOME_QUAT}")
                elif key == ord("t"):
                    orientation_test_mode = not orientation_test_mode
                elif key == ord("f"):
                    active_filter_mode = (
                        "EMA" if active_filter_mode == "EURO" else "EURO"
                    )
                    filter_needs_reset = True
                elif key == ord("0"):
                    current_shape_id = 0
                    study_state = STATE_IDLE
                    visited_checkpoints.clear()
                elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
                    current_shape_id = int(chr(key))
                    study_state = STATE_WAITING_START
                    logging.info(f"Shape {current_shape_id} Selected")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Loop Error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    finally:
        logging.info("Cleaning up...")
        vision.stop()
        imu.stop()
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()
