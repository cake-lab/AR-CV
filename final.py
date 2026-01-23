import pyrealsense2 as rs
import numpy as np
import cv2
import time
import signal
import sys
import serial
import threading
import mediapipe as mp
import asyncio
import websockets
import json
import serial.tools.list_ports
import socket
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
import math
from scipy.spatial.transform import Rotation as R
import logging
import traceback
import random
import struct  # REQUIRED FOR BINARY PROTOCOL


# ----------------------------
# Configuration & Constants
# ----------------------------
# CV Pre-processing Flags
ENABLE_CLAHE = False
ENABLE_SHARPENING = False


@dataclass
class AppConfig:
    # Robot Constants
    HOME_POS: np.ndarray = np.array([0.05, 0.0, 0.20])

    # GOLDEN FEATURE: The Robot's "Front Facing" Orientation
    HOME_QUAT: np.ndarray = np.array([0.0, 0.675, 0.0, 0.738])

    # AUTO-PILOT PRECISE ORIENTATION
    AUTO_QUAT: np.ndarray = np.array([0.0, 0.6757188764954564, 0.0, 0.7371594128461755])

    # SMART TRANSFORMATION MAPPING
    USER_FRAME_MAPPING: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

    # FILTER CONFIGURATION
    POS_FILTER_MIN_CUTOFF: float = 0.05
    POS_FILTER_BETA: float = 0.5  # TUNED for responsiveness (Retained)

    # Rotation Filter REMOVED in Phase 3
    ROT_FILTER_MIN_CUTOFF: float = 0.02
    ROT_FILTER_BETA: float = 0.02

    FILTER_DERIVATIVE_CUTOFF: float = 1.0

    # EMA Configuration (Simple Smoothing)
    EMA_ALPHA: float = 0.35

    # App Settings
    SERIAL_BAUD: int = 115200
    WEBSOCKET_PORT: int = 8765

    # Hand Configuration
    PRIMARY_HAND: str = "Left"

    # Camera Config (Native HD)
    DEPTH_WIDTH: int = 1024
    DEPTH_HEIGHT: int = 768
    COLOR_WIDTH: int = 1280
    COLOR_HEIGHT: int = 720
    FPS: int = 30

    # Logic
    FIST_EXIT_SECONDS: float = 200.0
    ORIGIN_SET_DURATION: float = 3.0
    MAX_DEPTH_DROPOUT_FRAMES: int = 5

    # UI Constants
    SQUARE_SIZE: int = 75
    PANEL_HEIGHT: int = 400
    BG_COLOR: Tuple[int, int, int] = (45, 45, 45)
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
    HIGHLIGHT_COLOR: Tuple[int, int, int] = (0, 255, 255)
    WARNING_COLOR: Tuple[int, int, int] = (0, 165, 255)
    OK_COLOR: Tuple[int, int, int] = (0, 210, 0)
    ROBOT_FEEDBACK_COLOR: Tuple[int, int, int] = (0, 0, 255)
    AUTO_PILOT_COLOR: Tuple[int, int, int] = (255, 0, 0)

    # Logic Thresholds
    LAG_THRESHOLD: float = 0.05

    # User Study Config
    SHAPE_COLOR: Tuple[int, int, int] = (0, 255, 0)
    SHAPE_THICKNESS: int = 9
    SHAPE_ALPHA: float = 0.5
    SHAPE_SCALE: float = 0.3
    SHAPE_OFFSET_Y: int = 150
    SAFETY_MIN_X: float = 0.200
    START_ZONE_RADIUS: int = 30
    CHECKPOINT_RADIUS: int = 30
    TRACING_DURATION: float = 60.0
    COMPLETION_DELAY: float = 2.0

    # Auto-Pilot Config
    AUTO_SPEED_PPS: float = 50.0


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


class TrajectoryGenerator:
    @staticmethod
    def lerp(
        p1: Tuple[float, float], p2: Tuple[float, float], t: float
    ) -> Tuple[float, float]:
        return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)

    @staticmethod
    def get_auto_target(
        shape_id: int, progress: float, cx: int, cy: int, base_size: int
    ) -> Tuple[float, float]:
        if shape_id == 1:  # Square
            half = base_size // 2
            p_tl = (cx - half, cy - half)
            p_tr = (cx + half, cy - half)
            p_br = (cx + half, cy + half)
            p_bl = (cx - half, cy + half)

            if progress < 0.25:
                return TrajectoryGenerator.lerp(p_tl, p_tr, progress * 4)
            elif progress < 0.50:
                return TrajectoryGenerator.lerp(p_tr, p_br, (progress - 0.25) * 4)
            elif progress < 0.75:
                return TrajectoryGenerator.lerp(p_br, p_bl, (progress - 0.50) * 4)
            else:
                return TrajectoryGenerator.lerp(p_bl, p_tl, (progress - 0.75) * 4)

        elif shape_id == 2:  # Circle
            radius = base_size // 2
            angle = -math.pi / 2 + (2 * math.pi * progress)
            return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

        elif shape_id == 3:  # S-Shape
            half_h = base_size // 2
            half_w = base_size // 4
            y = (cy - half_h) + (2 * half_h * progress)
            x = cx + half_w * math.cos(progress * 2 * math.pi * 1.5)
            return (x, y)

        elif shape_id == 4:  # Triangle
            radius = base_size // 2
            p_top = (cx, cy - radius)
            p_right = (cx + radius, cy + radius)
            p_left = (cx - radius, cy + radius)

            if progress < 0.333:
                return TrajectoryGenerator.lerp(p_top, p_right, progress * 3)
            elif progress < 0.666:
                return TrajectoryGenerator.lerp(p_right, p_left, (progress - 0.333) * 3)
            else:
                return TrajectoryGenerator.lerp(p_left, p_top, (progress - 0.666) * 3)

        elif shape_id == 5 or shape_id == 6:  # Rectangles
            half_h = base_size // 2
            half_w = int(base_size * 0.75)
            p_tl = (cx - half_w, cy - half_h)
            p_tr = (cx + half_w, cy - half_h)
            p_br = (cx + half_w, cy + half_h)
            p_bl = (cx - half_w, cy + half_h)

            w_len = half_w * 2
            h_len = half_h * 2
            perimeter = 2 * (w_len + h_len)
            s1 = w_len / perimeter
            s2 = h_len / perimeter
            s3 = w_len / perimeter

            if progress < s1:
                return TrajectoryGenerator.lerp(p_tl, p_tr, progress / s1)
            elif progress < (s1 + s2):
                return TrajectoryGenerator.lerp(p_tr, p_br, (progress - s1) / s2)
            elif progress < (s1 + s2 + s3):
                return TrajectoryGenerator.lerp(p_br, p_bl, (progress - (s1 + s2)) / s3)
            else:
                s4 = 1.0 - (s1 + s2 + s3)
                return TrajectoryGenerator.lerp(
                    p_bl, p_tl, (progress - (s1 + s2 + s3)) / s4
                )

        return (cx, cy)


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
            else:
                logging.info("CUDA not found. Using CPU for image processing.")
        except AttributeError:
            logging.info("cv2.cuda module not found. Using CPU.")
        except Exception as e:
            logging.warning(f"Error initializing CUDA: {e}. Using CPU.")

    def to_rgb(self, bgr_img: np.ndarray) -> np.ndarray:
        if self.use_cuda:
            try:
                self.gpu_mat.upload(bgr_img)
                gpu_rgb = cv2.cuda.cvtColor(self.gpu_mat, cv2.COLOR_BGR2RGB)
                return gpu_rgb.download()
            except Exception:
                return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

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
        self.firmware_baud = 500000

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
        while self.running:
            port = self._find_port()
            if not port:
                time.sleep(1.0)
                continue

            try:
                logging.info(
                    f"Opening IMU Serial: {port} @ {self.firmware_baud} (BINARY)..."
                )
                self.ser = serial.Serial(port, self.firmware_baud, timeout=0.1)

                logging.info("Waiting 2.0s for ESP32 Boot...")
                time.sleep(2.0)

                self.ser.reset_input_buffer()
                logging.info("Sending 'start' command...")
                self.ser.write(b"start\n")

                logging.info("IMU Connection LOCKED. Streaming Binary...")
                last_data_time = time.time()

                while self.running:
                    # 22 Bytes: Header(1) + Timestamp(4) + Quat(16) + Checksum(1)
                    if self.ser.in_waiting >= 22:

                        # 1. Sync: Find Header (0xAA)
                        header = self.ser.read(1)
                        if header != b"\xaa":
                            # Not aligned. Continue loop (Self-Healing)
                            continue

                        # 2. Read remaining 21 bytes
                        payload = self.ser.read(21)
                        if len(payload) != 21:
                            continue

                        # 3. Checksum Verification
                        # XOR sum of Header(0xAA) + first 20 bytes of payload
                        calc_checksum = 0xAA
                        data_bytes = payload[0:20]
                        received_checksum = payload[20]

                        for b in data_bytes:
                            calc_checksum ^= b

                        if calc_checksum != received_checksum:
                            logging.warning("Checksum Mismatch! Dropping Packet.")
                            continue

                        # 4. Unpack Data
                        # < = Little Endian, I = uint32, 4f = 4 floats
                        try:
                            ts_micros, w, x, y, z = struct.unpack("<I4f", data_bytes)

                            # Sanity check
                            if (
                                abs(w) > 1.1
                                or abs(x) > 1.1
                                or abs(y) > 1.1
                                or abs(z) > 1.1
                            ):
                                continue

                            # Reconstruct Quaternion (x, y, z, w)
                            q = np.array([x, y, z, w])

                            with self.lock:
                                self.latest_quat = q

                            last_data_time = time.time()

                        except Exception as e:
                            logging.error(f"Unpack Error: {e}")
                            continue

                    else:
                        # Sleep briefly to allow buffer to fill (prevents CPU spin)
                        time.sleep(0.0001)
                        if time.time() - last_data_time > 1.0:
                            logging.warning(
                                "IMU Stream Silent (>1s). Resetting connection."
                            )
                            break

            except Exception as e:
                logging.error(f"IMU Serial Error: {e}")
                time.sleep(1.0)

            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass

    def get_quat(self) -> np.ndarray:
        with self.lock:
            return self.latest_quat.copy()

    def start_stream(self):
        pass

    def stop(self):
        self.running = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


class AsyncWebSocketServer:
    def __init__(self):
        self.clients = set()
        self.latest_payload = None
        self.latest_feedback = None
        self.lock = asyncio.Lock()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)

    def start(self):
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        logging.info(f"WebSocket serving on {config.WEBSOCKET_PORT}")
        try:
            # PHASE 4 FIX: Ensure TCP_NODELAY is respected (Library defaults to True usually, but validation is key)
            async with websockets.serve(
                self._register, "localhost", config.WEBSOCKET_PORT
            ):
                await self._broadcast_loop()
        except OSError as e:
            logging.error(f"WebSocket port blocked: {e}")

    async def _register(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "feedback":
                        async with self.lock:
                            self.latest_feedback = data
                except Exception as e:
                    logging.warning(f"WS Input Error: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)

    def update_data(self, data: Dict):
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._set_data(data), self.loop)

    def get_feedback(self) -> Optional[Dict]:
        if self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._get_feedback_async(), self.loop
            )
            try:
                return future.result(timeout=0.05)
            except Exception:
                return None
        return None

    async def _get_feedback_async(self):
        async with self.lock:
            return self.latest_feedback

    async def _set_data(self, data):
        async with self.lock:
            self.latest_payload = json.dumps(data)

    async def _broadcast_loop(self):
        while True:
            # PHASE 4 FIX: ARTIFICIAL LATENCY REMOVAL
            # Old: await asyncio.sleep(0.033) -> ~30Hz (33ms lag)
            # New: await asyncio.sleep(0.008) -> ~120Hz (8ms lag)
            await asyncio.sleep(0.008)

            async with self.lock:
                payload = self.latest_payload

            if payload and self.clients:
                websockets.broadcast(self.clients, payload)

    def get_client_count(self):
        return len(self.clients)


# ----------------------------
# Vision Pipeline (OPTIMIZED)
# ----------------------------
class VisionSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
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

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def start(self):
        self.profile = self.pipeline.start(self.rs_config)

        # ---------------------------------------------------------------------
        # PHASE 4 FIX: REALSENSE LATENCY KILLER
        # ---------------------------------------------------------------------
        # The RealSense SDK defaults to a queue size of 16 or 2 depending on FW.
        # At 30FPS, a 2-frame buffer = 66ms of latency before we even see it.
        # Setting this to 1 ensures we get the "freshest" frame possible.
        # ---------------------------------------------------------------------
        try:
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()
            if depth_sensor.supports(rs.option.frames_queue_size):
                depth_sensor.set_option(rs.option.frames_queue_size, 1.0)
                logging.info("RealSense: Queue Size forced to 1 (Low Latency Mode)")
            else:
                logging.warning("RealSense: frames_queue_size not supported.")
        except Exception as e:
            logging.warning(f"RealSense Config Error: {e}")

        self.depth_scale = device.first_depth_sensor().get_depth_scale()

        self.intrinsics = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _preprocess_frame(
        self, color_img: np.ndarray, depth_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if ENABLE_CLAHE:
            try:
                lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = self.clahe.apply(l)
                lab = cv2.merge((l, a, b))
                color_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception:
                pass

        if ENABLE_SHARPENING:
            try:
                color_img = cv2.filter2D(color_img, -1, self.kernel_sharpen)
            except Exception:
                pass

        return color_img, depth_img

    def _capture_loop(self):
        while self.running:
            try:
                # -----------------------------------------------------------------
                # PHASE 4 FIX: THREAD STARVATION
                # -----------------------------------------------------------------
                # OLD: frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                # ISSUE: wait_for_frames is blocking. It can hold the GIL or starve
                # the IMU thread if the camera is slightly out of sync.
                # NEW: poll_for_frames. We check instantly. If nothing, we yield.
                # -----------------------------------------------------------------
                frames = self.pipeline.poll_for_frames()

                if not frames:
                    # Yield CPU explicitly to let IMUReader thread run
                    time.sleep(0.001)
                    continue

                capture_time = time.time()
                aligned_frames = self.align.process(frames)

                c_frame = aligned_frames.get_color_frame()
                d_frame = aligned_frames.get_depth_frame()

                if not c_frame or not d_frame:
                    continue

                color_arr = np.asanyarray(c_frame.get_data())
                depth_arr = np.asanyarray(d_frame.get_data())

                processed_color, processed_depth = self._preprocess_frame(
                    color_arr, depth_arr
                )

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

    def is_generic_fist(self, landmarks) -> bool:
        lm = landmarks.landmark
        tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP,
        ]
        pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP,
        ]

        curled = sum(1 for tip, pip in zip(tips, pips) if lm[tip].y > lm[pip].y)

        thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
        index_mcp = lm[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        if abs(thumb_tip.x - index_mcp.x) < 0.05:
            curled += 1

        return curled >= 3


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
    elif shape_id == 3:  # S-Shape
        half_h = base_size // 2
        half_w = base_size // 4
        start_pt = (int(cx + half_w * math.cos(0)), int((cy - half_h)))
        end_pt = (
            int(cx + half_w * math.cos(1.0 * 2 * math.pi * 1.5)),
            int((cy + half_h)),
        )
        refs = [(cx, cy)]

    elif shape_id == 4:  # Triangle
        radius = base_size // 2
        start_pt = (cx, cy - radius)
        end_pt = start_pt
        refs = [(cx + radius, cy + radius), (cx - radius, cy + radius)]

    elif shape_id == 5:  # Rectangle
        half_h = base_size // 2
        half_w = int(base_size * 0.75)
        start_pt = (cx - half_w, cy - half_h)
        end_pt = start_pt
        refs = [
            (cx + half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
        ]
    elif shape_id == 6:  # Rectangle (Fixed X)
        half_h = base_size // 2
        half_w = int(base_size * 0.75)
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
        pts = []
        for i in range(50):
            prog = i / 49.0
            px, py = TrajectoryGenerator.get_auto_target(3, prog, cx, cy, base_size)
            pts.append((int(px), int(py)))
        cv2.polylines(overlay, [np.array(pts)], False, config.SHAPE_COLOR, thickness)

    elif shape_id == 4:
        radius = base_size // 2
        p1 = (cx, cy - radius)
        p2 = (cx + radius, cy + radius)
        p3 = (cx - radius, cy + radius)
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
    elif shape_id == 6:
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
    ws_server = AsyncWebSocketServer()
    ws_server.start()
    vision = VisionSystem()
    try:
        vision.start()
        logging.info("Camera initialized.")
    except RuntimeError as e:
        logging.critical(f"Camera failed: {e}")
        return

    origin_pos_robot_frame = None
    origin_quat = np.array([0.0, 0.0, 0.0, 1.0])

    origin_set_start = None
    origin_screen_coords = None
    manual_imu_calibration_done = False

    fist_start_time = None
    fist_active = False

    pos_filter = None
    is_first_detection = True
    orientation_test_mode = False
    use_precision_mode = False
    simulate_error_mode = False
    active_filter_mode = "EURO"
    filter_needs_reset = False

    last_valid_target_cam = None
    depth_dropout_counter = 0

    auto_mode = False
    auto_progress = 0.0
    last_time = time.time()

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

    title = "Real-Time Hand Tracking Dashboard (Phase 5 BINARY)"
    logging.info("System initialized. Starting main loop...")

    try:
        while True:
            robot_fb = None
            packet_dropped = False
            try:
                loop_start = time.time()
                dt = loop_start - last_time
                last_time = loop_start

                color_img, depth_img, frame_ts = vision.get_frames()
                if color_img is None or depth_img is None:
                    time.sleep(0.001)
                    continue

                # 2. Get IMU Data IMMEDIATELY
                current_quat_raw = imu.get_quat()

                if frame_ts == 0.0:
                    frame_ts = loop_start

                img_rgb = img_processor.to_rgb(color_img)
                h, w, _ = color_img.shape

                results = vision.hands.process(img_rgb)

                current_target_cam = None
                raw_px, raw_py = 0, 0
                tracking_mode = "None"
                num_visible = 0
                pinch_triangle_pts = []

                tracking_hand_landmarks = None
                is_fist = False

                if results.multi_hand_landmarks and results.multi_handedness:
                    for idx, hand_info in enumerate(results.multi_handedness):
                        label = hand_info.classification[0].label
                        lm_instance = results.multi_hand_landmarks[idx]

                        if label == config.PRIMARY_HAND:
                            tracking_hand_landmarks = lm_instance
                        else:
                            if vision.is_generic_fist(lm_instance):
                                is_fist = True

                if is_fist:
                    if not fist_active:
                        fist_active = True
                        fist_start_time = time.time()
                    elif time.time() - fist_start_time >= config.FIST_EXIT_SECONDS:
                        print("Safety Fist Detected. Exiting...")
                        break
                else:
                    fist_active = False
                    fist_start_time = None

                if auto_mode and current_shape_id != 0:
                    cx = w // 2
                    cy = (h // 2) + config.SHAPE_OFFSET_Y
                    base_size = int(h * config.SHAPE_SCALE)

                    perimeter = base_size * 4
                    if current_shape_id == 2:
                        perimeter = 2 * math.pi * (base_size / 2)
                    elif current_shape_id == 3:
                        perimeter = base_size * 2

                    increment = (config.AUTO_SPEED_PPS * dt) / perimeter
                    auto_progress = auto_progress + increment
                    if auto_progress > 1.0:
                        auto_progress = 1.0

                    auto_px, auto_py = TrajectoryGenerator.get_auto_target(
                        current_shape_id, auto_progress, cx, cy, base_size
                    )

                    raw_px, raw_py = int(auto_px), int(auto_py)
                    fixed_depth = 0.4
                    current_target_cam = np.array(
                        rs.rs2_deproject_pixel_to_point(
                            vision.intrinsics, [raw_px, raw_py], fixed_depth
                        )
                    )
                    tracking_mode = "AUTO-PILOT"
                    num_visible = 1
                    current_quat = config.AUTO_QUAT

                elif tracking_hand_landmarks:
                    lm = tracking_hand_landmarks

                    if use_precision_mode:
                        p_thumb = lm.landmark[vision.mp_hands.HandLandmark.THUMB_TIP]
                        p_index = lm.landmark[
                            vision.mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ]
                        p_middle = lm.landmark[
                            vision.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                        ]

                        cx_norm = (p_thumb.x + p_index.x + p_middle.x) / 3.0
                        cy_norm = (p_thumb.y + p_index.y + p_middle.y) / 3.0
                        raw_px, raw_py = int(cx_norm * w), int(cy_norm * h)

                        pinch_triangle_pts = [
                            (int(p_thumb.x * w), int(p_thumb.y * h)),
                            (int(p_index.x * w), int(p_index.y * h)),
                            (int(p_middle.x * w), int(p_middle.y * h)),
                        ]
                        tracking_mode = "Precision"
                    else:
                        mcp = lm.landmark[
                            vision.mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ]
                        raw_px, raw_py = int(mcp.x * w), int(mcp.y * h)
                        tracking_mode = "Primary"

                    dist = vision.get_depth_vectorized(depth_img, raw_px, raw_py)

                    if dist > 0:
                        current_target_cam = np.array(
                            rs.rs2_deproject_pixel_to_point(
                                vision.intrinsics, [raw_px, raw_py], dist
                            )
                        )
                        num_visible = 1
                        last_valid_target_cam = current_target_cam
                        depth_dropout_counter = 0
                    elif (
                        last_valid_target_cam is not None
                        and depth_dropout_counter < config.MAX_DEPTH_DROPOUT_FRAMES
                    ):
                        current_target_cam = last_valid_target_cam
                        depth_dropout_counter += 1
                        num_visible = 1
                        tracking_mode += " (LKV)"

                visual_px = w - raw_px if raw_px > 0 else 0
                visual_py = raw_py

                tracked_pos_cam = None

                if current_target_cam is not None:
                    if auto_mode:
                        tracked_pos_cam = current_target_cam
                    elif is_first_detection or pos_filter is None or filter_needs_reset:
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
                    pos_filter = None

                if not auto_mode:
                    current_quat = MathUtils.normalize_quat(current_quat_raw)
                else:
                    current_quat = config.AUTO_QUAT

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
                        logging.info("Origin Set!")
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

                    if auto_mode:
                        output_pose.x = 0.220
                    else:
                        if orientation_test_mode:
                            output_pose.x = float(config.HOME_POS[0])
                            output_pose.y = float(config.HOME_POS[1])
                            output_pose.z = float(config.HOME_POS[2])

                        if current_shape_id == 4:
                            output_pose.x = 0.250
                        elif current_shape_id == 6:
                            output_pose.x = 0.220

                safety_violation = False
                if output_pose and current_shape_id != 0:
                    if (
                        current_shape_id not in [4, 6]
                        and not auto_mode
                        and output_pose.x < config.SAFETY_MIN_X
                    ):
                        safety_violation = True

                start_pt, end_pt, refs = get_shape_targets(current_shape_id, w, h)

                if study_state == STATE_WAITING_START:
                    visited_checkpoints.clear()
                    completion_wait_start = None

                    if auto_mode:
                        auto_progress = 0.0
                        study_state = STATE_TRACING
                        tracing_start_time = time.time()
                        has_left_start_zone = False
                        logging.info("Auto-Pilot: Tracing Started")

                    elif not safety_violation and start_pt and visual_px > 0:
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

                        if auto_mode and auto_progress >= 1.0:
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
                            if not auto_mode:
                                completion_wait_start = None

                elif study_state == STATE_COMPLETED:
                    if last_frozen_pose:
                        output_pose = last_frozen_pose

                display_img = img_processor.flip(color_img)
                dashboard = np.full(
                    (h + config.PANEL_HEIGHT, w, 3), config.BG_COLOR, dtype=np.uint8
                )
                display_img = draw_reference_shape(display_img, current_shape_id)

                if current_shape_id != 0:
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
                    if end_pt:
                        color_end = (0, 200, 0)
                        if completion_wait_start is not None:
                            color_end = (255, 255, 255)

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

                    for idx, pt in enumerate(refs):
                        col = (
                            (0, 255, 0) if idx in visited_checkpoints else (0, 255, 255)
                        )
                        cv2.circle(display_img, pt, 10, col, -1)

                robot_fb = ws_server.get_feedback()
                if robot_fb and origin_pos_robot_frame is not None:
                    rx = robot_fb.get("x", config.HOME_POS[0])
                    ry = robot_fb.get("y", config.HOME_POS[1])
                    rz = robot_fb.get("z", config.HOME_POS[2])
                    r_pos = np.array([rx, ry, rz])
                    rel_pos = r_pos - config.HOME_POS
                    robot_frame_pos = rel_pos + origin_pos_robot_frame
                    cam_x, cam_y, cam_z = (
                        robot_frame_pos[1],
                        -robot_frame_pos[2],
                        -robot_frame_pos[0],
                    )

                    if cam_z > 0 and vision.intrinsics:
                        fb_pixel = rs.rs2_project_point_to_pixel(
                            vision.intrinsics, [cam_x, cam_y, cam_z]
                        )
                        raw_fb_px, raw_fb_py = int(fb_pixel[0]), int(fb_pixel[1])
                        vis_fb_px = w - raw_fb_px
                        cv2.drawMarker(
                            display_img,
                            (vis_fb_px, raw_fb_py),
                            config.ROBOT_FEEDBACK_COLOR,
                            cv2.MARKER_TILTED_CROSS,
                            25,
                            3,
                        )
                        cv2.putText(
                            display_img,
                            "ROBOT",
                            (vis_fb_px + 10, raw_fb_py),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            config.ROBOT_FEEDBACK_COLOR,
                            2,
                        )

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

                if visual_px > 0:
                    marker_color = (
                        config.AUTO_PILOT_COLOR if auto_mode else (0, 255, 255)
                    )
                    cv2.drawMarker(
                        display_img,
                        (visual_px, visual_py),
                        marker_color,
                        cv2.MARKER_CROSS,
                        20,
                        2,
                    )
                    cv2.rectangle(
                        display_img,
                        (visual_px - 12, visual_py - 12),
                        (visual_px + 12, visual_py + 12),
                        (0, 255, 0) if not auto_mode else config.AUTO_PILOT_COLOR,
                        2,
                    )
                    if (
                        use_precision_mode
                        and len(pinch_triangle_pts) == 3
                        and not auto_mode
                    ):
                        tri_visual = []
                        for tx, ty in pinch_triangle_pts:
                            tri_visual.append((w - tx, ty))
                        cv2.line(
                            display_img, tri_visual[0], tri_visual[1], (255, 255, 0), 2
                        )
                        cv2.line(
                            display_img, tri_visual[1], tri_visual[2], (255, 255, 0), 2
                        )
                        cv2.line(
                            display_img, tri_visual[2], tri_visual[0], (255, 255, 0), 2
                        )

                if output_pose and robot_fb:
                    t_vec = np.array([output_pose.x, output_pose.y, output_pose.z])
                    a_vec = np.array(
                        [
                            robot_fb.get("x", 0),
                            robot_fb.get("y", 0),
                            robot_fb.get("z", 0),
                        ]
                    )
                    deviation = np.linalg.norm(t_vec - a_vec)
                    if deviation > config.LAG_THRESHOLD:
                        text = "SLOW DOWN - HIGH LAG"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_size = cv2.getTextSize(text, font, 1.2, 3)[0]
                        text_x = (w - text_size[0]) // 2
                        if int(time.time() * 5) % 2 == 0:
                            cv2.putText(
                                display_img,
                                text,
                                (text_x, h // 2),
                                font,
                                1.2,
                                config.ROBOT_FEEDBACK_COLOR,
                                3,
                            )

                if safety_violation:
                    text = "MOVE HAND FORWARD (>20cm)"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[
                        0
                    ]
                    text_x = (w - text_size[0]) // 2
                    cv2.putText(
                        display_img,
                        text,
                        (text_x, h // 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 165, 255),
                        3,
                    )

                if study_state == STATE_TRACING:
                    elapsed = time.time() - tracing_start_time
                    remaining = max(0.0, config.TRACING_DURATION - elapsed)
                    timer_text = f"TIME: {remaining:.1f}s"

                    if completion_wait_start is not None:
                        hold_elapsed = time.time() - completion_wait_start
                        hold_remain = max(0.0, config.COMPLETION_DELAY - hold_elapsed)
                        hold_text = f"HOLD: {hold_remain:.1f}s"

                        ht_size = cv2.getTextSize(
                            hold_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4
                        )[0]
                        cv2.putText(
                            display_img,
                            hold_text,
                            (w // 2 - ht_size[0] // 2, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 255, 255),
                            4,
                        )

                    text_size = cv2.getTextSize(
                        timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4
                    )[0]
                    cv2.putText(
                        display_img,
                        timer_text,
                        (w - text_size[0] - 25, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        4,
                    )
                    if remaining == 0:
                        cv2.putText(
                            display_img,
                            "TIME UP!",
                            (w - 250, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            4,
                        )

                if study_state == STATE_COMPLETED:
                    msg = "TASK COMPLETE - Press ESC/0"
                    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[
                        0
                    ]
                    text_x = (w - text_size[0]) // 2
                    cv2.putText(
                        display_img,
                        msg,
                        (text_x, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                    )

                if fist_active and fist_start_time:
                    elapsed = time.time() - fist_start_time
                    ratio = min(elapsed / config.FIST_EXIT_SECONDS, 1.0)
                    cv2.ellipse(
                        display_img,
                        (80, 80),
                        (30, 30),
                        0,
                        -90,
                        -90 + 360 * ratio,
                        (60, 180, 255),
                        4,
                    )

                dashboard[0:h, 0:w] = display_img

                y = h + 40
                cv2.putText(
                    dashboard,
                    title,
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    config.TEXT_COLOR,
                    2,
                )

                shape_names = [
                    "None",
                    "Square",
                    "Circle",
                    "Letter S",
                    "Triangle (Fixed X)",
                    "Horizontal Rectangle",
                    "Rectangle (Fixed X)",
                ]
                active_shape_name = shape_names[current_shape_id]
                state_names = ["IDLE", "WAIT START", "TRACING", "COMPLETED"]
                status_text = (
                    f"Task: {active_shape_name} | State: {state_names[study_state]}"
                )
                cv2.putText(
                    dashboard,
                    status_text,
                    (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

                if study_state == STATE_TRACING:
                    check_text = f"Checkpoints: {len(visited_checkpoints)}/{len(refs)}"
                    cv2.putText(
                        dashboard,
                        check_text,
                        (w - 300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )

                source_text = "PINCH CENTROID" if use_precision_mode else "MCP"
                control_text = (
                    f"CONTROL HAND: {config.PRIMARY_HAND.upper()} (Press 'H')"
                )
                cv2.putText(
                    dashboard,
                    f"MODE: {source_text} | {control_text} | Tracking: {tracking_mode}",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    config.HIGHLIGHT_COLOR,
                    2,
                )
                y += 30

                exit_hand = "RIGHT" if config.PRIMARY_HAND == "Left" else "LEFT"
                cv2.putText(
                    dashboard,
                    f"Close {exit_hand} fist for {int(config.FIST_EXIT_SECONDS)}s to exit. | 'W' Precision",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    config.TEXT_COLOR,
                    2,
                )
                y += 30

                client_count = ws_server.get_client_count()
                ws_status = "Connected" if client_count > 0 else "Disconnected"
                ws_color = config.OK_COLOR if client_count > 0 else config.WARNING_COLOR
                cv2.putText(
                    dashboard,
                    f"WebSocket: {ws_status} (Clients: {client_count})",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    ws_color,
                    2,
                )
                y += 30

                origin_text = (
                    "Set"
                    if origin_pos_robot_frame is not None
                    else (
                        "Setting..."
                        if origin_set_start
                        else "Not Set (Hold in green square)"
                    )
                )
                cv2.putText(
                    dashboard,
                    f"Origin Status: {origin_text}",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    config.TEXT_COLOR,
                    2,
                )
                y += 30

                cal_status = (
                    "MANUAL (C)" if manual_imu_calibration_done else "AUTO (GreenBox)"
                )
                cal_color = (
                    (0, 255, 0) if manual_imu_calibration_done else (200, 200, 200)
                )
                cv2.putText(
                    dashboard,
                    f"IMU Calibration: {cal_status} (Press 'C' to Tare)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cal_color,
                    2,
                )
                y += 30

                if output_pose:
                    cv2.putText(
                        dashboard,
                        f"Target Pos (m): x={output_pose.x:.3f}, y={output_pose.y:.3f}, z={output_pose.z:.3f}",
                        (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        config.TEXT_COLOR,
                        2,
                    )
                    y += 28
                    cv2.putText(
                        dashboard,
                        f"Final Quat: qx={output_pose.qx:+.2f}, qy={output_pose.qy:+.2f}, qz={output_pose.qz:+.2f}, qw={output_pose.qw:+.2f}",
                        (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        config.TEXT_COLOR,
                        2,
                    )
                    y += 28

                if robot_fb:
                    cv2.putText(
                        dashboard,
                        f"ACTUAL Robot: x={robot_fb.get('x',0):.3f}, y={robot_fb.get('y',0):.3f}, z={robot_fb.get('z',0):.3f}",
                        (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        config.ROBOT_FEEDBACK_COLOR,
                        2,
                    )
                    y += 28

                lat_ms = (time.time() - loop_start) * 1000
                cv2.putText(
                    dashboard,
                    f"Loop Latency: {lat_ms:.1f} ms",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    config.TEXT_COLOR,
                    2,
                )
                y += 30

                test_text = (
                    "ON - Position FIXED"
                    if orientation_test_mode
                    else "OFF - Press 't' to toggle"
                )
                test_color = (0, 255, 0) if orientation_test_mode else config.TEXT_COLOR
                cv2.putText(
                    dashboard,
                    f"Orientation Test Mode: {test_text}",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    test_color,
                    2,
                )

                err_text = (
                    "ON (5% Loss)"
                    if simulate_error_mode
                    else "OFF - Press 'E' to toggle"
                )
                err_color = (0, 0, 255) if simulate_error_mode else config.TEXT_COLOR
                y += 30

                auto_text = "ACTIVE (Running)" if auto_mode else "OFF (Press 'A')"
                auto_color = config.AUTO_PILOT_COLOR if auto_mode else config.TEXT_COLOR

                filter_color = config.HIGHLIGHT_COLOR
                cv2.putText(
                    dashboard,
                    f"Auto-Pilot: {auto_text} | FILTER: {active_filter_mode} (Press 'F')",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    auto_color,
                    2,
                )

                if packet_dropped:
                    cv2.putText(
                        dashboard,
                        "PACKET LOSS",
                        (w - 200, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                should_send = True
                if simulate_error_mode and random.random() < 0.05:
                    should_send = False
                    packet_dropped = True
                    cv2.putText(
                        dashboard,
                        "PACKET LOSS",
                        (w - 220, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )

                if output_pose and should_send:
                    payload = {
                        "timestamp": time.time(),
                        "capture_timestamp": frame_ts,
                        "mode": "position",
                        "position": asdict(output_pose),
                    }
                    ws_server.update_data(payload)

                cv2.imshow(title, dashboard)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    origin_quat = current_quat
                    manual_imu_calibration_done = True
                    logging.info(
                        f"IMU Tare Complete. Robot Reset to Front: {config.HOME_QUAT}"
                    )
                elif key == ord("t"):
                    orientation_test_mode = not orientation_test_mode
                    logging.info(f"Test Mode: {orientation_test_mode}")
                elif key == ord("w"):
                    use_precision_mode = not use_precision_mode
                    mode_name = (
                        "PRECISION (Pinch Centroid)"
                        if use_precision_mode
                        else "STANDARD (MCP)"
                    )
                    logging.info(f"Tracking Mode toggled: {mode_name}")
                elif key == ord("h"):
                    config.PRIMARY_HAND = (
                        "Left" if config.PRIMARY_HAND == "Right" else "Right"
                    )
                    logging.info(f"Control Hand Swapped to: {config.PRIMARY_HAND}")
                elif key == ord("e"):
                    simulate_error_mode = not simulate_error_mode
                    logging.info(f"Simulate Error Mode: {simulate_error_mode}")
                elif key == ord("f"):
                    active_filter_mode = (
                        "EMA" if active_filter_mode == "EURO" else "EURO"
                    )
                    filter_needs_reset = True
                    logging.info(f"Active Filter Swapped to: {active_filter_mode}")
                elif key == ord("a"):
                    auto_mode = not auto_mode
                    if auto_mode:
                        auto_progress = 0.0
                        logging.info("Auto-Pilot: ENABLED (Ground Truth Mode)")
                    else:
                        logging.info("Auto-Pilot: DISABLED (Manual Mode)")
                elif key == ord("0") or key == ord("x") or key == 27:
                    current_shape_id = 0
                    study_state = STATE_IDLE
                    visited_checkpoints.clear()
                    completion_wait_start = None
                    auto_mode = False
                    logging.info("Shape: Cleared / Reset")
                elif key == ord("1"):
                    current_shape_id = 1
                    study_state = STATE_WAITING_START
                    auto_progress = 0.0
                    logging.info("Shape: Square")
                elif key == ord("2"):
                    current_shape_id = 2
                    study_state = STATE_WAITING_START
                    auto_progress = 0.0
                    logging.info("Shape: Circle")
                elif key == ord("3"):
                    current_shape_id = 3
                    study_state = STATE_WAITING_START
                    auto_progress = 0.0
                    logging.info("Shape: Letter S")
                elif key == ord("4"):
                    current_shape_id = 4
                    study_state = STATE_WAITING_START
                    auto_progress = 0.0
                    logging.info("Shape: Triangle (Fixed X)")
                elif key == ord("5"):
                    current_shape_id = 5
                    study_state = STATE_WAITING_START
                    auto_progress = 0.0
                    logging.info("Shape: Horizontal Rectangle")
                elif key == ord("6"):
                    current_shape_id = 6
                    study_state = STATE_WAITING_START
                    auto_progress = 0.0
                    logging.info("Shape: Rectangle (Fixed X=0.220)")

            except KeyboardInterrupt:
                logging.info("User interrupted.")
                break
            except Exception as e:
                logging.error(f"CRITICAL LOOP ERROR: {e}")
                traceback.print_exc()
                time.sleep(0.1)
                continue

    finally:
        logging.info("Cleaning up resources...")
        try:
            vision.stop()
            imu.stop()
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
        logging.info("Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
