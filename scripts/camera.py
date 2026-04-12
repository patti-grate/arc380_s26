import json
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

SHARED_DIR = ROOT_DIR / "realsense_shared"
REQUEST_PATH = SHARED_DIR / "request.json"
READY_PATH = SHARED_DIR / "ready.json"
COLOR_PATH = SHARED_DIR / "color.png"
DEPTH_PATH = SHARED_DIR / "depth.npy"
META_PATH = SHARED_DIR / "meta.json"

POLL_INTERVAL_SEC = 0.2
CAPTURE_TIMEOUT_MS = 5000

COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080
COLOR_FPS = 30

DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS = 30

ALIGN_DEPTH_TO_COLOR = True


@dataclass
class ReadyPayload:
    request_id: int
    status: str
    timestamp_utc: str
    color_path: str
    depth_path: str
    meta_path: str


@dataclass
class MetaPayload:
    request_id: int
    timestamp_utc: str
    device_name: str
    device_serial: str
    depth_scale_m_per_unit: float
    color_shape: list[int]
    depth_shape: list[int]
    color_dtype: str
    depth_dtype: str
    aligned_to_color: bool


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
        newline="\n",
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def atomic_write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        delete=False,
        suffix=".tmp.png",
    ) as tmp:
        tmp_path = Path(tmp.name)

    ok = cv2.imwrite(str(tmp_path), image)
    if not ok:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to write PNG to temporary path: {tmp_path}")

    os.replace(tmp_path, path)


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        delete=False,
        suffix=".tmp.npy",
    ) as tmp:
        tmp_path = Path(tmp.name)

    with open(tmp_path, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class RealSenseCaptureServer:
    def __init__(self) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color) if ALIGN_DEPTH_TO_COLOR else None
        self.running = True
        self.last_request_id = -1
        self.pipeline_profile: rs.pipeline_profile | None = None
        self.depth_scale = 0.001
        self.device_name = "unknown"
        self.device_serial = "unknown"

    def start(self) -> None:
        SHARED_DIR.mkdir(parents=True, exist_ok=True)

        self.config.enable_stream(
            rs.stream.color,
            COLOR_WIDTH,
            COLOR_HEIGHT,
            rs.format.bgr8,
            COLOR_FPS,
        )
        self.config.enable_stream(
            rs.stream.depth,
            DEPTH_WIDTH,
            DEPTH_HEIGHT,
            rs.format.z16,
            DEPTH_FPS,
        )

        self.pipeline_profile = self.pipeline.start(self.config)

        device = self.pipeline_profile.get_device()
        self.device_name = device.get_info(rs.camera_info.name)
        self.device_serial = device.get_info(rs.camera_info.serial_number)

        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        for _ in range(10):
            self.pipeline.wait_for_frames()

        print(f"Started RealSense capture server for {self.device_name} ({self.device_serial})")

    def stop(self) -> None:
        if self.pipeline_profile is not None:
            self.pipeline.stop()
            self.pipeline_profile = None
        print("Stopped RealSense capture server")

    def read_request(self) -> dict[str, Any] | None:
        if not REQUEST_PATH.exists():
            return None

        try:
            with open(REQUEST_PATH, "r", encoding="utf-8") as f:
                request = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not read request.json yet: {e}")
            return None

        if not isinstance(request, dict):
            print("Ignoring malformed request.json: root is not an object")
            return None

        return request

    def capture_frame(self) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + (CAPTURE_TIMEOUT_MS / 1000.0)

        while time.monotonic() < deadline:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)

            if self.align is not None:
                frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()
            return color, depth

        raise TimeoutError(f"Timed out waiting for frames after {CAPTURE_TIMEOUT_MS} ms")

    def write_response(self, request_id: int, color: np.ndarray, depth: np.ndarray) -> None:
        timestamp_utc = utc_now_iso()

        meta = MetaPayload(
            request_id=request_id,
            timestamp_utc=timestamp_utc,
            device_name=self.device_name,
            device_serial=self.device_serial,
            depth_scale_m_per_unit=self.depth_scale,
            color_shape=list(color.shape),
            depth_shape=list(depth.shape),
            color_dtype=str(color.dtype),
            depth_dtype=str(depth.dtype),
            aligned_to_color=ALIGN_DEPTH_TO_COLOR,
        )

        safe_unlink(READY_PATH)

        atomic_write_png(COLOR_PATH, color)
        atomic_write_npy(DEPTH_PATH, depth)
        atomic_write_json(META_PATH, asdict(meta))

        ready = ReadyPayload(
            request_id=request_id,
            status="ok",
            timestamp_utc=timestamp_utc,
            color_path=COLOR_PATH.name,
            depth_path=DEPTH_PATH.name,
            meta_path=META_PATH.name,
        )
        atomic_write_json(READY_PATH, asdict(ready))

    def write_error(self, request_id: int, message: str) -> None:
        payload = {
            "request_id": request_id,
            "status": "error",
            "timestamp_utc": utc_now_iso(),
            "message": message,
        }
        safe_unlink(READY_PATH)
        atomic_write_json(READY_PATH, payload)

    def handle_request(self, request: dict[str, Any]) -> None:
        capture = bool(request.get("capture", False))
        request_id = request.get("request_id")

        if not isinstance(request_id, int):
            print("Ignoring request without integer request_id")
            return

        if not capture:
            return

        if request_id <= self.last_request_id:
            return

        print(f"Handling capture request_id={request_id}")

        try:
            color, depth = self.capture_frame()
            self.write_response(request_id, color, depth)
            self.last_request_id = request_id
            print(f"Completed capture request_id={request_id}")
        except Exception as e:
            self.write_error(request_id, str(e))
            self.last_request_id = request_id
            print(f"Capture failed for request_id={request_id}: {e}")

    def serve_forever(self) -> None:
        while self.running:
            request = self.read_request()
            if request is not None:
                self.handle_request(request)
            time.sleep(POLL_INTERVAL_SEC)


def main() -> int:
    server = RealSenseCaptureServer()

    def _handle_signal(signum: int, frame: Any) -> None:
        print(f"Received signal {signum}, shutting down")
        server.running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        server.start()
        server.serve_forever()
        return 0
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            server.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
