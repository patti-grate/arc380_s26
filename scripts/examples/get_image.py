import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SHARED_DIR = Path("/realsense_shared")
REQUEST_PATH = SHARED_DIR / "request.json"
READY_PATH = SHARED_DIR / "ready.json"
COLOR_PATH = SHARED_DIR / "color.png"
DEPTH_PATH = SHARED_DIR / "depth.npy"
META_PATH = SHARED_DIR / "meta.json"

POLL_INTERVAL_SEC = 0.1
TIMEOUT_SEC = 10.0


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
    tmp_path.replace(path)


def get_next_request_id() -> int:
    if READY_PATH.exists():
        try:
            ready = read_json(READY_PATH)
            if isinstance(ready.get("request_id"), int):
                return ready["request_id"] + 1
        except Exception:
            pass

    if REQUEST_PATH.exists():
        try:
            req = read_json(REQUEST_PATH)
            if isinstance(req.get("request_id"), int):
                return req["request_id"] + 1
        except Exception:
            pass

    return 1


def request_capture(timeout_sec: float = TIMEOUT_SEC) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    request_id = get_next_request_id()

    if READY_PATH.exists():
        READY_PATH.unlink()

    request_payload = {
        "request_id": request_id,
        "capture": True,
    }
    atomic_write_json(REQUEST_PATH, request_payload)

    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        if READY_PATH.exists():
            try:
                ready = read_json(READY_PATH)
            except Exception:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            if ready.get("request_id") != request_id:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            status = ready.get("status")
            if status != "ok":
                raise RuntimeError(f"Capture failed: {ready}")

            if not COLOR_PATH.exists():
                raise FileNotFoundError(f"Missing file: {COLOR_PATH}")
            if not DEPTH_PATH.exists():
                raise FileNotFoundError(f"Missing file: {DEPTH_PATH}")
            if not META_PATH.exists():
                raise FileNotFoundError(f"Missing file: {META_PATH}")

            color = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
            if color is None:
                raise RuntimeError(f"Failed to load color image from {COLOR_PATH}")

            depth = np.load(str(DEPTH_PATH))
            meta = read_json(META_PATH)

            return color, depth, meta

        time.sleep(POLL_INTERVAL_SEC)

    raise TimeoutError(f"Timed out waiting for capture response after {timeout_sec} seconds")


def main() -> int:
    color, depth, meta = request_capture()

    print("Capture succeeded")
    print(f"Timestamp: {meta.get('timestamp_utc')}")
    print(f"Color shape: {color.shape}, dtype={color.dtype}")
    print(f"Depth shape: {depth.shape}, dtype={depth.dtype}")
    print(f"Depth scale: {meta.get('depth_scale_m_per_unit')} m/unit")

    # Example: inspect center pixel depth in meters
    h, w = depth.shape[:2]
    z_raw = int(depth[h // 2, w // 2])
    depth_scale = float(meta["depth_scale_m_per_unit"])
    z_m = z_raw * depth_scale
    print(f"Center depth raw={z_raw}, meters={z_m:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())