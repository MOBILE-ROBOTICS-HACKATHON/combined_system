import time
import threading
import urllib.request
from typing import Optional, Tuple
import cv2
import numpy as np


class VideoStreamReader:
    """
    Universal High-Performance Video & Stream Ingestion Engine.
    
    Supports:
      1. HTTP / MJPEG Network Streams (e.g. http://10.164.99.122:8080/)
         - Threaded, zero-latency background ingestion
         - Eliminates OpenCV FFmpeg network stalling, dropped sockets & buffer lag
         - Auto-reconnects on network packet jitter
      2. USB / Web Cameras (e.g. device index '0', '1')
      3. Local Video Files (.mp4, .avi, .mkv)
      4. Auto-detects side-by-side Dual Streams (RGB + Depth) and splits them into clean frames.
    """

    def __init__(
        self,
        source: str = "http://10.164.99.122:8080/",
        auto_split_side_by_side: bool = True,
        reconnect_timeout_sec: float = 3.0
    ):
        self.source = source
        self.auto_split = auto_split_side_by_side
        self.reconnect_timeout = reconnect_timeout_sec

        self.is_http: bool = isinstance(self.source, str) and (
            self.source.startswith("http://") or self.source.startswith("https://")
        )

        self.cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()

        self._latest_raw_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0.0
        self._frame_seq: int = 0
        self._last_consumed_seq: int = -1

        self.is_side_by_side: bool = False
        self.raw_width: int = 0
        self.raw_height: int = 0
        self.rgb_width: int = 0
        self.rgb_height: int = 0
        self.fps: float = 30.0

        self._open_stream()

    def _open_stream(self) -> bool:
        """Initializes the capture device or starts the background HTTP worker."""
        if self.is_http:
            self._running = True
            self._thread = threading.Thread(target=self._http_worker, daemon=True)
            self._thread.start()

            # Wait briefly for first frame to arrive to probe dimensions
            t0 = time.time()
            while time.time() - t0 < max(2.5, self.reconnect_timeout):
                with self._lock:
                    if self._latest_raw_frame is not None:
                        self._configure_dimensions(self._latest_raw_frame)
                        return True
                time.sleep(0.05)

            return False
        else:
            if isinstance(self.source, str) and self.source.isdigit():
                cap_source = int(self.source)
            else:
                cap_source = self.source

            self.cap = cv2.VideoCapture(cap_source)
            if not self.cap.isOpened():
                return False

            self.raw_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_val = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps_val if (fps_val > 0 and not np.isnan(fps_val)) else 30.0

            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                self._configure_dimensions(test_frame)
                return True
            return False

    def _configure_dimensions(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        self.raw_width = w
        self.raw_height = h
        if self.auto_split and w >= 1000 and (w / max(h, 1)) >= 2.5:
            self.is_side_by_side = True
            self.rgb_width = w // 2
            self.rgb_height = h
        else:
            self.is_side_by_side = False
            self.rgb_width = w
            self.rgb_height = h

    def _http_worker(self):
        """Dedicated background thread continuously ingesting HTTP MJPEG frames."""
        while self._running:
            stream = None
            try:
                req = urllib.request.Request(
                    self.source,
                    headers={"User-Agent": "RoboHack-VisionStream/1.0"}
                )
                stream = urllib.request.urlopen(req, timeout=self.reconnect_timeout)
                bytes_buf = b""

                while self._running:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    bytes_buf += chunk
                    a = bytes_buf.find(b"\xff\xd8")
                    b = bytes_buf.find(b"\xff\xd9")
                    if a != -1 and b != -1 and b > a:
                        jpg = bytes_buf[a:b+2]
                        bytes_buf = bytes_buf[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._latest_raw_frame = frame
                                self._last_frame_time = time.time()
                                self._frame_seq += 1
            except Exception:
                time.sleep(0.3)
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def is_connected(self) -> bool:
        if self.is_http:
            with self._lock:
                return (self._latest_raw_frame is not None) and (time.time() - self._last_frame_time < 5.0)
        return self.cap is not None and self.cap.isOpened()

    def has_new_frame(self) -> bool:
        """Returns True if a fresh, unconsumed frame has arrived since last read."""
        with self._lock:
            return self._frame_seq > self._last_consumed_seq

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Reads the latest frame from the stream with zero blocking.

        Returns:
            ret (bool): True if frame read was successful.
            rgb_frame (np.ndarray): The RGB color frame (shape H, W, 3).
            depth_frame (np.ndarray or None): The aligned Depth map if dual-stream, else None.
        """
        if self.is_http:
            with self._lock:
                if self._latest_raw_frame is None:
                    return False, None, None
                raw_frame = self._latest_raw_frame.copy()
                self._last_consumed_seq = self._frame_seq
        else:
            if self.cap is None or not self.cap.isOpened():
                if not self._open_stream():
                    return False, None, None
            ret, raw_frame = self.cap.read()
            if not ret or raw_frame is None:
                return False, None, None

        if self.is_side_by_side:
            rgb_frame = raw_frame[:, :self.rgb_width]
            depth_frame = raw_frame[:, self.rgb_width:]
            return True, rgb_frame, depth_frame

        return True, raw_frame, None

    def release(self):
        """Releases all capture resources and terminates worker threads."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
