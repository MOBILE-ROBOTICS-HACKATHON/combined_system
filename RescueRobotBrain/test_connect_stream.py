#!/usr/bin/env python3
"""Quick probe to test reading frames from http://10.164.99.122:8080/"""
import cv2
import time
import urllib.request
import numpy as np

url = "http://10.164.99.122:8080/"
print(f"Connecting to video stream: {url} ...")

# Method 1: Try cv2.VideoCapture
cap = cv2.VideoCapture(url)
if cap.isOpened():
    ret, frame = cap.read()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        print(f"[SUCCESS] cv2.VideoCapture read frame: {w}x{h} px")
        cap.release()
        exit(0)
    cap.release()

print("[Notice] cv2.VideoCapture did not read frame directly, trying HTTP multipart stream reader...")

# Method 2: HTTP multipart reader
try:
    stream = urllib.request.urlopen(url, timeout=5)
    bytes_data = b""
    for _ in range(50):
        bytes_data += stream.read(4096)
        a = bytes_data.find(b"\xff\xd8")  # JPEG start
        b = bytes_data.find(b"\xff\xd9")  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                h, w = frame.shape[:2]
                print(f"[SUCCESS] HTTP MJPEG stream decoded frame: {w}x{h} px")
                exit(0)
    print("[Error] Could not find valid JPEG frame in stream")
except Exception as e:
    print(f"[Error] Stream connection failed: {e}")
