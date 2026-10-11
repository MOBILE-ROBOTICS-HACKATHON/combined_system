#!/usr/bin/env python3
"""
Generates a realistic sample sky image with a drone for testing single-image processing.
"""
import numpy as np
import cv2

w, h = 1280, 720
# Gradient daylight sky background (light blue to deep sky blue)
sky = np.zeros((h, w, 3), dtype=np.uint8)
for y in range(h):
    r = int(220 - (y / h) * 40)
    g = int(200 - (y / h) * 30)
    b = int(120 - (y / h) * 20)
    sky[y, :] = (b, g, r)

# Draw an 18cm x 18cm quadcopter drone flying overhead at approx 7.5 meters altitude
cx, cy = 680, 260
body_r = 8
arm_span = 24

# Central fuselage
cv2.circle(sky, (cx, cy), body_r, (40, 40, 45), -1)
cv2.circle(sky, (cx, cy), body_r - 2, (60, 60, 65), -1)

# Rotor arms (X-frame configuration for 18cm drone)
cv2.line(sky, (cx - arm_span, cy - arm_span // 2), (cx + arm_span, cy + arm_span // 2), (30, 30, 30), 3)
cv2.line(sky, (cx - arm_span, cy + arm_span // 2), (cx + arm_span, cy - arm_span // 2), (30, 30, 30), 3)

# 4 Motors & Propeller disc blurs
rotor_r = 7
rotors = [
    (cx - arm_span, cy - arm_span // 2),
    (cx + arm_span, cy + arm_span // 2),
    (cx - arm_span, cy + arm_span // 2),
    (cx + arm_span, cy - arm_span // 2),
]

for rx, ry in rotors:
    cv2.circle(sky, (rx, ry), 3, (20, 20, 20), -1)
    prop_overlay = sky.copy()
    cv2.circle(prop_overlay, (rx, ry), rotor_r, (70, 70, 70), -1)
    cv2.addWeighted(prop_overlay, 0.4, sky, 0.6, 0, sky)

cv2.imwrite("sample_drone.jpg", sky)
print("Created sample_drone.jpg for testing.")
