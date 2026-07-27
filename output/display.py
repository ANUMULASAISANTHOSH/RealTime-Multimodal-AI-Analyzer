"""
output/display.py

Pure drawing module — no windows, no threads.
draw_face(amplitude) returns a single OpenCV BGR frame of the cartoon avatar.
amplitude: 0.0 = mouth closed / resting smile, 1.0 = mouth fully open.
"""
import cv2
import numpy as np

# Canvas size
W, H = 420, 520


def draw_face(amplitude: float = 0.0) -> np.ndarray:
    amplitude = max(0.0, min(1.0, float(amplitude)))

    # ── Background ────────────────────────────────────────────────────────────
    frame = np.full((H, W, 3), (235, 228, 220), dtype=np.uint8)  # warm off-white

    # ── Hair ──────────────────────────────────────────────────────────────────
    cv2.ellipse(frame, (W // 2, 155), (155, 165), 0, 180, 360, (50, 30, 15), -1)
    cv2.rectangle(frame, (95, 155), (325, 230), (50, 30, 15), -1)  # sides

    # ── Face ──────────────────────────────────────────────────────────────────
    cv2.ellipse(frame, (W // 2, 270), (148, 175), 0, 0, 360, (255, 218, 185), -1)
    cv2.ellipse(frame, (W // 2, 270), (148, 175), 0, 0, 360, (210, 170, 140), 2)

    # ── Ears ──────────────────────────────────────────────────────────────────
    for ex, ey in [(62, 270), (358, 270)]:
        cv2.ellipse(frame, (ex, ey), (18, 28), 0, 0, 360, (255, 218, 185), -1)
        cv2.ellipse(frame, (ex, ey), (18, 28), 0, 0, 360, (210, 170, 140), 2)

    # ── Eyebrows ──────────────────────────────────────────────────────────────
    for bx in [148, 272]:
        cv2.ellipse(frame, (bx, 195), (28, 10), 0, 200, 340, (60, 35, 10), 4)

    # ── Eyes ──────────────────────────────────────────────────────────────────
    for ex in [148, 272]:
        cv2.circle(frame, (ex, 225), 26, (255, 255, 255), -1)          # white
        cv2.circle(frame, (ex, 225), 26, (190, 150, 110), 2)           # rim
        cv2.circle(frame, (ex, 226), 14, (90, 60, 30), -1)             # iris
        cv2.circle(frame, (ex, 226), 7,  (25, 20, 18), -1)             # pupil
        cv2.circle(frame, (ex - 5, 221), 4, (255, 255, 255), -1)       # highlight

    # ── Nose ──────────────────────────────────────────────────────────────────
    cv2.line(frame,  (W // 2, 255), (200, 295), (190, 145, 100), 2)
    cv2.line(frame,  (W // 2, 255), (220, 295), (190, 145, 100), 2)
    cv2.ellipse(frame, (W // 2, 297), (14, 7), 0, 0, 180, (190, 145, 100), 2)

    # ── Mouth ─────────────────────────────────────────────────────────────────
    mouth_cx, mouth_cy = W // 2, 345
    mouth_w = 48                          # half-width of mouth
    max_open = 36                         # pixels at amplitude 1.0
    opening = int(amplitude * max_open)

    if opening < 5:
        # Closed — a gentle smile using a polyline
        pts = np.array([
            [mouth_cx - mouth_w, mouth_cy],
            [mouth_cx - 20,      mouth_cy + 12],
            [mouth_cx,           mouth_cy + 16],
            [mouth_cx + 20,      mouth_cy + 12],
            [mouth_cx + mouth_w, mouth_cy],
        ], np.int32)
        cv2.polylines(frame, [pts], False, (180, 90, 90), 3, cv2.LINE_AA)
    else:
        # Open mouth — dark cavity + teeth + lips
        cv2.ellipse(frame, (mouth_cx, mouth_cy), (mouth_w, opening),
                    0, 0, 180, (40, 15, 15), -1)                          # cavity
        # Teeth (upper row)
        teeth_h = min(opening // 2, 18)
        cv2.rectangle(frame,
                      (mouth_cx - mouth_w + 6, mouth_cy),
                      (mouth_cx + mouth_w - 6, mouth_cy + teeth_h),
                      (242, 242, 242), -1)
        # Upper lip
        cv2.ellipse(frame, (mouth_cx, mouth_cy), (mouth_w, opening),
                    0, 180, 360, (190, 90, 90), 3)
        # Lower lip
        cv2.ellipse(frame, (mouth_cx, mouth_cy), (mouth_w, opening),
                    0, 0, 180, (200, 100, 100), 3)

    # ── Status label ──────────────────────────────────────────────────────────
    label = "Speaking..." if amplitude > 0.05 else "Listening..."
    color = (60, 120, 60) if amplitude > 0.05 else (100, 100, 160)
    cv2.putText(frame, label, (W // 2 - 55, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return frame
