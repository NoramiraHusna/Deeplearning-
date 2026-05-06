# pothole_severity.py
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PotholeResult:
    mask_area:    float
    true_size:    float
    depth_score:  float
    size_score:   float
    hazard_score: float
    severity:     str
    centroid:     tuple

def get_severity(hazard_score):
    if hazard_score <= 40:   return "MINOR"
    elif hazard_score <= 70: return "MODERATE"
    else:                    return "SEVERE"

SEVERITY_COLORS = {
    "MINOR":    (0, 255, 0),    # green
    "MODERATE": (0, 165, 255),  # orange
    "SEVERE":   (0, 0, 255),    # red
}

def analyze_pothole(mask_xy, frame_gray, frame_h, frame_w):
    """
    Given a polygon mask, compute severity metrics
    mask_xy: numpy array of polygon points (N, 2) in pixel coords
    """
    # ── 1. Render mask to binary image ──────────────────────────
    binary_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    pts = mask_xy.astype(np.int32)
    cv2.fillPoly(binary_mask, [pts], 255)

    # ── 2. Mask area (exact pixel count) ────────────────────────
    mask_area = float(np.sum(binary_mask > 0))

    # ── 3. Centroid ──────────────────────────────────────────────
    M  = cv2.moments(binary_mask)
    cx = int(M["m10"] / M["m00"]) if M["m00"] else frame_w // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] else frame_h // 2

    # ── 4. TrueSize (perspective correction) ────────────────────
    y_coord   = max(cy, 1)  # avoid division by zero
    true_size = mask_area * (frame_h / y_coord) ** 2

    # Normalize to 0-100 (tune max_true_size to your footage)
    max_true_size = 5_000_000
    size_score    = min(true_size / max_true_size, 1.0) * 100

    # ── 5. ShadowMath (depth via contrast inside mask) ──────────
    masked_pixels = frame_gray[binary_mask > 0]
    if len(masked_pixels) > 0:
        stddev     = float(np.std(masked_pixels))
        depth_score = min(stddev / 40.0, 1.0) * 100
    else:
        depth_score = 0.0

    # ── 6. Hazard Score ──────────────────────────────────────────
    hazard_score = (size_score * 0.6) + (depth_score * 0.4)
    severity     = get_severity(hazard_score)

    return PotholeResult(
        mask_area    = mask_area,
        true_size    = true_size,
        depth_score  = depth_score,
        size_score   = size_score,
        hazard_score = hazard_score,
        severity     = severity,
        centroid     = (cx, cy)
    )


def draw_results(frame, mask_xy, result):
    """Draw mask overlay + severity label on frame"""
    color = SEVERITY_COLORS[result.severity]
    pts   = mask_xy.astype(np.int32)

    # Draw filled semi-transparent mask
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Draw mask outline
    cv2.polylines(frame, [pts], True, color, 2)

    # Draw label at centroid
    cx, cy = result.centroid
    label  = f"{result.severity} {result.hazard_score:.0f}/100"
    sub    = f"Size:{result.size_score:.0f} Depth:{result.depth_score:.0f}"

    cv2.putText(frame, label, (cx - 60, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, sub, (cx - 60, cy + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame


def run_pipeline(model_path, source, output_path="output.mp4", conf=0.25):
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    # ── Summary counters ─────────────────────────────────────────
    counts      = {"MINOR": 0, "MODERATE": 0, "SEVERE": 0}
    all_hazards = []
    all_depths  = []
    all_sizes   = []
    frame_idx   = 0

    print(f"Processing {total} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(frame, task="segment", conf=conf, verbose=False)[0]

        if results.masks is not None:
            for mask in results.masks.xy:
                if len(mask) < 3:
                    continue

                result = analyze_pothole(mask, gray, height, width)
                frame  = draw_results(frame, mask, result)

                counts[result.severity] += 1
                all_hazards.append(result.hazard_score)
                all_depths.append(result.depth_score)
                all_sizes.append(result.size_score)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total} frames done")

    cap.release()
    writer.release()

    # ── Print summary ────────────────────────────────────────────
    total_detections = sum(counts.values())
    print("\n" + "="*50)
    print("  POTHOLE DETECTION SYSTEM: RUN SUMMARY")
    print("="*50)
    print(f"  Frames Processed: {frame_idx}")
    print(f"\n  Detection Analytics (Total: {total_detections}):")
    print(f"  - MINOR    Hazards: {counts['MINOR']}")
    print(f"  - MODERATE Hazards: {counts['MODERATE']}")
    print(f"  - SEVERE   Hazards: {counts['SEVERE']}")
    if all_hazards:
        print(f"\n  Scores (Max / Avg):")
        print(f"  - Hazard : {max(all_hazards):.0f} / {np.mean(all_hazards):.0f}")
        print(f"  - Depth  : {max(all_depths):.0f}  / {np.mean(all_depths):.0f}")
        print(f"  - Size   : {max(all_sizes):.0f}  / {np.mean(all_sizes):.0f}")
    print("="*50)
    print(f"\n  Output saved to: {output_path}")


if __name__ == "__main__":
    run_pipeline(
        model_path  = "C:\\Users\\ahmad\\Desktop\\fypee\\pothole\\best_seg.pt",
        source      = "C:\\Users\\ahmad\\Desktop\\fypee\\pothole\\kaggle.mp4",
        output_path = "C:\\Users\\ahmad\\Desktop\\fypee\\pothole\\pothole_severity_output.mp4",
        conf        = 0.6
    )