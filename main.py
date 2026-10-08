import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
from ultralytics import YOLO

from field import detect_field_lines
from tracking import BallTracker, BallTrail, CameraMotion, ShotDetector


ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "data/clip_41-25_42-25.mp4"
OUTPUT = ROOT / "runs" / datetime.now().strftime("field_%Y%m%d_%H%M%S")
WEIGHTS = ROOT / "weights/yolo11m.pt"
# Alternative: football-specific weights or a second detector for tiny balls.
# For this test task, keep one YOLO11m model and evaluate its limitations.

DEVICE = 0
IMAGE_SIZE = 1280
CANDIDATE_CONF = 0.05
ACCEPT_CONF = 0.25
CONFIDENCE_MARGIN = 0.05
SOURCE_OFFSET = 41 * 60 + 25
MAX_FRAMES = None
SHOW_CANDIDATES = False
TRAIL_SECONDS = 3.0

# Fixed broadcast graphics in this video; normalized image coordinates.
IGNORE_REGIONS = {
    "scoreboard": (0.040, 0.045, 0.285, 0.105),
    "logo": (0.915, 0.880, 0.952, 0.945),
}


def detect(model, image, class_id):
    result = model.predict(
        image, classes=[class_id], imgsz=IMAGE_SIZE,
        conf=CANDIDATE_CONF, iou=0.5, device=DEVICE,
        half=DEVICE != "cpu", verbose=False,
    )[0]
    return result.boxes.data[:, :5].cpu().numpy()


def ignored_region(box, width, height):
    x = (box[0] + box[2]) / (2 * width)
    y = (box[1] + box[3]) / (2 * height)

    for name, (x1, y1, x2, y2) in IGNORE_REGIONS.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name

    return ""


def draw_box(image, box, color, thickness=1):
    x1, y1, x2, y2 = map(round, box[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def main():
    if not VIDEO.is_file():
        raise FileNotFoundError(VIDEO)

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    ball_class = next(k for k, v in model.names.items() if v == "sports ball")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    tracker = BallTracker(ACCEPT_CONF, CONFIDENCE_MARGIN)
    shots = ShotDetector()
    camera = CameraMotion()
    trail = BallTrail(TRAIL_SECONDS)
    shot_id = 0

    with av.open(str(VIDEO)) as source, \
            av.open(str(OUTPUT / "annotated.mp4"), "w", options={"movflags": "+faststart"}) as output, \
            (OUTPUT / "ball.csv").open("w", newline="", encoding="utf-8") as ball_file, \
            (OUTPUT / "candidates.csv").open("w", newline="", encoding="utf-8") as candidates_file, \
            (OUTPUT / "field_lines.csv").open("w", newline="", encoding="utf-8") as field_file:
        stream = source.streams.video[0]
        encoded = output.add_stream("libx264", rate=stream.average_rate)
        encoded.width = stream.width + stream.width % 2
        encoded.height = stream.height + stream.height % 2
        encoded.pix_fmt = "yuv420p"
        encoded.time_base = encoded.codec_context.time_base = stream.time_base
        encoded.options = {"crf": "18", "preset": "fast"}

        balls, candidates = csv.writer(ball_file), csv.writer(candidates_file)
        balls.writerow([
            "frame", "timestamp", "source_timestamp", "x", "y", "confidence", "status",
            "predicted_x", "predicted_y", "shot_id", "cut",
        ])
        candidates.writerow(["frame", "timestamp", "x1", "y1", "x2", "y2", "confidence", "ignored_region"])
        field_lines = csv.writer(field_file)
        field_lines.writerow([
            "frame", "timestamp", "shot_id", "segment", "x1", "y1", "x2", "y2", "white_support", "status",
        ])
        first_time = None

        for index, frame in enumerate(source.decode(stream)):
            if MAX_FRAMES is not None and index >= MAX_FRAMES:
                break

            if frame.pts is None:
                raise ValueError(f"Frame {index} has no timestamp")

            frame_time = frame.pts * frame.time_base

            if first_time is None:
                first_time = frame_time

            elapsed = frame_time - first_time
            timestamp = float(elapsed)
            image = frame.to_ndarray(format="bgr24")
            height, width = image.shape[:2]

            cut = shots.update(image)

            if cut:
                shot_id += 1
                tracker.reset()
                print(f"Cut: frame {index}, t={timestamp:.2f}s")

            motion = camera.update(image, cut)
            boxes = detect(model, image, ball_class)
            visualization = image.copy()
            lines = detect_field_lines(image, IGNORE_REGIONS)

            for segment, (line, support) in enumerate(lines):
                coordinates = [round(float(value), 3) for value in line]
                field_lines.writerow([index, f"{timestamp:.6f}", shot_id, segment,
                                      *coordinates, round(support, 3), "detected"])
                x1, y1, x2, y2 = map(round, line)
                cv2.line(visualization, (x1, y1), (x2, y2), (220, 120, 220), 2, cv2.LINE_AA)

            if not lines:
                field_lines.writerow([index, f"{timestamp:.6f}", shot_id, "", *[""] * 5, "missing"])

            keep = []

            for box in boxes:
                ignored = ignored_region(box, width, height)
                candidates.writerow([index, f"{timestamp:.6f}", *map(float, box), ignored])
                keep.append(not ignored)

                if SHOW_CANDIDATES and not ignored:
                    draw_box(visualization, box, (0, 165, 255))

            boxes = boxes[np.array(keep, dtype=bool)]
            box, status, prediction = tracker.update(boxes, timestamp, width, height, motion)
            x = y = confidence = predicted_x = predicted_y = ""
            point = tracker.center(box) if box is not None else None
            color = (0, 165, 255) if status == "detected_low_confidence" else (60, 230, 60)
            trail.update(point, timestamp, motion, color, cut or status == "detected_reacquired")
            trail.draw(visualization, timestamp)

            if box is not None:
                x, y = map(float, point)
                confidence = float(box[4])
                cv2.circle(visualization, (round(x), round(y)), 12, color, 2)
                x, y = round(x, 3), round(y, 3)

            if prediction is not None:
                predicted_x, predicted_y = map(float, prediction)
                point = (round(predicted_x), round(predicted_y))
                cv2.drawMarker(visualization, point, (255, 180, 50), cv2.MARKER_CROSS, 22, 2)
                predicted_x, predicted_y = round(predicted_x, 3), round(predicted_y, 3)

            balls.writerow([index, f"{timestamp:.6f}", f"{SOURCE_OFFSET + timestamp:.6f}",
                            x, y, confidence, status, predicted_x, predicted_y, shot_id, int(cut)])
            counts[status] += 1

            label = f"{index} | {timestamp:.2f}s | shot {shot_id} | {status} | lines {len(lines)}"

            if cut:
                label += " | CUT"

            cv2.rectangle(visualization, (0, 0), (850, 32), (20, 20, 20), -1)
            cv2.putText(visualization, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

            visualization = cv2.copyMakeBorder(visualization, 0, height % 2, 0, width % 2,
                                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
            video_frame = av.VideoFrame.from_ndarray(visualization, format="bgr24")
            video_frame.time_base = stream.time_base
            video_frame.pts = round(elapsed / stream.time_base)

            for packet in encoded.encode(video_frame):
                output.mux(packet)

            if (index + 1) % 100 == 0:
                print(f"{index + 1} frames: {dict(counts)}")

        for packet in encoded.encode():
            output.mux(packet)

    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
