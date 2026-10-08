import cv2
import numpy as np


def detect_field_lines(image, ignore_regions):
    # HSV and contrast thresholds are adapted to this clip's grass and exposure.
    height, width = image.shape[:2]
    scale = width / 960
    small = cv2.resize(image, (960, round(height / scale)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    grass = cv2.inRange(hsv, (30, 35, 35), (95, 255, 255))
    closed = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < small.shape[0] * small.shape[1] * 0.15:
        return []

    pitch = np.zeros_like(grass)
    cv2.drawContours(pitch, [largest], -1, 255, cv2.FILLED)
    white = cv2.inRange(hsv, (0, 0, 110), (179, 100, 255))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    saturation_dip = cv2.morphologyEx(hsv[:, :, 1], cv2.MORPH_BLACKHAT,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    white[(contrast < 10) | (saturation_dip < 15)] = 0
    white = cv2.bitwise_and(white, pitch)

    for x1, y1, x2, y2 in ignore_regions.values():
        cv2.rectangle(white, (round(x1 * small.shape[1]), round(y1 * small.shape[0])),
                      (round(x2 * small.shape[1]), round(y2 * small.shape[0])), 0, -1)

    lines = cv2.HoughLinesP(white, 1, np.pi / 720, threshold=55,
                            minLineLength=90, maxLineGap=8)

    if lines is None:
        return []

    # White shirts and advertising can also form lines. Require white pixels
    # along the segment and grass immediately beside it, not just inside a hull.
    candidates = []

    for line in lines[:, 0]:
        start, end = line[:2].astype(float), line[2:].astype(float)
        direction = end - start
        length = np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]]) / length
        points = np.linspace(start, end, max(20, round(length)))
        white_support = sample_mask(white, points)
        left = sample_mask(grass, points + normal * 6)
        right = sample_mask(grass, points - normal * 6)

        if white_support < 0.75 or max(left, right) < 0.80 or min(left, right) < 0.45:
            continue

        candidates.append((length, line.astype(float), white_support))

    # Keep the longest representative of overlapping, nearly collinear segments.
    # Segment IDs are local to a frame; no semantic line names or calibration.
    selected = []

    for _, line, support in sorted(candidates, key=lambda item: -item[0]):
        if not any(overlaps(line, previous) for previous, _ in selected):
            selected.append((line, support))

    ratios = np.array([width / small.shape[1], height / small.shape[0]] * 2)

    # Support is the fraction of sampled white pixels, not a probability of correctness.
    return [(line * ratios, support) for line, support in selected]


def sample_mask(mask, points):
    rounded = np.rint(points).astype(int)
    valid = ((rounded[:, 0] >= 0) & (rounded[:, 0] < mask.shape[1]) &
             (rounded[:, 1] >= 0) & (rounded[:, 1] < mask.shape[0]))
    values = np.zeros(len(points), dtype=bool)
    values[valid] = mask[rounded[valid, 1], rounded[valid, 0]] > 0

    return float(values.mean())


def overlaps(line, previous):
    start, end = previous[:2], previous[2:]
    direction = (end - start) / np.linalg.norm(end - start)
    other_direction = (line[2:] - line[:2]) / np.linalg.norm(line[2:] - line[:2])

    if abs(direction @ other_direction) < np.cos(np.deg2rad(2)):
        return False

    normal = np.array([-direction[1], direction[0]])
    offsets = (line.reshape(2, 2) - start) @ normal

    if np.max(np.abs(offsets)) > 5:
        return False

    projections = (line.reshape(2, 2) - start) @ direction

    return max(projections) >= -12 and min(projections) <= np.linalg.norm(end - start) + 12
