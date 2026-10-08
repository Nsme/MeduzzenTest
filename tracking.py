from collections import deque

import cv2
import numpy as np


class ShotDetector:
    def __init__(self):
        self.previous_image = None
        self.previous_histogram = None

    def update(self, image):
        small = cv2.resize(image, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram)

        cut = False

        if self.previous_image is not None:
            difference = np.mean(cv2.absdiff(small, self.previous_image))
            histogram_distance = cv2.compareHist(
                histogram, self.previous_histogram, cv2.HISTCMP_BHATTACHARYYA,
            )

            # Combining both signals avoids treating a fast pan as a cut here.
            # These thresholds are tuned to this clip, not arbitrary broadcasts.
            cut = difference > 30 and histogram_distance > 0.30

        self.previous_image = small
        self.previous_histogram = histogram

        return bool(cut)


class CameraMotion:
    def __init__(self):
        self.previous = None

    def update(self, image, cut=False):
        height, width = image.shape[:2]
        small = cv2.resize(image, (640, round(height * 640 / width)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        previous, self.previous = self.previous, gray

        if previous is None or cut:
            return None

        points = cv2.goodFeaturesToTrack(previous, 200, 0.01, 12)

        if points is None or len(points) < 12:
            return None

        moved, valid, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray, points, None, winSize=(21, 21), maxLevel=3,
        )
        before, after = points[valid.ravel() == 1], moved[valid.ravel() == 1]

        if len(before) < 12:
            return None

        # Background features usually dominate; RANSAC rejects moving players.
        # This affine approximation is motion compensation, not field calibration.
        motion, inliers = cv2.estimateAffinePartial2D(
            before, after, method=cv2.RANSAC, ransacReprojThreshold=2,
        )

        if motion is None or inliers.mean() < 0.5:
            return None

        scale = np.hypot(motion[0, 0], motion[0, 1])

        if not 0.8 < scale < 1.25:
            return None

        motion[:, 2] *= width / 640

        return motion


class BallTrail:
    def __init__(self, seconds=3.0):
        self.seconds = seconds
        self.points = deque()

    def update(self, point, timestamp, motion, color, reset=False):
        # Never join different shots, reacquisitions, or gaps with a solid line.
        if reset or point is None or motion is None:
            self.points.clear()

        elif self.points:
            # Reproject old image points to the current camera view. This is an
            # approximate visual trail, not a trajectory in field coordinates.
            self.points = deque(
                (time, motion[:, :2] @ position + motion[:, 2], previous_color)
                for time, position, previous_color in self.points
            )

        while self.points and timestamp - self.points[0][0] > self.seconds:
            self.points.popleft()

        if point is not None:
            self.points.append((timestamp, np.asarray(point, dtype=float), color))

    def draw(self, image, timestamp):
        points = list(self.points)

        for previous, current in zip(points, points[1:]):
            age = timestamp - current[0]
            fade = 1 - 0.7 * age / self.seconds
            color = tuple(round(channel * fade) for channel in current[2])
            start = tuple(np.rint(previous[1]).astype(int))
            end = tuple(np.rint(current[1]).astype(int))
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)


class BallTracker:
    def __init__(self, accept_conf=0.25, confidence_margin=0.05, max_gap=0.28):
        self.accept_conf = accept_conf
        self.confidence_margin = confidence_margin
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.filter = None
        self.previous_time = None
        self.last_detection_time = None
        self.hits = 0
        self.pending = None
        self.pending_hits = 0

    @staticmethod
    def center(box):
        return (box[:2] + box[2:4]) / 2

    def initialize(self, box, timestamp, diagonal):
        self.filter = cv2.KalmanFilter(4, 2, 0, cv2.CV_64F)
        self.filter.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        x, y = self.center(box)
        self.filter.statePost = np.array([[x], [y], [0], [0]], dtype=np.float64)
        self.filter.errorCovPost = np.diag([
            self.measurement_sigma ** 2, self.measurement_sigma ** 2,
            diagonal ** 2, diagonal ** 2,
        ])

        self.previous_time = self.last_detection_time = timestamp
        self.hits = 1

    def update(self, boxes, timestamp, width, height, motion=None):
        diagonal = np.hypot(width, height)
        self.measurement_sigma = max(6.0, diagonal * 0.008)

        # Expire before association: a stale trajectory must not choose a new ball.
        if self.filter is not None and timestamp - self.last_detection_time > self.max_gap + 1e-6:
            self.reset()

        if self.filter is None:
            if len(boxes) == 0:
                return None, "missing", None

            ordered = boxes[np.argsort(-boxes[:, 4])]

            if ordered[0, 4] < self.accept_conf:
                return None, "low_confidence", None

            if len(ordered) > 1 and ordered[0, 4] - ordered[1, 4] < self.confidence_margin:
                return None, "ambiguous", None

            self.initialize(ordered[0], timestamp, diagonal)

            return ordered[0], "detected", None

        dt = timestamp - self.previous_time

        if dt <= 0:
            raise ValueError("Tracking requires strictly increasing timestamps")

        self.previous_time = timestamp

        if motion is not None:
            transform = np.zeros((4, 4))
            transform[:2, :2] = transform[2:, 2:] = motion[:, :2]
            state = transform @ self.filter.statePost
            state[:2, 0] += motion[:, 2]
            self.filter.statePost = state
            self.filter.errorCovPost = transform @ self.filter.errorCovPost @ transform.T

        self.filter.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        acceleration = np.array([
            [dt ** 2 / 2, 0],
            [0, dt ** 2 / 2],
            [dt, 0],
            [0, dt],
        ])
        self.filter.processNoiseCov = (diagonal * 2) ** 2 * acceleration @ acceleration.T
        self.filter.measurementNoiseCov = np.eye(2) * self.measurement_sigma ** 2

        prediction = self.filter.predict()[:2, 0]
        covariance = self.filter.errorCovPre[:2, :2] + self.filter.measurementNoiseCov
        gap = timestamp - self.last_detection_time
        matches = []

        for box in boxes:
            offset = self.center(box) - prediction
            distance = np.linalg.norm(offset)
            mahalanobis = float(offset @ np.linalg.solve(covariance, offset))
            weak = box[4] < self.accept_conf

            # A weak detection needs an established track and a tighter gate.
            if weak and (box[4] < 0.10 or self.hits < 2):
                continue

            limit = diagonal * ((0.025 + 0.15 * gap) if weak else (0.07 + 0.35 * gap))

            if distance <= limit and mahalanobis <= (9 if weak else 16):
                score = mahalanobis + 2 * (1 - float(box[4]))
                matches.append((score, box))

        matches.sort(key=lambda match: match[0])
        status = "rejected_motion" if len(boxes) else "missing"

        # Close scores mean that motion did not resolve the ambiguity.
        if len(matches) > 1 and matches[1][0] - matches[0][0] < 0.20:
            status = "ambiguous"

        elif matches:
            box = matches[0][1]
            measurement = self.center(box).astype(np.float64).reshape(2, 1)
            self.filter.correct(measurement)
            self.last_detection_time = timestamp
            self.hits += 1
            self.pending = None
            self.pending_hits = 0
            status = "detected" if box[4] >= self.accept_conf else "detected_low_confidence"

            # Export the measured box center, not the smoothed Kalman state.
            return box, status, None

        # A kick can invalidate constant velocity. Reinitialize only after three
        # consecutive strong, mutually consistent detections outside the gate.
        ordered = boxes[np.argsort(-boxes[:, 4])]

        if not matches and len(ordered) and ordered[0, 4] >= max(0.40, self.accept_conf):
            unique = len(ordered) == 1 or ordered[0, 4] - ordered[1, 4] >= self.confidence_margin
            center = self.center(ordered[0])
            consistent = self.pending is not None and np.linalg.norm(center - self.pending) < diagonal * 0.08
            self.pending_hits = (self.pending_hits + 1 if consistent else 1) if unique else 0
            self.pending = center

            if self.pending_hits >= 3:
                self.reset()
                self.initialize(ordered[0], timestamp, diagonal)

                return ordered[0], "detected_reacquired", None

        else:
            self.pending = None
            self.pending_hits = 0

        if self.hits >= 2 and 0 <= prediction[0] < width and 0 <= prediction[1] < height:
            return None, "predicted", prediction.copy()

        return None, status, None
