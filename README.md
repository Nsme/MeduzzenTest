# Football ball tracking and field lines

Offline ball tracking and field-line extraction on a 60-second broadcast clip: **41:25–42:25 in the source file**, 1920×1080, 25 FPS, 1,500 frames, with camera movement and two shot changes.

## Run

Tested on Windows x64 with Python 3.13 and an NVIDIA GPU. Dependencies pin PyTorch 2.8.0 with CUDA 12.8.

Keep `main.py`, `tracking.py`, `field.py` and `requirements.txt` together. Put the input clip at `data/clip_41-25_42-25.mp4`; the input video is not included in this submission.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

In an existing environment, install requirements and run `python main.py`. Ultralytics downloads `yolo11m.pt` into `weights/` on first use, or you can supply it there. Subsequent processing can run offline.

Settings are constants in `main.py`; there are no CLI arguments. For another clip, update `VIDEO`, `SOURCE_OFFSET` and `IGNORE_REGIONS`. `DEVICE`, `IMAGE_SIZE`, `MAX_FRAMES` and `TRAIL_SECONDS` control processing and display.

## Approach

**Ready-made components:** one pretrained [Ultralytics YOLO11m](https://github.com/ultralytics/ultralytics) detector, OpenCV algorithms, and PyAV for video decoding/encoding. No model was trained or fine-tuned.

**Adaptation:** COCO `sports ball` candidates at `imgsz=1280`, fixed scoreboard/logo exclusions, and Kalman association with state `[x, y, vx, vy]`. Selection considers motion, uncertainty and confidence. Weak candidates need an established track and a tighter gate. Forecasts last at most 0.28 seconds; persistent strong candidates can reinitialize an inconsistent trajectory.

Sparse optical flow estimates camera translation, rotation and zoom. Image/histogram differences trigger shot resets. Field segments use grass masking, local brightness/saturation contrast, Hough detection and support filtering. Thresholds were adjusted to this clip.

## Outputs

Each run creates `runs/field_<date>_<time>/`. The annotated video and CSV files are supplied separately with the submission (`results/` in the accompanying archive).

| File | Contents |
| --- | --- |
| `ball.csv` | One row per frame; measured ball center, detector confidence, status, separate prediction coordinates, shot ID and cut flag. |
| `candidates.csv` | All ball candidates after YOLO postprocessing/NMS, including excluded candidates and exclusion reason. |
| `field_lines.csv` | Segment endpoints per frame; empty coordinates with `missing` when none are found. |
| `annotated.mp4` | H.264 video without audio, with ball markers, a three-second trail and field segments. |

Coordinates are original-image pixels, with origin at the top left; frames start at zero. `timestamp` comes from frame PTS relative to the clip start. `source_timestamp` adds 2,485 seconds, referring to file time, not the broadcast match clock.

Measured `x/y/confidence` stay empty without an accepted detection. Kalman-only positions appear exclusively in `predicted_x/y`; no interpolation is used. Green circles indicate accepted detections, orange circles indicate accepted low-confidence detections, and blue crosses indicate forecasts. The solid trail contains observations only and resets across gaps, cuts, reacquisitions or unavailable camera-motion estimates. Field segments are purple; `white_support` measures pixel support, not detection probability. Segment IDs are local to each frame.

## Validation and limitations

Checked: all 1,500 frames using saved YOLO candidates, plus 50 frames with GPU inference. Frame count, timestamps, CSV consistency, forecast limits and shot resets passed. Field/trail additions left the ball CSV unchanged.

Final counts: 976 accepted detections at confidence ≥0.25, 109 accepted low-confidence detections, 249 forecast-only frames, and 166 frames without either position. Both cuts were found at 23.08 and 26.96 seconds. **These are output counts, not precision/recall:** no manually annotated quantitative evaluation was completed.

Tiny/blurred balls, occlusions and white clutter still cause misses or false tracks. Camera compensation is approximate, and field extraction can miss faint markings or retain false segments. Curves are represented by short straight fragments; semantic field-line labels, calibration, metric pitch coordinates and player tracking are not implemented.

Next steps: annotate a sample spanning all shots, measure localization error and false detections, then consider a football-specific detector/fine-tuning and semantic field keypoints for calibration.

**Estimated effort:** approximately 6 hours, including clip preparation, environment setup, implementation, iterative review and documentation; active time was not logged precisely.
