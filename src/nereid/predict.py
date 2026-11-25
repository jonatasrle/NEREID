import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter

from .weights import resolve_weights


def encode_video(ffmpeg, annotated, original, destination):
    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(annotated), "-i", str(original),
               "-map", "0:v:0", "-map", "1:a:0?",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(destination)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])


def predict(weights, source, output, confidence=0.25, image_size=640, device=None):
    from ultralytics import YOLO

    with resolve_weights(weights) as path:
        model = YOLO(str(path))
        if model.task != "detect" or list(model.names.values()) != ["marine_debris"]:
            raise ValueError("Expected the single-class NEREID detection model.")
        if str(source).isdecimal():
            for _ in model.predict(source=int(source), stream=True, show=True, save=False,
                                   conf=confidence, imgsz=image_size, device=device):
                pass
            return None
        return predict_video(model, source, output, confidence, image_size, device)


def predict_video(model, source, output, confidence, image_size, device):
    import cv2
    from tqdm import tqdm

    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("FFmpeg is required for video output. Install it and add it to PATH.")
    if not source.is_file():
        raise ValueError(f"Video not found: {source}")
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot read video: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Cannot determine video frame rate.")
    output.mkdir(parents=True)
    frames, positive_frames, detections = 0, 0, 0
    writer = None
    with tempfile.TemporaryDirectory(prefix="nereid_video_", dir=output) as directory:
        temporary = Path(directory) / "annotated.mp4"
        predictions = model.predict(source=str(source), stream=True, conf=confidence,
                                    imgsz=image_size, device=device, save=False, verbose=False)
        start = perf_counter()
        try:
            with tqdm(total=total if total > 0 else None, unit="frame") as progress:
                for result in predictions:
                    frame = result.plot()
                    size = (frame.shape[1], frame.shape[0])
                    if writer is None:
                        frame_size = size
                        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create annotated video.")
                    if size != frame_size:
                        raise ValueError("Video frame dimensions changed during processing.")
                    writer.write(frame)
                    count = len(result.boxes) if result.boxes is not None else 0
                    frames += 1
                    detections += count
                    positive_frames += int(count > 0)
                    progress.update(1)
        finally:
            if writer is not None:
                writer.release()
            predictions.close()
        if not frames:
            raise ValueError("The video contains no readable frames.")
        elapsed = perf_counter() - start
        destination = output / "detections.mp4"
        encode_video(ffmpeg, temporary, source, destination)
    summary = {"frames": frames, "frames_with_detections": positive_frames,
               "detections_across_frames": detections, "source_fps": fps,
               "processing_fps_before_final_encoding": frames / elapsed,
               "confidence": confidence, "imgsz": image_size}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return destination

