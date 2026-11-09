from pathlib import Path
import math
import cv2


def frame_bounds(fps, start=0, end=None, start_frame=None, end_frame=None):
    """Half-open bounds; snap sub-frame floating-point noise at second boundaries."""
    if start_frame is not None or end_frame is not None:
        if start != 0 or end is not None:
            raise ValueError('Use either second bounds or frame bounds, not both')
        first = 0 if start_frame is None else start_frame
        last = end_frame
        if first < 0 or (last is not None and last <= first):
            raise ValueError('Require 0 <= start-frame < end-frame (exclusive)')
    else:
        if not math.isfinite(start) or start < 0 or (end is not None and
                (not math.isfinite(end) or end <= start)):
            raise ValueError('Require finite 0 <= start < end')
        first = math.ceil(start * fps - 1e-7)
        last = None if end is None else math.ceil(end * fps - 1e-7)
        if last is not None and last <= first:
            raise ValueError('Selected time interval contains no source frames')
    return first, last


class VideoReader:
    def __init__(self, path):
        if not Path(path).is_file():
            raise ValueError(f'Video not found: {path}')
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.cap.isOpened() or not math.isfinite(self.fps) or self.fps <= 0:
            self.close()
            raise ValueError('Cannot decode video or read positive FPS. Convert to constant-frame-rate MP4.')
        if self.fps > 1000:
            self.close()
            raise ValueError('FPS above 1000 is unsupported by millisecond timestamps.')

    def frames(self, start_frame=0, end_frame=None):
        """Seek to a half-open source range; fall back to exact sequential grabs."""
        if start_frame < 0 or (end_frame is not None and end_frame <= start_frame):
            raise ValueError('Invalid frame interval')
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != start_frame:
            sought = self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            position = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not sought or abs(position-start_frame) > 0.01:
                if not self.cap.set(cv2.CAP_PROP_POS_FRAMES,0) or abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) > 0.01:
                    raise ValueError('Decoder cannot seek reliably; extract the desired clip first')
                for _ in range(start_frame):
                    if not self.cap.grab():
                        return
        index = start_frame
        while end_frame is None or index < end_frame:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield index, index / self.fps, frame
            index += 1

    def close(self):
        self.cap.release()


def writer(path, fps, width, height):
    output = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not output.isOpened():
        output.release()
        raise ValueError('MP4 encoder unavailable. Check OpenCV installation and output path.')
    return output
