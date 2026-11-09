from pathlib import Path
import time
import numpy as np
from .geometry import angle

SIDES = {'left': (23, 25, 27), 'right': (24, 26, 28)}


class PoseEstimator:
    def __init__(self, model_path):
        if not Path(model_path).is_file():
            raise ValueError(f'Model missing: {model_path}. Run python scripts/download_model.py')
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path),
                                             delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.inference_s = 0.0

    def detect(self, rgb, timestamp_ms):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        start = time.perf_counter()
        result = self.detector.detect_for_video(image, timestamp_ms)
        self.inference_s += time.perf_counter() - start
        return [[dict(x=p.x, y=p.y, z=p.z, visibility=p.visibility,
                      presence=p.presence) for p in pose] for pose in result.pose_landmarks]

    def close(self):
        self.detector.close()


def measure(poses, side, width, height):
    quality = dict(visibility=0.0, presence=0.0, segment_px=0.0, in_frame=False)
    if not poses:
        return None, quality
    joints = [poses[0][i] for i in SIDES[side]]
    points = np.array([[p['x'] * width, p['y'] * height] for p in joints])
    quality = dict(visibility=min(p['visibility'] for p in joints),
                   presence=min(p['presence'] for p in joints),
                   segment_px=float(min(np.linalg.norm(points[0]-points[1]),
                                        np.linalg.norm(points[2]-points[1]))),
                   in_frame=all(0 <= p['x'] <= 1 and 0 <= p['y'] <= 1 for p in joints))
    return angle(*points), quality
