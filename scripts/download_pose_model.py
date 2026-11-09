"""Fetch the pinned pretrained MediaPipe Lite pose model from Google storage."""
from pathlib import Path
import hashlib
import urllib.request

URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
SHA256 = '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    target = Path(__file__).resolve().parents[1] / 'assets/pose_landmarker_lite.task'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and digest(target) == SHA256:
        print('Expected pose model already present:', target)
        return
    temporary = target.with_suffix('.task.download')
    try:
        print('Downloading pretrained MediaPipe Lite model...')
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if digest(temporary) != SHA256:
            raise RuntimeError('Downloaded model checksum differs from the pinned asset.')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('Pose model ready:', target)

if __name__ == '__main__':
    main()
