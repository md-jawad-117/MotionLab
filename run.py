"""Launch MotionLab from a fresh checkout without installing the source package."""
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    from motionlab.gui import main
    main()
