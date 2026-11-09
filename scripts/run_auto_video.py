"""Optional command-line entry for the same automatic workflow used by the GUI."""
import argparse
from datetime import datetime
from pathlib import Path
import uuid
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from motionlab.ai_workout import run_auto


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    output=args.output or project/'runs/gui'/('ai_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:5])
    result=run_auto(args.video,output,progress=lambda p,s:print(f'{p}% {s}',flush=True))
    print(result)


if __name__=='__main__':main()
