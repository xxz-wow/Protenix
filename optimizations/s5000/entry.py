"""Compatibility CLI; optimization implementations now live in model source."""
import sys
from pathlib import Path
if __name__ == "__main__":
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print("Use: python optimizations/s5000/launch.py --help")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from runner.inference import run
        run()
