"""Rebuild datasets, fit both regressions, and validate the generated results."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    for script in ("build_data.py", "analyse.py", "validate.py"):
        print(f"\nRunning {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], check=True)
    print(f"\nFinished. Open {ROOT / 'output'} for results and charts.")
