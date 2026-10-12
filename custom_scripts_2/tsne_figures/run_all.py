"""
Run every figure script in this folder.

Usage (from the EMBER2024 root):
    python custom_scripts_2/tsne_figures/run_all.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["fig01_family_grid.py", "fig02_malware_vs_benign.py", "fig03_04_family_subsets.py"]

for name in SCRIPTS:
    print(f"== {name}")
    subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=HERE)
