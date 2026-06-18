#!/usr/bin/env python3
"""Run bnot_data_gen.py for all ICONS-TIMES-V2 num_sites values.

After each run, the output subfolders (target, timestamps) are renamed
with a _BNOT_<NUM_SITES> postfix. source/ is left untouched.
"""

import subprocess
import sys
from pathlib import Path

# -- Configuration -------------------------------------------------------------

SCRIPT_PATH = Path(__file__).parent / "bnot_data_gen.py"

DATA_PATH = Path(
    "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion"
    "/experiments/outputs/icons_results_runtimes"
)

# ICONS - TIMES - V2  (actual site counts; comments show NxN equivalent)
NUM_SITES = [
    256,    # 16
    576,    # 24
    1024,   # 32
    1600,   # 40
    2304,   # 48
    3136,   # 56
    4096,   # 64
    5184,   # 72
    6400,   # 80
    7744,   # 88
    9216,   # 96
    10816,  # 104
    12544,  # 112
]

# Subfolders produced by each run that should be renamed after completion
# source/ is left untouched
OUTPUT_SUBDIRS = ["target", "timestamps"]

# -- Main ----------------------------------------------------------------------

def main():
    if not SCRIPT_PATH.exists():
        print(f"ERROR: Script not found at {SCRIPT_PATH}")
        sys.exit(1)

    print(f"Running {len(NUM_SITES)} num_sites values: {NUM_SITES}")
    print(f"Script: {SCRIPT_PATH}\n")

    for num_sites in NUM_SITES:
        print(f"\n{'='*70}")
        print(f"  NUM_SITES = {num_sites}")
        print(f"{'='*70}\n")

        cmd = [sys.executable, str(SCRIPT_PATH), "--num_sites", str(num_sites)]
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nWarning: num_sites={num_sites} exited with code {result.returncode}")

        for subdir in OUTPUT_SUBDIRS:
            src = DATA_PATH / subdir
            dst = DATA_PATH / f"{subdir}_BNOT_{num_sites}"
            if src.exists():
                src.rename(dst)
                print(f"[rename] {subdir}  ->  {dst.name}")

    print(f"\n{'='*70}")
    print("All num_sites runs complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
