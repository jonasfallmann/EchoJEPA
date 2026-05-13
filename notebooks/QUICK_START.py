#!/usr/bin/env python3
"""
QUICK START GUIDE - Attention Visualization

This is the simplest way to get started with visualizing attention scores.

1. Modify the configuration below
2. Run: python notebooks/QUICK_START.py
"""

import subprocess
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION - Modify These Values For Your Setup
# ============================================================================

# Path to your evaluation config (find in configs/eval/)
EVAL_CONFIG = "configs/eval/vitl/config.yaml"

# Path to your trained probe checkpoint
PROBE_CHECKPOINT = "results/video_classification_frozen/best.pt"

# Path to your test dataset CSV
TEST_CSV = "data/test_videos.csv"

# Which video to visualize (0 = first video in CSV)
VIDEO_INDEX = 0

# Where to save the output
OUTPUT_DIR = "./attention_visualizations"

# Name of the output video file
OUTPUT_VIDEO = "attention_viz.mp4"

# Which GPU to use (0 = first GPU, -1 = CPU)
GPU_ID = 0

# ============================================================================
# RUN THE VISUALIZATION
# ============================================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║   Per-Frame Attention Visualization - Quick Start             ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # Verify files exist
    print("Step 1: Checking files...")

    missing_files = []

    if not Path(EVAL_CONFIG).exists():
        missing_files.append(f"  ❌ Eval config not found: {EVAL_CONFIG}")
    else:
        print(f"  ✓ Eval config: {EVAL_CONFIG}")

    if not Path(PROBE_CHECKPOINT).exists():
        missing_files.append(f"  ❌ Probe checkpoint not found: {PROBE_CHECKPOINT}")
    else:
        print(f"  ✓ Probe checkpoint: {PROBE_CHECKPOINT}")

    if not Path(TEST_CSV).exists():
        missing_files.append(f"  ❌ Test CSV not found: {TEST_CSV}")
    else:
        print(f"  ✓ Test CSV: {TEST_CSV}")

    if missing_files:
        print("\n" + "\n".join(missing_files))
        print("\n⚠️  ERROR: File not found. Please update the configuration at the top of this script.")
        return 1

    # Create output directory
    print(f"\nStep 2: Creating output directory...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output directory: {OUTPUT_DIR}")

    # Build command
    print(f"\nStep 3: Running visualization...")
    print(f"  Video index: {VIDEO_INDEX}")
    print(f"  GPU: {GPU_ID}")

    cmd = [
        sys.executable,
        "notebooks/visualize_attention_per_frame.py",
        "--eval_config", EVAL_CONFIG,
        "--probe_checkpoint", PROBE_CHECKPOINT,
        "--test_csv", TEST_CSV,
        "--video_idx", str(VIDEO_INDEX),
        "--output_video", OUTPUT_VIDEO,
        "--output_dir", OUTPUT_DIR,
        "--gpu_id", str(GPU_ID),
    ]

    print("\nRunning command:")
    print("  " + " \\\n    ".join(cmd))
    print()

    # Execute
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║                      ✓ SUCCESS!                               ║
╚════════════════════════════════════════════════════════════════╝

Output saved to: {OUTPUT_DIR}

Files created:
  • {OUTPUT_VIDEO} - Video with attention visualization
  • metadata.yaml - Metadata about the visualization

Next steps:
  1. Open the video file to see the attention overlay
  2. Check the metadata.yaml for details about the video
  3. To visualize another video, modify VIDEO_INDEX and run again
  4. For more advanced options, see ATTENTION_VISUALIZATION_README.md

        """)
        return 0
    else:
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║                      ✗ FAILED                                 ║
╚════════════════════════════════════════════════════════════════╝

The visualization script encountered an error.

Troubleshooting:
  1. Verify all file paths are correct
  2. Check that your GPU has enough memory
  3. Ensure all dependencies are installed:
     pip install torch torchvision opencv-python matplotlib
  4. For detailed error messages, check the console output above

For more help, see: ATTENTION_VISUALIZATION_README.md
        """)
        return 1


if __name__ == "__main__":
    sys.exit(main())

