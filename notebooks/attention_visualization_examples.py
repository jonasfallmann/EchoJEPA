"""
Example Usage Scenarios for Attention Visualization Script

This file shows different ways to use visualize_attention_per_frame.py
"""

# ============================================================================
# Example 1: Visualize the first video in test set
# ============================================================================
"""
python notebooks/visualize_attention_per_frame.py \
    --eval_config configs/eval/vitl/config.yaml \
    --probe_checkpoint results/video_classification_frozen/best.pt \
    --test_csv data/test_videos.csv \
    --output_dir ./attention_viz
"""

# ============================================================================
# Example 2: Visualize a specific video (index 5)
# ============================================================================
"""
python notebooks/visualize_attention_per_frame.py \
    --eval_config configs/eval/vitl/config.yaml \
    --probe_checkpoint results/video_classification_frozen/best.pt \
    --test_csv data/test_videos.csv \
    --video_idx 5 \
    --output_video patient_005_attention.mp4 \
    --output_dir ./visualizations
"""

# ============================================================================
# Example 3: Using larger model (ViT-L)
# ============================================================================
"""
python notebooks/visualize_attention_per_frame.py \
    --eval_config configs/eval/vitl/config.yaml \
    --probe_checkpoint models/vitl_probe_best.pt \
    --test_csv data/test_videos.csv \
    --video_idx 0 \
    --gpu_id 0 \
    --output_dir ./large_model_viz
"""

# ============================================================================
# Example 4: Using different GPU (GPU 1)
# ============================================================================
"""
python notebooks/visualize_attention_per_frame.py \
    --eval_config configs/eval/vitg-384/config.yaml \
    --probe_checkpoint results/vitg384_probe.pt \
    --test_csv data/test_videos.csv \
    --video_idx 0 \
    --gpu_id 1 \
    --output_dir ./viz_gpu1
"""

# ============================================================================
# Example 5: Batch visualization script - Process multiple videos
# ============================================================================
"""
Create file: batch_visualize.py

#!/usr/bin/env python

import subprocess
import pandas as pd
import sys
from pathlib import Path

def batch_visualize(eval_config, probe_checkpoint, test_csv, num_videos=5):
    '''Process first N videos from test set'''
    
    output_base = Path("./batch_attention_viz")
    output_base.mkdir(exist_ok=True)
    
    # Read CSV to get number of videos
    df = pd.read_csv(test_csv, sep=None, engine='python', header=None)
    num_videos = min(num_videos, len(df))
    
    print(f"Processing {num_videos} videos...")
    
    for idx in range(num_videos):
        print(f"\n[{idx+1}/{num_videos}] Processing video {idx}...")
        
        # Create output subdirectory for each video
        output_dir = output_base / f"video_{idx:03d}"
        
        cmd = [
            "python", "notebooks/visualize_attention_per_frame.py",
            "--eval_config", eval_config,
            "--probe_checkpoint", probe_checkpoint,
            "--test_csv", test_csv,
            "--video_idx", str(idx),
            "--output_video", f"attention_viz_{idx:03d}.mp4",
            "--output_dir", str(output_dir),
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            print(f"❌ Failed to process video {idx}")
            continue
        
        print(f"✓ Completed video {idx}")
    
    print(f"\nAll visualizations saved to {output_base}")

if __name__ == "__main__":
    # Adjust these paths for your setup
    batch_visualize(
        eval_config="configs/eval/vitl/config.yaml",
        probe_checkpoint="results/video_classification_frozen/best.pt",
        test_csv="data/test_videos.csv",
        num_videos=10
    )

# Run with: python batch_visualize.py
"""

# ============================================================================
# Example 6: Python script (no command line) - Direct API usage
# ============================================================================
"""
Create file: direct_api_usage.py

import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the visualization module
from notebooks.visualize_attention_per_frame import (
    load_single_video,
    load_model_and_probe,
    get_video_frames,
    process_frames_for_inference,
    extract_attention_per_frame,
    create_visualization_with_attention_bar,
)
import yaml
import torch

def main():
    # Configuration
    eval_config_path = "configs/eval/vitl/config.yaml"
    probe_checkpoint_path = "results/video_classification_frozen/best.pt"
    test_csv_path = "data/test_videos.csv"
    video_idx = 0
    output_dir = Path("./attention_viz")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(eval_config_path, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    # Load model
    print("Loading model...")
    encoder, classifier, config_dict = load_model_and_probe(
        eval_config, probe_checkpoint_path, device
    )
    
    # Load video
    print("Loading video...")
    video_path, label, patient_id = load_single_video(test_csv_path, video_idx)
    
    args_data = config_dict.get("experiment").get("data")
    resolution = args_data.get("resolution", 224)
    frames, fps, original_size = get_video_frames(video_path, resolution=resolution)
    
    # Prepare clips
    print("Preparing clips...")
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    clips, frame_indices = process_frames_for_inference(
        frames, frames_per_clip=frames_per_clip, frame_step=frame_step
    )
    
    # Extract attention
    print("Extracting attention...")
    attention_scores = extract_attention_per_frame(
        encoder, classifier, frames, clips, frame_indices, device
    )
    
    # Create visualization
    print("Creating visualization...")
    output_video = output_dir / f"attention_viz_{video_idx}.mp4"
    create_visualization_with_attention_bar(
        frames, attention_scores, frame_indices, str(output_video),
        fps=fps, bar_height=40, bar_position='top'
    )
    
    print(f"✓ Done! Saved to {output_video}")

if __name__ == "__main__":
    main()
"""

# ============================================================================
# Example 7: Advanced - Custom attention analysis
# ============================================================================
"""
Create file: analyze_attention_patterns.py

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.visualize_attention_per_frame import (
    load_single_video, load_model_and_probe, get_video_frames,
    process_frames_for_inference, extract_attention_per_frame,
)
import yaml
import torch

def analyze_attention(eval_config_path, probe_checkpoint_path, test_csv_path, video_idx):
    '''Extract and analyze attention patterns'''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(eval_config_path, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    encoder, classifier, config_dict = load_model_and_probe(
        eval_config, probe_checkpoint_path, device
    )
    
    video_path, label, patient_id = load_single_video(test_csv_path, video_idx)
    args_data = config_dict.get("experiment").get("data")
    resolution = args_data.get("resolution", 224)
    frames, fps, _ = get_video_frames(video_path, resolution=resolution)
    
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    clips, frame_indices = process_frames_for_inference(
        frames, frames_per_clip=frames_per_clip, frame_step=frame_step
    )
    
    attention_scores = extract_attention_per_frame(
        encoder, classifier, frames, clips, frame_indices, device
    )
    
    # Analysis
    all_scores = np.concatenate(attention_scores)
    
    print(f"\\n=== Attention Analysis for Video {video_idx} ===")
    print(f"True Label: {label}")
    print(f"Patient ID: {patient_id}")
    print(f"Video Duration: {len(frames) / fps:.1f}s ({len(frames)} frames @ {fps} FPS)")
    print(f"\\n--- Attention Statistics ---")
    print(f"Min attention: {all_scores.min():.4f}")
    print(f"Max attention: {all_scores.max():.4f}")
    print(f"Mean attention: {all_scores.mean():.4f}")
    print(f"Std attention: {all_scores.std():.4f}")
    print(f"Median attention: {np.median(all_scores):.4f}")
    
    # Find most attended frames
    for clip_idx, scores in enumerate(attention_scores):
        top_3 = np.argsort(scores)[-3:][::-1]
        print(f"\\nClip {clip_idx} - Top 3 attended tokens:")
        for rank, token_idx in enumerate(top_3, 1):
            print(f"  {rank}. Token {token_idx}: {scores[token_idx]:.4f}")
    
    # Plot attention distribution
    plt.figure(figsize=(12, 4))
    plt.hist(all_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Attention Score')
    plt.ylabel('Frequency')
    plt.title(f'Attention Score Distribution - Video {video_idx}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'attention_distribution_video_{video_idx}.png', dpi=100)
    print(f"\\nSaved attention distribution plot to attention_distribution_video_{video_idx}.png")

# Run with:
# python analyze_attention_patterns.py
#
# Or call directly:
# analyze_attention("configs/eval/vitl/config.yaml", 
#                   "results/probe.pt", 
#                   "data/test.csv", 
#                   video_idx=0)
"""

# ============================================================================
# Example 8: Processing videos from different datasets
# ============================================================================
"""
# For a different test set:
python notebooks/visualize_attention_per_frame.py \
    --eval_config configs/eval/miracle-demo/config.yaml \
    --probe_checkpoint results/miracle_demo_probe.pt \
    --test_csv /data/different_dataset/test.csv \
    --video_idx 0 \
    --output_dir ./external_dataset_viz

# Or with custom model config:
python notebooks/visualize_attention_per_frame.py \
    --eval_config configs/eval/vitg-384/config.yaml \
    --probe_checkpoint models/vitg384_model.pt \
    --test_csv data/challenge_set.csv \
    --video_idx 3 \
    --output_dir ./challenge_set_viz \
    --output_video results_vitg384.mp4
"""

print(__doc__)

