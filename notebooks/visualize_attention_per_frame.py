"""
Attention Visualization Script for Video Classification

This script:
1. Loads a specific video from a test dataset CSV
2. Passes it through a pretrained encoder and attentive probe
3. Extracts per-frame attention scores from the attentive pooler
4. Creates a new video with attention intensity visualization (bar overlay)
5. Saves the output video

Usage:
    python visualize_attention_per_frame.py \
        --eval_config path/to/eval_config.yaml \
        --probe_checkpoint path/to/probe.pt \
        --test_csv path/to/test.csv \
        --video_idx 0 \
        --output_video output_with_attention.mp4 \
        --batch_size 1 \
        --gpu_id 0
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.video_classification_frozen.models import init_module
from src.models.attentive_pooler import AttentiveClassifier
from src.models.linear_pooler import LinearClassifier, MLPClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class AttentionExtractor:
    """Helper class to extract attention weights from AttentiveClassifier"""

    def __init__(self):
        self.attention_weights = None
        self.hook_handles = []

    def hook_fn(self, module, input, output):
        """Hook to capture attention weights from CrossAttention"""
        q, x = input[0], input[1]

        # Manually compute attention weights (replicating forward pass)
        B, Q, C = q.shape
        num_heads = module.num_heads
        head_dim = C // num_heads
        scale = head_dim ** -0.5

        # Project q, k, v
        q_proj = module.q(q).reshape(B, Q, num_heads, head_dim).permute(0, 2, 1, 3)
        B, N, C = x.shape
        kv = module.kv(x).reshape(B, N, 2, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Compute attention
        attn = (q_proj @ k.transpose(-2, -1)) * scale  # [B, num_heads, Q, N]
        attn = F.softmax(attn, dim=-1)

        # Store attention weights (average across heads for visualization)
        # attn: [B, num_heads, Q, N] -> we care about N (attention to each token)
        self.attention_weights = attn.detach().cpu()

    def register_hooks(self, pooler):
        """Register hooks on the cross-attention module"""
        if hasattr(pooler, 'cross_attention_block'):
            if hasattr(pooler.cross_attention_block, 'xattn'):
                handle = pooler.cross_attention_block.xattn.register_forward_hook(self.hook_fn)
                self.hook_handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def get_attention_scores(self):
        """Get averaged attention scores across heads"""
        if self.attention_weights is None:
            return None

        attn = self.attention_weights  # [B, num_heads, Q, N]
        # Average across heads and queries (we have Q=1 for pooler)
        attn = attn.mean(dim=(0, 1, 2))  # [N] - attention to each token
        return attn.numpy()


def load_single_video(csv_path, video_idx):
    """Load a single video info from CSV"""
    df = pd.read_csv(csv_path, sep=None, engine='python', header=None)

    if video_idx >= len(df):
        raise ValueError(f"Video index {video_idx} out of range (CSV has {len(df)} videos)")

    row = df.iloc[video_idx]
    video_path = row[0]
    label = int(row[1])
    patient_id = row[2] if len(row) > 2 else None

    logger.info(f"Loaded video info: {video_path} | Label: {label} | Patient: {patient_id}")

    return video_path, label, patient_id


def load_model_and_probe(config_dict, probe_checkpoint_path, device):
    """Load pretrained encoder and probe classifier"""
    args_pretrain = config_dict.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = config_dict.get("experiment")
    args_classifier = args_exp.get("classifier")
    args_data = args_exp.get("data")

    # Load encoder
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=args_data.get("frames_per_clip", 16),
        resolution=args_data.get("resolution", 224),
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # Load probe checkpoint
    if not os.path.exists(probe_checkpoint_path):
        raise FileNotFoundError(f"Probe checkpoint not found: {probe_checkpoint_path}")

    checkpoint_dict = robust_checkpoint_loader(probe_checkpoint_path, map_location=device)
    num_classifier_heads = len(checkpoint_dict.get("classifiers", [1]))

    # Determine best head
    best_head_idx = 0
    if "best_val_acc_per_head" in checkpoint_dict:
        best_val_accs = checkpoint_dict["best_val_acc_per_head"]
        best_head_idx = int(np.argmax(best_val_accs))

    # Create classifier
    num_classes = args_data.get("num_classes")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    probe_type = args_classifier.get("probe_type", "attentive")
    use_layernorm = args_classifier.get("use_layernorm", True)
    probe_dropout = args_classifier.get("dropout", 0.0)

    if probe_type == "linear":
        classifier = LinearClassifier(
            embed_dim=encoder.embed_dim,
            num_classes=num_classes,
            use_layernorm=use_layernorm,
            dropout=probe_dropout,
        ).to(device)
    elif probe_type == "mlp":
        classifier = MLPClassifier(
            embed_dim=encoder.embed_dim,
            num_classes=num_classes,
            use_layernorm=use_layernorm,
            dropout=probe_dropout,
        ).to(device)
    else:  # attentive
        classifier = AttentiveClassifier(
            embed_dim=encoder.embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_classes=num_classes,
            use_activation_checkpointing=True,
        ).to(device)

    classifier.load_state_dict(checkpoint_dict["classifiers"][best_head_idx])

    logger.info(f"Loaded {probe_type} classifier from checkpoint (head {best_head_idx})")

    return encoder, classifier, config_dict


def get_video_frames(video_path, resolution=224):
    """Load video frames using OpenCV"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video info: {total_frames} frames @ {fps} FPS, {width}x{height}")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, (resolution, resolution))

        frames.append(frame)
        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return np.array(frames), fps, (width, height)


def process_frames_for_inference(frames, frames_per_clip=16, frame_step=4):
    """Prepare frames for inference (create clips)
    
    Returns actual frame indices used, not padded indices
    """
    num_frames = len(frames)
    clips = []
    frame_indices = []

    # Extract clips with step - only use actual frames, no padding
    start_idx = 0
    while start_idx + frames_per_clip * frame_step <= num_frames:
        clip_indices = np.arange(start_idx, start_idx + frames_per_clip * frame_step, frame_step)
        clip = frames[clip_indices]
        clips.append(clip)
        frame_indices.append(clip_indices)
        start_idx += frames_per_clip * frame_step

    if len(clips) == 0:
        logger.warning(f"Not enough frames for a full clip. Need {frames_per_clip * frame_step}, got {num_frames}")
        # Use what we have, but don't pad
        clip_indices = np.arange(min(frames_per_clip, num_frames))
        clip = frames[clip_indices]
        clips = [clip]
        frame_indices = [clip_indices]
    
    logger.info(f"Created {len(clips)} clips from {num_frames} frames")
    logger.info(f"Frames covered by clips: {list(range(frame_indices[0][0], frame_indices[-1][-1] + 1))}")
    if num_frames > frame_indices[-1][-1] + 1:
        uncovered = list(range(frame_indices[-1][-1] + 1, num_frames))
        logger.warning(f"⚠️  {len(uncovered)} frames at the end are NOT covered by any clip: {uncovered}")

    return clips, frame_indices


def normalize_frames(frames, normalization=DEFAULT_NORMALIZATION):
    """Normalize frames to ImageNet stats"""
    mean = np.array(normalization[0]).reshape(1, 1, 1, 3)
    std = np.array(normalization[1]).reshape(1, 1, 1, 3)

    # Convert to float and normalize to [0, 1]
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - mean) / std

    return torch.from_numpy(frames.transpose(0, 3, 1, 2)).float()


def extract_attention_per_frame(
    encoder,
    classifier,
    frames,
    clips,
    frame_indices,
    device
):
    """Extract per-frame attention scores
    
    Returns:
        all_attention_scores: List of attention arrays per clip
        valid_frame_range: (min_frame_idx, max_frame_idx) of frames with valid attention
    """
    encoder.eval()
    classifier.eval()

    # Set up attention extractor
    extractor = AttentionExtractor()
    if hasattr(classifier, 'pooler'):
        extractor.register_hooks(classifier.pooler)

    all_attention_scores = []

    logger.info(f"Extracting attention from {len(clips)} clips")
    
    with torch.no_grad():
        for clip_idx, clip in enumerate(clips):
            logger.info(f"Processing clip {clip_idx}/{len(clips)}")
            logger.info(f"  Clip shape: {clip.shape} (frames={len(clip)})")
            logger.info(f"  Frame indices in this clip: {frame_indices[clip_idx]}")
            
            # Normalize clip: [num_frames, H, W, 3] -> [num_frames, 3, H, W]
            clip_tensor = normalize_frames(clip, DEFAULT_NORMALIZATION)

            # Reshape to [B, C, F, H, W] = [1, 3, num_frames, H, W]
            clip_tensor = clip_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)

            logger.info(f"  Tensor shape for encoder: {clip_tensor.shape}")

            # Encoder expects: clips = list of (list of view tensors)
            clip_idx_tensor = torch.arange(len(frame_indices[clip_idx]), device=device)
            
            # Get encoder output
            encoder_output = encoder([[clip_tensor]], [clip_idx_tensor])

            logger.info(f"  Encoder output type: {type(encoder_output)}, len: {len(encoder_output) if isinstance(encoder_output, list) else 'N/A'}")
            if isinstance(encoder_output, list) and len(encoder_output) > 0:
                logger.info(f"  First element shape: {encoder_output[0].shape if hasattr(encoder_output[0], 'shape') else type(encoder_output[0])}")

            # encoder_output should be a list of outputs (one per view)
            if isinstance(encoder_output, list):
                encoder_output = encoder_output[0]

            # Forward through classifier to trigger hooks
            _ = classifier(encoder_output)

            # Get attention scores
            attn_scores = extractor.get_attention_scores()
            if attn_scores is not None:
                # Check for NaN values
                nan_count = np.isnan(attn_scores).sum()
                if nan_count > 0:
                    logger.warning(f"  ⚠️  Found {nan_count} NaN values in {len(attn_scores)} attention scores!")
                    logger.warning(f"      This clip may be padded or incomplete")
                    # Replace NaN with 0 or skip this clip
                    attn_scores = np.nan_to_num(attn_scores, nan=0.0)
                    logger.warning(f"      Replaced NaN with 0.0")
                
                all_attention_scores.append(attn_scores)
                logger.info(f"  ✓ Extracted {len(attn_scores)} attention scores")
                logger.info(f"    Attention range: [{attn_scores.min():.4f}, {attn_scores.max():.4f}], mean={attn_scores.mean():.4f}")
            else:
                logger.error(f"  ✗ Failed to extract attention scores for clip {clip_idx}")

    extractor.remove_hooks()

    logger.info(f"Successfully extracted attention from {len(all_attention_scores)}/{len(clips)} clips")
    
    # Determine valid frame range (first to last actual frame with attention)
    if all_attention_scores and len(frame_indices) > 0:
        valid_start_frame = int(frame_indices[0][0])
        valid_end_frame = int(frame_indices[-1][-1])
        valid_frame_range = (valid_start_frame, valid_end_frame)
        logger.info(f"Valid frame range for export: {valid_start_frame} to {valid_end_frame}")
    else:
        valid_frame_range = (0, len(frames) - 1)
    
    return all_attention_scores, valid_frame_range


def create_visualization_with_attention_bar(
    frames,
    attention_scores,
    frame_indices,
    output_path,
    valid_frame_range=None,
    fps=30.0,
    bar_height=40,
    bar_position='top',
    border_thickness=8,
    use_red_border=True
):
    """
    Create output video with attention intensity visualization.

    Args:
        frames: Original video frames [num_frames, H, W, 3]
        attention_scores: List of attention vectors per clip
        frame_indices: Indices of frames in each clip
        output_path: Path to save output video
        valid_frame_range: (min_idx, max_idx) - only export frames in this range
        fps: Frames per second
        bar_height: Height of attention bar in pixels
        bar_position: 'top', 'side', or 'border'
        border_thickness: Thickness of attention border
        use_red_border: If True, draw red border; if False, draw colored bar
    """

    # Use valid frame range if provided, otherwise use all frames
    if valid_frame_range is not None:
        min_frame_idx, max_frame_idx = valid_frame_range
        frames_to_export = frames[min_frame_idx:max_frame_idx + 1]
        logger.info(f"Exporting frames {min_frame_idx} to {max_frame_idx} (skipping {min_frame_idx} frames at start, {len(frames) - max_frame_idx - 1} at end)")
    else:
        min_frame_idx = 0
        max_frame_idx = len(frames) - 1
        frames_to_export = frames
        logger.info(f"Exporting all {len(frames)} frames")
    
    num_frames_to_export = len(frames_to_export)
    frame_height, frame_width = frames_to_export[0].shape[:2]

    # Normalize attention scores across all data
    all_attn = np.concatenate(attention_scores) if attention_scores else np.array([])
    if len(all_attn) > 0:
        attn_min = all_attn.min()
        attn_max = all_attn.max()
        if attn_max > attn_min:
            all_attn_normalized = (all_attn - attn_min) / (attn_max - attn_min)
        else:
            all_attn_normalized = np.ones_like(all_attn)
    else:
        all_attn_normalized = np.array([])

    # Create attention mapping: frame index -> attention score
    # Only for frames that are exported
    frame_attention_map = {}
    attn_idx = 0

    logger.info(f"Building attention map:")
    logger.info(f"  Total attention scores extracted: {len(all_attn_normalized)}")
    logger.info(f"  Total frames to export: {num_frames_to_export}")
    logger.info(f"  Number of clips: {len(attention_scores)}")
    
    for clip_idx, (clip_attn, indices) in enumerate(zip(attention_scores, frame_indices)):
        logger.info(f"  Clip {clip_idx}: {len(clip_attn)} tokens, frames {indices}")
        for token_idx, frame_idx in enumerate(indices):
            # Only map frames that are in the valid range
            if min_frame_idx <= frame_idx <= max_frame_idx:
                if attn_idx < len(all_attn_normalized):
                    # Map to position in exported frames (not original)
                    exported_frame_idx = frame_idx - min_frame_idx
                    frame_attention_map[exported_frame_idx] = all_attn_normalized[attn_idx]
                    attn_idx += 1

    logger.info(f"Created attention map for {len(frame_attention_map)} unique frames out of {num_frames_to_export}")

    # Log attention distribution
    if frame_attention_map:
        mapped_scores = list(frame_attention_map.values())
        logger.info(f"Attention score distribution:")
        logger.info(f"  Min: {min(mapped_scores):.4f}, Max: {max(mapped_scores):.4f}, Mean: {np.mean(mapped_scores):.4f}")
        unmapped_frames = [i for i in range(num_frames_to_export) if i not in frame_attention_map]
        if unmapped_frames:
            logger.warning(f"  WARNING: {len(unmapped_frames)} frames have no attention mapping")
            logger.warning(f"  Unmapped frame indices (in exported range): {unmapped_frames[:20]}{'...' if len(unmapped_frames) > 20 else ''}")

    # Setup video writer
    if use_red_border:
        output_height = frame_height
        output_width = frame_width
    elif bar_position == 'top':
        output_height = frame_height + bar_height
        output_width = frame_width
    else:  # side
        output_height = frame_height
        output_width = frame_width + bar_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    # Create colormaps
    cmap_viridis = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap_viridis)

    # Write frames with attention visualization
    for exported_frame_idx in range(num_frames_to_export):
        frame = frames_to_export[exported_frame_idx].copy()

        # Get attention score for this frame
        attn_score = frame_attention_map.get(exported_frame_idx, None)

        if attn_score is None:
            # Frame not in attention map (shouldn't happen with valid_frame_range)
            attn_score = 0.5
            in_map = False
        else:
            in_map = True

        if use_red_border:
            # Divergent Scale: 0.0 (Blue) -> 0.5 (Black) -> 1.0 (Red)
            if attn_score < 0.5:
                # Scale blue from 255 (at 0.0) down to 0 (at 0.5)
                blue_intensity = int(255 * (1.0 - 2.0 * attn_score))
                red_intensity = 0
            else:
                # Scale red from 0 (at 0.5) up to 255 (at 1.0)
                blue_intensity = 0
                red_intensity = int(255 * (2.0 * attn_score - 1.0))

            green_intensity = 0

            # The frame is currently in RGB, so the tuple must be (R, G, B)
            border_color = (red_intensity, green_intensity, blue_intensity)

            output_frame = frame.copy()
            cv2.rectangle(output_frame, (0, 0), (frame_width - 1, frame_height - 1),
                          border_color, border_thickness)

            text = f"Attention: {attn_score:.3f}" + (" [NO MAP]" if not in_map else "")
            text_pos = (12, 35)

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_bg_rect = (text_pos[0] - 5, text_pos[1] - text_size[1] - 5,
                            text_size[0] + 10, text_size[1] + 10)

            overlay = output_frame.copy()
            cv2.rectangle(overlay,
                          (text_bg_rect[0], text_bg_rect[1]),
                          (text_bg_rect[0] + text_bg_rect[2], text_bg_rect[1] + text_bg_rect[3]),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)

            cv2.putText(output_frame, text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        else:
            # ... existing code ...
            color = (sm.to_rgba(attn_score) * 255)[:3][::-1]
            color = tuple(int(c) for c in color)

            if bar_position == 'top':
                output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                bar_width = int(frame_width * attn_score)
                output_frame[0:bar_height, 0:bar_width] = color
                output_frame[bar_height:, :] = frame
                text = f"Attention: {attn_score:.3f}"
                cv2.putText(output_frame, text, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                bar_top = int(frame_height * (1 - attn_score))
                output_frame[bar_top:, frame_width:] = color
                output_frame[:, 0:frame_width] = frame

        output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        writer.write(output_frame_bgr)

    writer.release()
    logger.info(f"Saved visualization to {output_path}")


def main(args):
    """Main pipeline"""

    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load eval config
    with open(args.eval_config, 'r') as f:
        eval_config = yaml.safe_load(f)

    # Load model and probe
    logger.info("Loading model and probe...")
    encoder, classifier, config_dict = load_model_and_probe(
        eval_config,
        args.probe_checkpoint,
        device,
    )

    # Load single video from CSV
    video_path, label, patient_id = load_single_video(args.test_csv, args.video_idx)

    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load video frames
    args_data = config_dict.get("experiment").get("data")
    resolution = args_data.get("resolution", 224)

    logger.info(f"Loading video: {video_path}")
    frames, fps, original_size = get_video_frames(video_path, resolution=resolution)

    # Prepare clips for inference
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)

    clips, frame_indices = process_frames_for_inference(
        frames,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step
    )

    # Extract attention per frame
    logger.info("Extracting attention scores...")
    attention_scores, valid_frame_range = extract_attention_per_frame(
        encoder,
        classifier,
        frames,
        clips,
        frame_indices,
        device
    )

    if not attention_scores:
        logger.error("Failed to extract attention scores")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    output_video = output_dir / args.output_video
    logger.info(f"Creating visualization video with style: {args.viz_style}")

    # Parse visualization style arguments
    use_red_border = (args.viz_style == "red_border")
    bar_position = "top" if args.viz_style == "bar_top" else "side"

    create_visualization_with_attention_bar(
        frames,
        attention_scores,
        frame_indices,
        str(output_video),
        valid_frame_range=valid_frame_range,
        fps=fps,
        bar_height=40,
        bar_position=bar_position,
        border_thickness=args.border_thickness,
        use_red_border=use_red_border
    )

    logger.info(f"✓ Visualization complete!")
    logger.info(f"  Video: {output_video}")
    logger.info(f"  True Label: {label}")
    logger.info(f"  Patient ID: {patient_id}")

    # Save metadata
    metadata = {
        'video_path': video_path,
        'true_label': label,
        'patient_id': patient_id,
        'num_frames': len(frames),
        'fps': fps,
        'original_size': original_size,
        'output_video': str(output_video),
    }

    metadata_file = output_dir / 'metadata.yaml'
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f)

    logger.info(f"Saved metadata to {metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize per-frame attention from probe")

    parser.add_argument(
        "--eval_config",
        required=True,
        help="Path to evaluation config YAML file",
    )
    parser.add_argument(
        "--probe_checkpoint",
        required=True,
        help="Path to trained probe checkpoint (.pt file)",
    )
    parser.add_argument(
        "--test_csv",
        required=True,
        help="Path to test dataset CSV file",
    )
    parser.add_argument(
        "--video_idx",
        type=int,
        default=0,
        help="Index of video in CSV to visualize (default: 0)",
    )
    parser.add_argument(
        "--output_dir",
        default="./attention_visualizations",
        help="Directory to save output videos",
    )
    parser.add_argument(
        "--output_video",
        default="attention_visualization.mp4",
        help="Filename for output video",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--viz_style",
        choices=["red_border", "bar_top", "bar_side"],
        default="red_border",
        help="Visualization style for attention (default: red_border)",
    )
    parser.add_argument(
        "--border_thickness",
        type=int,
        default=8,
        help="Thickness of red border if using red_border style (default: 8)",
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

