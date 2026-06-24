#!/usr/bin/env python3
"""
UMAP projection of frozen encoder features for video samples.

Extracts latent representations directly from the pretrained foundation model
(bypassing probes), computes UMAP, and plots coloured by diagnosis/class.

Usage:
    python evals/video_classification_frozen/umap_projection.py \\
        --fname configs/eval/miracle-demo/miracle_kuk_mr_400_probe_single_step.yaml

Config override (same as main.py):
    --checkpoint /path/to/other.pt
    --batch_size 8
    --folder /path/to/output

UMAP overrides:
    --umap-n-neighbors 15
    --umap-min-dist 0.05
    --umap-metric euclidean
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import yaml
from tqdm import tqdm

# -- local imports (same as eval.py) --
from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    encoder: torch.nn.Module,
    data_loader,
    device: torch.device,
    use_bfloat16: bool = True,
) -> dict:
    """
    Iterate over data_loader and return:
        {
            "embeddings":  np.ndarray  [N, D],
            "labels":      np.ndarray  [N],
            "patient_ids": list[str | None],
        }

    For each video:
      - Encoder outputs List[Tensor[B, N_tokens, D]].
      - We mean-pool across tokens (dim=1), then average across views.
      - Final is a single D-dimensional vector per video.
    """
    all_embeddings = []
    all_labels = []
    all_patient_ids = []

    # Buffer for multi-clip videos: index → list of per-clip pooled tensors
    clip_buffer: dict[int, list[torch.Tensor]] = defaultdict(list)
    label_buffer: dict[int, int] = {}
    pid_buffer:   dict[int, str | None] = {}

    global_idx = 0

    from torch.amp import autocast

    for data in tqdm(data_loader, desc="Extracting features", unit="batch"):

        clips = [
            [dij.to(device, non_blocking=True) for dij in di]
            for di in data[0]
        ]
        clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
        labels = data[1]
        batch_size = len(labels)
        patient_ids = data[3] if len(data) > 3 else [None] * batch_size

        with autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            outputs = encoder(clips, clip_indices)

        # outputs: List[Tensor[B, N, D]] — mean-pool tokens, then average views
        pooled = sum(o.mean(dim=1) for o in outputs) / max(len(outputs), 1)  # [B, D]

        for i in range(batch_size):
            clip_buffer[global_idx].append(pooled[i].cpu())
            label_buffer[global_idx] = labels[i].item()
            pid_buffer[global_idx] = patient_ids[i]
            global_idx += 1

    # Average across clips per video
    for idx in sorted(clip_buffer.keys()):
        stacked = torch.stack(clip_buffer[idx])  # [num_clips, D]
        emb = stacked.mean(dim=0)
        all_embeddings.append(emb.numpy())
        all_labels.append(label_buffer[idx])
        all_patient_ids.append(pid_buffer[idx])

    logger.info(f"Extracted {len(all_embeddings)} video embeddings "
                f"(dim={all_embeddings[0].shape[0]})")
    return {
        "embeddings":  np.stack(all_embeddings, axis=0),
        "labels":      np.array(all_labels),
        "patient_ids": all_patient_ids,
    }


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def make_dataloader(
    csv_path: str,
    resolution: int,
    frames_per_clip: int,
    frame_step: int,
    num_segments: int,
    num_views: int,
    batch_size: int,
    num_workers: int,
    training: bool = False,
):
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
        normalize=DEFAULT_NORMALIZATION,
    )

    loader, _ = init_data(
        data="VideoDataset",
        root_path=[csv_path],
        transform=transform,
        batch_size=batch_size,
        world_size=1,
        rank=0,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=None,
        num_clips=num_segments,
        allow_clip_overlap=True,
        num_workers=num_workers,
        drop_last=False,
    )
    return loader


# ---------------------------------------------------------------------------
# UMAP + Plot
# ---------------------------------------------------------------------------

def compute_and_plot(
    all_features: dict[str, dict],
    class_names: list[str],
    output_dir: str,
    umap_kwargs: dict | None = None,
):
    """
    all_features:  {set_name:  {"embeddings": [N,D], "labels": [N]}, ...}
    class_names:   label-index → human name
    """
    if umap_kwargs is None:
        umap_kwargs = dict(n_neighbors=30, min_dist=0.1, n_components=2,
                           metric="cosine", random_state=42)

    set_labels = sorted(all_features.keys())

    # -- collect
    all_emb = []
    all_lbl = []
    all_set = []
    for sname in set_labels:
        feats = all_features[sname]
        all_emb.append(feats["embeddings"])
        all_lbl.append(feats["labels"])
        all_set.append(np.full(len(feats["labels"]), sname, dtype=object))

    X = np.concatenate(all_emb, axis=0)
    y = np.concatenate(all_lbl, axis=0)
    sets = np.concatenate(all_set, axis=0)

    logger.info(f"Total samples for UMAP: {len(X)} (dim={X.shape[1]})")
    logger.info(f"UMAP kwargs: {umap_kwargs}")

    # -- run UMAP
    reducer = umap.UMAP(**umap_kwargs)
    embedding_2d = reducer.fit_transform(X)

    unique = np.unique(y)
    n_classes = len(unique)

    # Custom class colours (fall back to tab20 for extra classes)
    custom_colours = ["#27AE60", "#F1C40F", "#E67E22", "#E74C3C"]
    cmap = matplotlib.colormaps.get_cmap("tab20")
    colours = [
        custom_colours[i] if i < len(custom_colours) else cmap(i % 20)
        for i in range(n_classes)
    ]

    # -------- Figure 1: two panels (class + split) --------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    set_markers = {"train": "o", "val": "s", "test": "^"}

    # Panel 1: colour by class, marker by set
    ax = axes[0]
    for lbl in unique:
        for sname in set_labels:
            mask = (y == lbl) & (sets == sname)
            if not mask.any():
                continue
            name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
            ax.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                c=[colours[lbl]], marker=set_markers.get(sname, "o"),
                s=25, alpha=0.7, edgecolors="none",
                label=f"{name} ({sname})",
            )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12, markerscale=1.5)

    # Panel 2: colour by set
    ax = axes[1]
    set_colours = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    for sname in set_labels:
        mask = sets == sname
        if not mask.any():
            continue
        ax.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=set_colours.get(sname, "gray"), marker="o",
            s=25, alpha=0.7, edgecolors="none", label=sname,
        )
    ax.legend(fontsize=12)

    plt.tight_layout()

    path1 = os.path.join(output_dir, "umap_projection.png")
    fig.savefig(path1, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {path1}")
    plt.close(fig)

    # -------- Figure 2: class-only --------
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    for lbl in unique:
        mask = y == lbl
        if not mask.any():
            continue
        name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
        ax2.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=[colours[lbl]], s=50, alpha=0.9, edgecolors="none", label=name,
        )
    ax2.legend(fontsize=14)
    plt.tight_layout()

    path2 = os.path.join(output_dir, "umap_projection_classes.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {path2}")
    plt.close(fig2)

    # -------- Save raw data --------
    np.savez(
        os.path.join(output_dir, "umap_data.npz"),
        embedding_2d=embedding_2d,
        labels=y,
        sets=sets,
        class_names=np.array(class_names[:n_classes]),
    )
    logger.info(f"Saved umap_data.npz to {output_dir}")


# ---------------------------------------------------------------------------
# Config loading (mirrors main.py)
# ---------------------------------------------------------------------------

def load_config(fname: str, overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    for key, val in overrides.items():
        if val is None:
            continue
        if key == "checkpoint":
            params["model_kwargs"]["checkpoint"] = val
        elif key == "model_name":
            params["model_kwargs"]["pretrain_kwargs"]["encoder"]["model_name"] = val
        elif key == "batch_size":
            params.setdefault("experiment", {}).setdefault("optimization", {})["batch_size"] = val
        elif key == "folder":
            params["folder"] = val

    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="UMAP projection of frozen encoder features")

    # -- Config file (same interface as main.py)
    p.add_argument("--fname", type=str, required=True,
                   help="Path to YAML config (same format as eval probe configs)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override model checkpoint path")
    p.add_argument("--model_name", type=str, default=None,
                   help="Override encoder model name")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size")
    p.add_argument("--folder", type=str, default=None,
                   help="Override output folder")

    # -- UMAP overrides
    p.add_argument("--umap-n-neighbors", type=int, default=None,
                   help="Override UMAP n_neighbors")
    p.add_argument("--umap-min-dist", type=float, default=None,
                   help="Override UMAP min_dist")
    p.add_argument("--umap-metric", type=str, default=None,
                   help="Override UMAP metric (e.g. cosine, euclidean)")

    # -- Device
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-bfloat16", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # -- Load config
    overrides = {
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "folder": args.folder,
    }
    cfg = load_config(args.fname, overrides)

    # -- Resolve sections
    pretrain = cfg.get("model_kwargs", {})
    exp = cfg.get("experiment", {})
    data_cfg = exp.get("data", {})
    opt_cfg = exp.get("optimization", {})
    umap_cfg = cfg.get("umap", {})

    # --- Encoder params ---
    module_name = pretrain.get("module_name",
        "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip")
    checkpoint = pretrain.get("checkpoint")
    pretrain_kwargs = pretrain.get("pretrain_kwargs", {})
    wrapper_kwargs = pretrain.get("wrapper_kwargs", {})

    # --- Data params ---
    resolution = data_cfg.get("resolution", 224)
    frames_per_clip = data_cfg.get("frames_per_clip", 16)
    frame_step = data_cfg.get("frame_step", 4)
    num_segments = data_cfg.get("num_segments", 1)
    num_views = data_cfg.get("num_views_per_segment", 1)
    num_classes = data_cfg.get("num_classes", 2)
    batch_size = opt_cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 8)
    use_bfloat16 = opt_cfg.get("use_bfloat16", True)

    # --- UMAP params (from config, optionally overridden by CLI) ---
    class_names = umap_cfg.get("class_names",
                               [f"Class {i}" for i in range(num_classes)])
    set_paths = umap_cfg.get("sets", {})
    if not set_paths:
        # if no umap.sets key, fall back to experiment.data val set
        val_path = data_cfg.get("dataset_val")
        if val_path:
            set_paths = {"val": val_path}
        else:
            logger.error("No datasets specified. Add a 'umap.sets' dict to your config.")
            sys.exit(1)

    output_dir = cfg.get("folder", "./umap_output")
    os.makedirs(output_dir, exist_ok=True)

    umap_kwargs = {
        "n_neighbors":  args.umap_n_neighbors or umap_cfg.get("n_neighbors", 30),
        "min_dist":     args.umap_min_dist    or umap_cfg.get("min_dist", 0.1),
        "metric":       args.umap_metric      or umap_cfg.get("metric", "cosine"),
        "n_components": 2,
        "random_state": 42,
    }

    # -- Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.fname}")
    logger.info(f"UMAP sets: {list(set_paths.keys())}")
    logger.info(f"UMAP kwargs: {umap_kwargs}")

    # -- Load encoder (exactly as eval.py does)
    logger.info("Loading frozen encoder ...")
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        device=device,
    )
    encoder.eval()
    logger.info(f"Encoder loaded. embed_dim = {encoder.embed_dim}")

    # -- Extract features for each set
    all_features = {}
    for sname, csv_path in set_paths.items():
        logger.info(f"Loading {sname}  from  {csv_path}")
        loader = make_dataloader(
            csv_path=csv_path,
            resolution=resolution,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_segments=num_segments,
            num_views=num_views,
            batch_size=batch_size,
            num_workers=num_workers,
            training=False,
        )
        feats = extract_features(encoder, loader, device, use_bfloat16=use_bfloat16)
        all_features[sname] = feats
        logger.info(f"  {sname}: {feats['embeddings'].shape[0]} samples, "
                    f"{len(np.unique(feats['labels']))} unique classes")

    # -- UMAP + plot
    compute_and_plot(all_features, class_names, output_dir, umap_kwargs)
    logger.info("Done.")


if __name__ == "__main__":
    main()
