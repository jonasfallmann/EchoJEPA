#!/usr/bin/env python3
"""
Cross-validation inference script for video classification frozen probes.

Evaluates checkpoints from nested cross-validation (one checkpoint per outer fold)
and exports unified predictions in JSON format with per-fold and aggregate metrics.

Usage:
    python notebooks/crossval_inference.py \\
        --eval_config configs/eval/vitg-384/view/echojepa_view_crossval.yaml \\
        --cv_checkpoint_dir /path/to/video_classification_frozen_crossval/tag/ \\
        --output_dir /path/to/output/ \\
        [--gpu_id 0] \\
        [--batch_size 32]

Output per fold:
    {output_dir}/fold_{k}_predictions.json
    {output_dir}/cv_aggregate_metrics.json
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.video_classification_frozen.eval import make_dataloader, make_probe
from evals.video_classification_frozen.eval_crossval import _write_temp_csv
from evals.video_classification_frozen.models import init_module
from src.utils.checkpoint_loader import robust_checkpoint_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def load_cv_config(eval_config_path):
    """Load and validate the eval config YAML."""
    with open(eval_config_path, "r") as f:
        config = yaml.safe_load(f)
    data = config.get("experiment", {}).get("data", {})
    if not data.get("dataset_crossval"):
        raise ValueError("eval_config must specify dataset_crossval for CV inference.")
    return config


def load_encoder(config_dict, device):
    """Load the pretrained frozen encoder once (shared across all folds)."""
    args_pretrain = config_dict.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")
    args_data = config_dict["experiment"]["data"]

    encoder = init_module(
        module_name=module_name,
        frames_per_clip=args_data.get("frames_per_clip", 16),
        resolution=args_data.get("resolution", 224),
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    logger.info("Loaded encoder from %s", checkpoint)
    return encoder


def load_probe(config_dict, encoder, probe_checkpoint_path, device):
    """Load a single probe from a CV fold checkpoint (reuses the shared encoder)."""
    args_exp = config_dict.get("experiment")
    args_classifier = args_exp.get("classifier")
    args_data = args_exp.get("data")

    if not os.path.exists(probe_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {probe_checkpoint_path}")

    ckpt = robust_checkpoint_loader(probe_checkpoint_path, map_location=device)
    classifier_state_dicts = ckpt.get("classifiers", [])
    if not classifier_state_dicts:
        raise ValueError(f"No classifiers found in checkpoint: {probe_checkpoint_path}")

    opt_grid = ckpt.get("opt_grid", [])
    if opt_grid:
        best_config = opt_grid[0]
        logger.info("Loaded best config from checkpoint: %s", best_config)

    # Build single probe
    num_classes = args_data.get("num_classes")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    probe_type = args_classifier.get("probe_type", "attentive")
    use_layernorm = args_classifier.get("use_layernorm", True)
    probe_dropout = args_classifier.get("dropout", 0.0)
    num_targets = args_classifier.get("num_targets", None)
    task_type = args_classifier.get("task_type", "classification")

    probe = make_probe(
        task=task_type,
        probe=probe_type,
        embed_dim=encoder.embed_dim,
        num_classes=num_classes,
        num_targets=num_targets,
        num_heads=num_heads,
        depth=num_probe_blocks,
        use_ln=use_layernorm,
        dropout=probe_dropout,
    ).to(device)

    probe.load_state_dict(classifier_state_dicts[0])
    probe.eval()

    logger.info("Loaded probe from %s", probe_checkpoint_path)
    return probe


def run_inference(encoder, probe, data_loader, device):
    """Run inference and return list of per-sample prediction dicts."""
    encoder.eval()
    probe.eval()

    predictions = []

    logger.info("Running inference on %d samples...", len(data_loader.dataset))
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            clips = [[dij.to(device) for dij in di] for di in batch[0]]
            labels = batch[1].to(device)
            clip_indices = [d.to(device) for d in batch[2]]
            patient_ids = batch[3] if len(batch) > 3 else [None] * len(labels)

            # Forward through encoder + probe
            outputs = encoder(clips, clip_indices)
            if isinstance(outputs, list):
                per_clip_logits = [probe(o) for o in outputs]
            else:
                per_clip_logits = [probe(outputs)]

            probs_per_clip = [F.softmax(o, dim=1) for o in per_clip_logits]
            avg_probs = sum(probs_per_clip) / len(probs_per_clip)

            batch_size_out = labels.size(0)
            for i in range(batch_size_out):
                pid = patient_ids[i] if i < len(patient_ids) else None
                if isinstance(pid, float) and np.isnan(pid):
                    pid = None

                predictions.append({
                    "subject_id": pid,
                    "video_id": f"batch_{batch_idx}_idx_{i}",
                    "true_label": labels[i].cpu().item(),
                    "probs": avg_probs[i].detach().cpu().numpy().tolist(),
                })

            if (batch_idx + 1) % 10 == 0:
                logger.info("  Processed %d batches", batch_idx + 1)

    return predictions


def compute_metrics(predictions):
    """Compute video-level and subject-level accuracy from prediction list."""
    # Video-level accuracy
    video_correct = sum(
        1 for p in predictions if np.argmax(p["probs"]) == p["true_label"]
    )
    video_acc = 100.0 * video_correct / len(predictions) if predictions else 0.0

    # Subject-level accuracy (majority vote per subject)
    subject_probs = defaultdict(list)
    subject_targets = {}
    for p in predictions:
        sid = p["subject_id"]
        if sid is None:
            continue
        subject_probs[sid].append(np.array(p["probs"]))
        subject_targets[sid] = p["true_label"]

    subj_correct = 0
    subj_total = 0
    for sid, probs_list in subject_probs.items():
        avg_prob = np.mean(probs_list, axis=0)
        pred_class = int(np.argmax(avg_prob))
        if pred_class == subject_targets[sid]:
            subj_correct += 1
        subj_total += 1

    subj_acc = 100.0 * subj_correct / subj_total if subj_total > 0 else 0.0

    return {
        "video_accuracy": float(video_acc),
        "subject_accuracy": float(subj_acc),
        "total_samples": int(len(predictions)),
        "total_subjects": int(subj_total),
    }


def main(args):
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load config
    config = load_cv_config(args.eval_config)
    args_data = config["experiment"]["data"]

    # Load crossval CSV to get per-fold test data
    cv_df = pd.read_csv(args_data["dataset_crossval"])

    all_fold_metrics = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the frozen encoder once (shared across all folds)
    encoder = load_encoder(config, device)

    # Inference per outer fold
    for outer_fold in sorted(cv_df["fold_id"].unique()):
        outer_fold = int(outer_fold)  # Convert numpy.int64 -> Python int for JSON
        logger.info("=" * 60)
        logger.info("Fold %d", outer_fold)
        logger.info("=" * 60)

        # Checkpoint path
        ckpt_path = os.path.join(args.cv_checkpoint_dir, f"fold_{outer_fold}", "last.pt")
        logger.info("Loading checkpoint: %s", ckpt_path)

        if not os.path.exists(ckpt_path):
            logger.warning("Checkpoint not found for fold %d, skipping.", outer_fold)
            continue

        # Load only the probe (encoder is already loaded)
        probe = load_probe(config, encoder, ckpt_path, device)

        # Prepare test data for this fold
        fold_test_df = cv_df[cv_df["fold_id"] == outer_fold]
        import tempfile
        import uuid
        test_csv = os.path.join(
            tempfile.gettempdir(), f"cv_inference_fold_{outer_fold}_{uuid.uuid4().hex[:8]}.csv"
        )
        _df_to_dataset_csv(fold_test_df, test_csv)

        batch_size = args.batch_size or 32
        num_workers = args.num_workers or 8

        test_loader, _ = make_dataloader(
            dataset_type=args_data.get("dataset_type", "VideoDataset"),
            root_path=[test_csv],
            img_size=args_data.get("resolution", 224),
            frames_per_clip=args_data.get("frames_per_clip", 16),
            frame_step=args_data.get("frame_step", 4),
            eval_duration=args_data.get("clip_duration", None),
            num_segments=args_data.get("num_segments", 1),
            num_views_per_segment=args_data.get("num_views_per_segment", 1),
            allow_segment_overlap=True,
            batch_size=batch_size,
            world_size=1,
            rank=0,
            training=False,
            num_workers=num_workers,
            normalization=DEFAULT_NORMALIZATION,
        )

        # Run inference
        preds = run_inference(encoder, probe, test_loader, device)

        # Save predictions for this fold
        pred_path = output_dir / f"fold_{outer_fold}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(preds, f, indent=2)
        logger.info("Saved %d predictions to %s", len(preds), pred_path)

        # Compute metrics
        metrics = compute_metrics(preds)
        metrics["fold_id"] = outer_fold
        all_fold_metrics.append(metrics)

        logger.info(
            "Fold %d: video_acc=%.2f%%  subject_acc=%.2f%%  samples=%d  subjects=%d",
            outer_fold,
            metrics["video_accuracy"],
            metrics["subject_accuracy"],
            metrics["total_samples"],
            metrics["total_subjects"],
        )

        # Cleanup (keep the encoder, only release probe + dataloader)
        try:
            os.remove(test_csv)
        except OSError:
            pass
        del probe, test_loader
        torch.cuda.empty_cache()

    # ---- Aggregate results across folds ----
    if all_fold_metrics:
        video_accs = [m["video_accuracy"] for m in all_fold_metrics]
        subj_accs = [m["subject_accuracy"] for m in all_fold_metrics]
        total_samples = sum(m["total_samples"] for m in all_fold_metrics)
        total_subjects = sum(m["total_subjects"] for m in all_fold_metrics)

        aggregate = {
            "video_accuracy_mean": float(np.mean(video_accs)),
            "video_accuracy_std": float(np.std(video_accs)),
            "subject_accuracy_mean": float(np.mean(subj_accs)),
            "subject_accuracy_std": float(np.std(subj_accs)),
            "total_samples": total_samples,
            "total_subjects": total_subjects,
            "per_fold": all_fold_metrics,
        }

        agg_path = output_dir / "cv_aggregate_metrics.json"
        with open(agg_path, "w") as f:
            json.dump(aggregate, f, indent=2)

        logger.info("=" * 60)
        logger.info("Cross-validation aggregate results:")
        logger.info(
            "  Video accuracy:    %.2f ± %.2f%%",
            aggregate["video_accuracy_mean"],
            aggregate["video_accuracy_std"],
        )
        logger.info(
            "  Subject accuracy:  %.2f ± %.2f%%",
            aggregate["subject_accuracy_mean"],
            aggregate["subject_accuracy_std"],
        )
        logger.info("  Total samples: %d  Total subjects: %d", total_samples, total_subjects)
        logger.info("Saved aggregate metrics to %s", agg_path)


def _df_to_dataset_csv(df, path):
    """Write a subset DataFrame in the space-delimited format VideoDataset expects."""
    cols = ["video_filename", "label", "patient_id"]
    subset = df[cols].copy()
    subset["patient_id"] = subset["patient_id"].apply(
        lambda x: "" if pd.isna(x) else str(int(x)) if isinstance(x, float) else str(x)
    )
    subset.to_csv(path, sep=" ", header=False, index=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validation inference")
    parser.add_argument("--eval_config", required=True, help="Path to eval YAML config")
    parser.add_argument("--cv_checkpoint_dir", required=True, help="Directory containing fold_*/last.pt")
    parser.add_argument("--output_dir", required=True, help="Output directory for predictions and metrics")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
