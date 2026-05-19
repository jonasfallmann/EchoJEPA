"""
Inference script for video classification with subject-level aggregation.

This script:
1. Loads a trained probe checkpoint
2. Runs inference on a test dataset
3. Aggregates predictions per subject
4. Reports accuracy metrics
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evals.video_classification_frozen.eval import make_dataloader

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


class InferenceConfig:
    """Configuration for inference."""

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.num_workers = 8


def load_model_and_probe(config_dict, probe_checkpoint_path, device):
    """
    Load pretrained encoder and probe classifier(s).

    Args:
        config_dict: Configuration dictionary from eval config
        probe_checkpoint_path: Path to saved probe checkpoint
        device: Torch device

    Returns:
        encoder, classifiers (list), config dict, best_head_idx
    """
    args_pretrain = config_dict.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = config_dict.get("experiment")
    args_classifier = args_exp.get("classifier")
    args_data = args_exp.get("data")
    args_opt = args_exp.get("optimization")

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

    # Load probe checkpoint to get number of heads
    if not os.path.exists(probe_checkpoint_path):
        raise FileNotFoundError(f"Probe checkpoint not found: {probe_checkpoint_path}")

    checkpoint_dict = robust_checkpoint_loader(probe_checkpoint_path, map_location=device)
    num_classifier_heads = len(checkpoint_dict.get("classifiers", [1]))  # Default to 1 if single

    # Determine best head based on checkpoint metrics
    best_head_idx = 0
    if "best_val_acc_per_head" in checkpoint_dict:
        best_val_accs = checkpoint_dict["best_val_acc_per_head"]
        best_head_idx = int(np.argmax(best_val_accs))
        logger.info(
            f"Found {num_classifier_heads} classifier heads. Best head (highest val acc): {best_head_idx} with acc={best_val_accs[best_head_idx]:.4f}")
    else:
        logger.info(f"Found {num_classifier_heads} classifier head(s). Using head 0 (metrics not found in checkpoint)")

    # Create classifiers
    num_classes = args_data.get("num_classes")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    probe_type = args_classifier.get("probe_type", "attentive")
    use_layernorm = args_classifier.get("use_layernorm", True)
    probe_dropout = args_classifier.get("dropout", 0.0)

    classifiers = []
    for head_idx in range(num_classifier_heads):
        if probe_type == "linear":
            clf = LinearClassifier(
                embed_dim=encoder.embed_dim,
                num_classes=num_classes,
                use_layernorm=use_layernorm,
                dropout=probe_dropout,
            ).to(device)
        elif probe_type == "mlp":
            clf = MLPClassifier(
                embed_dim=encoder.embed_dim,
                num_classes=num_classes,
                use_layernorm=use_layernorm,
                dropout=probe_dropout,
            ).to(device)
        else:  # attentive
            clf = AttentiveClassifier(
                embed_dim=encoder.embed_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes,
                use_activation_checkpointing=True,
            ).to(device)

        clf.load_state_dict(checkpoint_dict["classifiers"][head_idx])
        classifiers.append(clf)

    logger.info(f"Loaded {len(classifiers)} classifier head(s) from {probe_checkpoint_path}")

    return encoder, classifiers, config_dict, best_head_idx


def run_inference(encoder, classifiers, best_head_idx, data_loader, device):
    """
    Run inference and extract raw probabilities into a unified format.
    """
    encoder.eval()

    # Select the best classifier head
    if best_head_idx < 0 or best_head_idx >= len(classifiers):
        logger.warning("best_head_idx out of bounds. Falling back to head 0.")
        chosen_head = 0
    else:
        chosen_head = best_head_idx

    classifier = classifiers[chosen_head]
    classifier.eval()

    unified_predictions = []

    logger.info(f"Running inference with classifier head {chosen_head} with total number of {len(data_loader.dataset)} samples")

    logger.info("Running inference to extract probabilities...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Unpack batch based on Code 1's dataloader format
            clips = [[dij.to(device) for dij in di] for di in batch[0]]
            labels = batch[1].to(device)
            clip_indices = [d.to(device) for d in batch[2]]
            patient_ids = batch[3] if len(batch) > 3 else [None] * len(labels)
            video_paths = batch[4] if len(batch) > 4 else [f"video_{batch_idx}_{i}" for i in range(len(labels))]

            # Forward pass through encoder
            outputs = encoder(clips, clip_indices)

            # Apply classifier
            if isinstance(outputs, list):
                per_clip_logits = [classifier(o) for o in outputs]
            else:
                per_clip_logits = [classifier(outputs)]

            # Average probabilities across clips for the video
            probs_per_clip = [F.softmax(o, dim=1) for o in per_clip_logits]
            avg_probs = sum(probs_per_clip) / len(probs_per_clip)

            batch_size = labels.size(0)
            for i in range(batch_size):
                pid = patient_ids[i] if i < len(patient_ids) else None

                # Handle potential NaN or missing patient IDs cleanly
                if isinstance(pid, float) and np.isnan(pid):
                    pid = None

                unified_predictions.append({
                    'subject_id': pid,
                    'video_id': video_paths[i] if i < len(video_paths) else f"video_{batch_idx}_{i}",
                    'true_label': labels[i].cpu().item(),
                    'probs': avg_probs[i].detach().cpu().numpy().tolist()  # Convert to list for JSON serialization
                })

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * batch_size} samples")

    return unified_predictions


def main(args):
    # Setup
    config = InferenceConfig()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {config.device}")

    # Load config
    import yaml
    with open(args.eval_config, 'r') as f:
        eval_config = yaml.safe_load(f)

    # Load model
    logger.info("Loading model and probe...")
    encoder, classifiers, config_dict, best_head_idx = load_model_and_probe(
        eval_config,
        args.probe_checkpoint,
        config.device,
    )

    # Prepare data
    args_data = config_dict.get("experiment").get("data")
    test_data_path = [args.test_data_path or args_data.get("dataset_val")]

    logger.info(f"Loading test data from {test_data_path}")

    test_loader, _ = make_dataloader(
        dataset_type=args_data.get("dataset_type", "VideoDataset"),
        root_path=test_data_path,
        img_size=args_data.get("resolution", 224),
        frames_per_clip=args_data.get("frames_per_clip", 16),
        frame_step=args_data.get("frame_step", 4),
        eval_duration=args_data.get("clip_duration", None),
        num_segments=args_data.get("num_segments", 1),
        num_views_per_segment=args_data.get("num_views_per_segment", 1),
        allow_segment_overlap=True,
        batch_size=config.batch_size,
        world_size=1,
        rank=0,
        training=False,
        num_workers=config.num_workers,
        normalization=DEFAULT_NORMALIZATION,
    )

    # override head index if provided
    effective_head_idx = best_head_idx
    if args.head_idx is not None:
        logger.info(f"Inference head overwritten. Using head {args.head_idx}")
        effective_head_idx = args.head_idx

    # 1. Extract Predictions
    unified_predictions = run_inference(
        encoder,
        classifiers,
        effective_head_idx,
        test_loader,  # Assuming test_loader is initialized above
        device
    )

    # 2. Save Unified Predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "unified_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(unified_predictions, f, indent=2)

    logger.info(f"Successfully saved {len(unified_predictions)} predictions to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and export unified predictions")
    parser.add_argument("--eval_config", required=True, help="Path to config")
    parser.add_argument("--probe_checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output_dir", required=True, help="Directory to save JSON")
    parser.add_argument("--test_data_path", default=None, help="Optional test data override")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument('--head_idx', type=int, default=None)
    args = parser.parse_args()
    main(args)
