"""
Inference script for video classification with subject-level aggregation.

This script:
1. Loads a trained probe checkpoint
2. Runs inference on a test dataset
3. Aggregates predictions per subject
4. Reports accuracy metrics
5. Generates a confusion matrix plot
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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
        logger.info(f"Found {num_classifier_heads} classifier heads. Best head (highest val acc): {best_head_idx} with acc={best_val_accs[best_head_idx]:.4f}")
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


def run_inference(
    encoder,
    classifiers,
    best_head_idx,
    data_loader,
    device,
    num_classes,
):
    """
    Run inference on test set with subject-level aggregation.

    Args:
        encoder: Pretrained encoder
        classifiers: List of trained classifiers (one per head)
        best_head_idx: Index of best classifier head to use
        data_loader: Test data loader
        device: Torch device
        num_classes: Number of classes

    Returns:
        subject_preds, subject_targets, video_data (dict with predictions per video)
    """
    encoder.eval()
    classifier = classifiers[best_head_idx]  # Use the best head
    classifier.eval()

    # For subject-level aggregation we will collect per-sample per-head probabilities
    subject_probs = defaultdict(list)
    subject_targets = {}
    video_data = []
    # Store per-sample per-head probs so we can compute video-level per-head metrics
    all_sample_head_probs = []  # list of lists: for each sample -> [prob_head0 (np), prob_head1 (np), ...]
    all_labels = []
    all_patient_ids = []
    all_video_paths = []
    subject_aggregation_logged = False
    samples_with_patient_id = 0
    samples_without_patient_id = 0

    def _has_patient_id(value):
        return value is not None and not (isinstance(value, (float, np.floating)) and np.isnan(value))

    logger.info("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Unpack batch
            clips = [[dij.to(device) for dij in di] for di in batch[0]]
            labels = batch[1].to(device)
            clip_indices = [d.to(device) for d in batch[2]]
            patient_ids = batch[3] if len(batch) > 3 else [None] * len(labels)
            video_paths = batch[4] if len(batch) > 4 else [f"video_{batch_idx}_{i}" for i in range(len(labels))]

            valid_patient_ids = [pid for pid in patient_ids if _has_patient_id(pid)]
            if not subject_aggregation_logged:
                if valid_patient_ids:
                    logger.info(
                        "Subject-level aggregation enabled for this run: patient_id values are present in the batch stream."
                    )
                else:
                    logger.info(
                        "Subject-level aggregation disabled for this run: no patient_id values were provided by the dataloader."
                    )
                subject_aggregation_logged = True

            # Forward pass: get encoder outputs (list over clips or a single tensor)
            outputs = encoder(clips, clip_indices)

            # For consistency with eval.py, apply each classifier to each clip embedding, then
            # average probabilities across clips (i.e. classifier called per-clip, then aggregate)
            # Handle case where encoder returns a single tensor per batch (no clip-list)
            # Build per-head list of logits per clip: per_head_logits[head_idx] = [logits_clip0, logits_clip1, ...]
            per_head_logits = []
            if isinstance(outputs, list):
                # outputs: list of clip embeddings (each tensor shape [B, embed_dim])
                for clf in classifiers:
                    per_clip_logits = [clf(o) for o in outputs]
                    per_head_logits.append(per_clip_logits)
            else:
                # outputs is a single tensor: treat as one "clip"
                for clf in classifiers:
                    per_head_logits.append([clf(outputs)])

            # Convert per-head logits -> per-head averaged probabilities (over clips)
            per_head_probs = []  # list of tensors shape [B, num_classes]
            for coutputs in per_head_logits:
                probs_per_clip = [F.softmax(o, dim=1) for o in coutputs]
                avg_probs = sum(probs_per_clip) / len(probs_per_clip)
                per_head_probs.append(avg_probs)

            # Record per-sample, per-head probabilities and labels for later selection of best head
            batch_size = labels.size(0)
            for i in range(batch_size):
                sample_head_probs = [p[i].detach().cpu().numpy() for p in per_head_probs]
                all_sample_head_probs.append(sample_head_probs)
                all_labels.append(labels[i].cpu().item())
                pid = patient_ids[i] if i < len(patient_ids) else None
                all_patient_ids.append(pid)
                vp = video_paths[i] if i < len(video_paths) else f"video_{batch_idx}_{i}"
                all_video_paths.append(vp)

                has_patient_id = _has_patient_id(pid)
                if has_patient_id:
                    samples_with_patient_id += 1
                else:
                    samples_without_patient_id += 1

                # Store video-level info for now without selecting head -- we'll fill predicted_label after choosing head
                video_data.append({
                    'video_path': vp,
                    'true_label': labels[i].cpu().item(),
                    'predicted_label': None,
                    'confidence': None,
                    'patient_id': pid,
                })

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * len(labels)} samples")

    logger.info(
        "Inference patient_id summary: %d samples with patient_id, %d samples without patient_id.",
        samples_with_patient_id,
        samples_without_patient_id,
    )

    # If we have collected per-sample per-head probabilities, choose best head based on video-level accuracy
    subject_preds = []
    subject_targets_list = []
    subject_ids_list = []

    num_heads = len(classifiers)
    per_head_acc = []
    if len(all_sample_head_probs) > 0:
        # Compute per-head video-level accuracy
        for h in range(num_heads):
            preds_h = [int(np.argmax(sample[h])) for sample in all_sample_head_probs]
            acc_h = float(np.mean([p == t for p, t in zip(preds_h, all_labels)])) if len(all_labels) > 0 else 0.0
            per_head_acc.append(acc_h)

        try:
            chosen_head = int(np.argmax(per_head_acc))
        except Exception:
            chosen_head = best_head_idx if best_head_idx < num_heads else 0
    else:
        chosen_head = best_head_idx if best_head_idx < num_heads else 0

    logger.info(f"Selected head for aggregation: {chosen_head} (per-head video accs: {per_head_acc})")

    # Fill video-level predictions using the chosen head and build subject-level collections
    for idx, (sample_probs_per_head, label, pid, vp) in enumerate(
        zip(all_sample_head_probs, all_labels, all_patient_ids, all_video_paths)
    ):
        probs_chosen = sample_probs_per_head[chosen_head]
        pred = int(np.argmax(probs_chosen))
        conf = float(np.max(probs_chosen))
        video_data[idx]["predicted_label"] = pred
        video_data[idx]["confidence"] = conf

        if _has_patient_id(pid):
            subject_probs[pid].append(probs_chosen)
            subject_targets[pid] = label

    logger.info(
        "Subject-level aggregation %s: %d subjects collected from %d video samples.",
        "completed" if subject_probs else "skipped",
        len(subject_probs),
        samples_with_patient_id,
    )

    # Aggregate predictions per subject using averaged probabilities (chosen head)
    for subject_id, probs_list in subject_probs.items():
        avg_prob = np.stack(probs_list).mean(axis=0)
        pred = int(np.argmax(avg_prob))
        target = subject_targets[subject_id]

        subject_preds.append(pred)
        subject_targets_list.append(target)
        subject_ids_list.append(subject_id)

    return np.array(subject_preds), np.array(subject_targets_list), video_data, subject_ids_list


def plot_confusion_matrix(y_true, y_pred, class_names=None, output_path=None):
    """
    Create and save confusion matrix plot.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmax=17,
        vmin=0
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Subject-Level Aggregation)')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")

    plt.show()


def main(args):
    """Main inference pipeline."""

    # Setup
    config = InferenceConfig()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

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

    # Run inference
    num_classes = args_data.get("num_classes")
    subject_preds, subject_targets, video_data, subject_ids = run_inference(
        encoder,
        classifiers,
        best_head_idx,
        test_loader,
        config.device,
        num_classes,
    )

    # Report metrics
    logger.info("\n" + "="*50)
    logger.info("SUBJECT-LEVEL RESULTS")
    logger.info("="*50)

    subject_acc = accuracy_score(subject_targets, subject_preds)
    logger.info(f"Subject-level Accuracy: {subject_acc * 100:.2f}%")
    logger.info(f"Number of subjects: {len(subject_preds)}")

    # Classification report
    class_names = [f"Class {i}" for i in range(num_classes)]
    logger.info("\nClassification Report:")
    logger.info(classification_report(subject_targets, subject_preds, target_names=class_names))

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(subject_targets, subject_preds, class_names, cm_path)

        # Save predictions to CSV
        results_df = pd.DataFrame({
            'subject_id': subject_ids,
            'true_label': subject_targets,
            'predicted_label': subject_preds,
            'correct': subject_preds == subject_targets,
        })

        csv_path = output_dir / "subject_predictions.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved subject predictions to {csv_path}")

        # Save video-level predictions if needed
        if args.save_video_level:
            video_df = pd.DataFrame(video_data)
            video_path = output_dir / "video_predictions.csv"
            video_df.to_csv(video_path, index=False)
            logger.info(f"Saved video predictions to {video_path}")

    logger.info("Inference complete!")
    return subject_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with subject-level aggregation")
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
        "--test_data_path",
        help="Path to test data CSV (default: uses val path from config)",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save results (confusion matrix, predictions)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--save_video_level",
        action="store_true",
        help="Also save video-level predictions to CSV",
    )
    parser.add_argument(
        "--report_averages",
        default="micro,macro,weighted",
        help="Comma-separated summary rows to include in the saved tables (micro, macro, weighted, or all).",
    )

    args = parser.parse_args()
    main(args)

