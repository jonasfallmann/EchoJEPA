# Video Classification with Subject-Level Aggregation

This document describes the changes made to support patient/subject-level aggregation for video classification accuracy metrics.

## Overview

The video classification system has been adapted to:
1. **Load patient/subject IDs** from CSV files (3rd column)
2. **Aggregate logits over subjects** for more accurate evaluation metrics
3. **Report subject-level accuracy** during validation and testing
4. **Provide an inference script** for loading trained models and generating confusion matrices

## Changes Made

### 1. Dataset Changes (`src/datasets/video_dataset.py`)

- **Added patient_id support**: The `VideoDataset` class now reads an optional 3rd column from CSV files as `patient_id`
- **Updated return format**: Both `get_item_video()` and `get_item_image()` now return 4 elements: `(buffer, label, clip_indices, patient_id)`
- **Custom collate function**: Added `default_collate_with_patient_ids()` in `src/datasets/utils/utils.py` to properly batch patient IDs

### 2. Evaluation Script Changes (`evals/video_classification_frozen/eval.py`)

Key modifications in `run_one_epoch()`:

- **Extract patient IDs from batch**: Unpack patient_ids from the batch data
- **Subject-level aggregation**: During validation/test (non-training), accumulate predictions grouped by subject
- **Aggregate logits per subject**: Average softmax probabilities across all videos for each subject
- **Report subject-level accuracy**: Log subject-level classification accuracy after aggregation
- **Save patient IDs in predictions CSV**: Now includes patient_id column when saving predictions

### 3. New Inference Script (`notebooks/inference_with_subject_aggregation.py`)

A standalone script for running inference with subject-level aggregation:

```bash
python notebooks/inference_with_subject_aggregation.py \
  --eval_config /path/to/eval_config.yaml \
  --probe_checkpoint /path/to/probe.pt \
  --test_data_path /path/to/test_data.csv \
  --output_dir ./inference_results \
  --gpu_id 0 \
  --save_video_level
```

**Features:**
- Loads a trained encoder and classifier probe
- Runs inference on test set with automatic subject-level aggregation
- Reports:
  - Subject-level accuracy
  - Per-class metrics (precision, recall, F1)
  - Confusion matrix visualization
- Saves results to CSV files:
  - `subject_predictions.csv`: Subject-level predictions
  - `confusion_matrix.png`: Visual confusion matrix
  - `video_predictions.csv` (optional): Video-level predictions

## CSV Format

The input CSV files should have the following format (space or :: delimited):

```
/path/to/video1.mp4 0 patient_001
/path/to/video2.mp4 1 patient_001  
/path/to/video3.mp4 0 patient_002
/path/to/video4.mp4 1 patient_002
...
```

**Columns:**
1. **Video path** (required): Path to video file (local or S3)
2. **Label** (required): Integer class label
3. **Patient ID** (optional): Unique identifier for subject/patient aggregation

If the 3rd column is not present, patient_id will be `None` and subject-level aggregation will be skipped.

## Subject-Level Aggregation Algorithm

During validation and test phases:

1. For each sample, extract its softmax probability distribution
2. Group all samples by their `patient_id`
3. For each patient, compute the **mean** of all softmax probabilities
4. Take `argmax` of mean probabilities to get patient-level prediction
5. Compare patient-level predictions to patient-level labels
6. Compute accuracy and other metrics at the patient level

**Advantages:**
- Reduces label noise from multiple samples per patient
- More realistic evaluation for clinical applications
- Better reflects actual model performance when used in practice

## Integration into Training Pipeline

The subject-level aggregation is automatically applied during validation and testing in the standard training script:

1. **Training phase**: Video-level metrics only (no aggregation)
2. **Validation phase** (each epoch): 
   - Logs video-level accuracy for quick feedback
   - Also logs subject-level accuracy if patient IDs are available
3. **Test phase**: Final evaluation with subject-level aggregation

No changes needed to your training configuration—just ensure your CSV files have the patient_id column!

## Example Usage

### Training with subject-level evaluation:

```bash
python evals/video_classification_frozen/eval.py \
  --config /path/to/eval_config.yaml
```

The training logs will now show both:
```
train_acc: 95.2%
val_acc: 89.3%        # video-level accuracy
val_subject_acc: 92.1% # NEW: subject-level accuracy
```

### Running inference on test set:

```bash
python notebooks/inference_with_subject_aggregation.py \
  --eval_config /path/to/eval_config.yaml \
  --probe_checkpoint /path/to/best.pt \
  --test_data_path /path/to/test_data.csv \
  --output_dir ./results
```

This will generate:
- `results/confusion_matrix.png`: Visual confusion matrix
- `results/subject_predictions.csv`: Per-subject predictions and correctness
- `results/video_predictions.csv`: Per-video predictions (if `--save_video_level` is used)

## Backward Compatibility

- **CSV files without patient_id**: The code gracefully handles this by setting patient_ids to `None`. Subject-level aggregation is skipped, and the system behaves as before.
- **Existing code**: No changes needed to existing evaluation code—subject-level metrics are logged alongside video-level metrics.

## Reference Implementation

The aggregation logic is based on the reference implementation provided, which uses the following pattern:

```python
subject_probs = defaultdict(list)  # Group probabilities by subject
subject_targets = {}               # Map subject to true label

# Aggregate during epoch
for batch in data_loader:
    # ... forward pass ...
    for prob, label, subj_id in zip(probs, labels, subject_ids):
        if subj_id is not None:
            subject_probs[subj_id].append(prob.cpu())
            subject_targets[subj_id] = label.item()

# After epoch, compute subject-level accuracy
subject_preds = []
subject_targets_list = []
for subj_id, probs_list in subject_probs.items():
    avg_prob = torch.stack(probs_list).mean(dim=0)
    subject_preds.append(torch.argmax(avg_prob).item())
    subject_targets_list.append(subject_targets[subj_id])

accuracy = np.mean([p == t for p, t in zip(subject_preds, subject_targets_list)])
```

## Files Modified

1. `src/datasets/video_dataset.py`: Added patient_id loading and return
2. `src/datasets/utils/utils.py`: Added custom collate function
3. `evals/video_classification_frozen/eval.py`: Added subject-level aggregation logic

## Files Created

1. `notebooks/inference_with_subject_aggregation.py`: Standalone inference script

## Future Enhancements

- Support for weighted aggregation based on video quality scores
- TorchScript export with built-in subject aggregation
- Multi-label classification support
- Hierarchical aggregation (subject → site → dataset level)

