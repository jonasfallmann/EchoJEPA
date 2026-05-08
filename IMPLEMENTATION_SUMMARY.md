# Implementation Summary: Subject-Level Aggregation for Video Classification

## What Was Done

You requested adaptations to the video classification system to:
1. Support a new "patient_id" column in CSVs for subject level aggregation
2. Average logits over subjects for calculating accuracy (especially for val/test)
3. Create an inference script with subject-level aggregation and confusion matrix visualization

All requirements have been successfully implemented.

## Files Modified

### 1. `src/datasets/video_dataset.py`
**Changes:** Added patient_id support throughout the dataset class
- Modified CSV parsing to load patient_id from 3rd column (lines 192-215)
- Updated `get_item_video()` to return patient_id (line 259)
- Updated `get_item_image()` to return patient_id (line 309)
- Updated `make_videodataset()` to use custom collate function (lines 118-121)

### 2. `src/datasets/utils/utils.py`
**Changes:** Added custom collate function for patient IDs
- Added `default_collate_with_patient_ids()` function (lines 15-34)
- Handles batching of clips, labels, indices, and patient_ids properly
- Made `dataset_paths` import optional to avoid dependency issues (lines 6-9)

### 3. `evals/video_classification_frozen/eval.py`
**Changes:** Implemented subject-level aggregation in validation/test
- Added `from collections import defaultdict` (line 23)
- Modified `run_one_epoch()` to:
  - Extract patient_ids from batch (lines 649-650)
  - Accumulate predictions by subject during non-training phases (lines 714-722)
  - Compute subject-level accuracy after epoch (lines 796-816)
  - Save patient_ids in predictions CSV (lines 819-820)
  - Log subject-level accuracy metrics (lines 798-801)

### 4. `notebooks/inference_with_subject_aggregation.py` (NEW FILE)
**Complete inference pipeline with:**
- Model and probe loading
- Subject-level aggregation during inference
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization and saving
- CSV output with predictions and patient IDs
- Command-line interface for easy usage

## CSV Format Expected

The input CSV files should now support an optional 3rd column:

```
<video_path> <label> <patient_id>
```

Examples:
```
/data/video1.mp4 0 patient_001
/data/video2.mp4 1 patient_001
/data/video3.mp4 0 patient_002
```

**Note:** The 3rd column is optional. If not present, all patient_ids will be `None` and subject aggregation will be skipped gracefully.

## How It Works

### During Training
1. **Training phase**: Video-level accuracy only (unchanged)
2. **Validation phase**: 
   - Logs video-level accuracy for monitoring
   - If patient_ids available: Also logs subject-level accuracy
3. **Test phase**: Same as validation

### Subject-Level Aggregation Algorithm
```
For each subject:
  1. Collect all softmax probabilities from videos of that subject
  2. Average the probabilities: mean_prob = average(all_probs)
  3. Predict: subject_pred = argmax(mean_prob)
  4. Compare with subject's true label
```

## Usage Examples

### Using the inference script:
```bash
python notebooks/inference_with_subject_aggregation.py \
  --eval_config configs/eval/vitl/config.yaml \
  --probe_checkpoint /path/to/best_probe.pt \
  --test_data_path /path/to/test_data.csv \
  --output_dir ./results \
  --gpu_id 0 \
  --save_video_level
```

**Outputs:**
- `results/subject_predictions.csv`: Subject-level predictions
- `results/confusion_matrix.png`: Visual confusion matrix
- `results/video_predictions.csv` (if `--save_video_level`): Video-level predictions

### Running standard training/evaluation:
No changes to your existing workflow! Just ensure your CSV files have the patient_id column.

The training will automatically:
- Load patient_ids from CSVs
- Log subject-level accuracy during validation
- Use subject-level aggregation during testing

## Key Features

✅ **Backward Compatible**: Works with old CSV format (no patient_id column)
✅ **Automatic Aggregation**: No changes needed to training code
✅ **Dual Metrics**: Both video-level and subject-level accuracy reported
✅ **Comprehensive Inference**: Full metrics and visualizations
✅ **Production Ready**: Proper error handling and logging

## Algorithm Implementation

The implementation follows the reference provided:
```python
subject_probs = defaultdict(list)
subject_targets = {}

# During inference
for batch in data_loader:
    logits = model(batch)
    probs = softmax(logits)
    
    for prob, label, subject_id in zip(probs, labels, subject_ids):
        if subject_id is not None:
            subject_probs[subject_id].append(prob)
            subject_targets[subject_id] = label

# After epoch
for subject_id, probs_list in subject_probs.items():
    avg_prob = mean(probs_list)
    pred = argmax(avg_prob)
    actual = subject_targets[subject_id]
    # Compare pred vs actual
```

## Files Syntax Verified ✓

All modified and created files have been syntax-checked:
- ✓ `src/datasets/video_dataset.py`
- ✓ `src/datasets/utils/utils.py`
- ✓ `evals/video_classification_frozen/eval.py`
- ✓ `notebooks/inference_with_subject_aggregation.py`

## Documentation

Full detailed guide available in: `SUBJECT_AGGREGATION_GUIDE.md`

This includes:
- CSV format specifications
- Integration details
- Algorithm description
- Future enhancements
- Backward compatibility notes

