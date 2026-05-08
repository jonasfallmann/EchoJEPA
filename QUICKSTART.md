# Quick Start Guide: Subject-Level Aggregation

This guide shows you how to quickly get started with the new subject-level aggregation features.

## Step 1: Prepare Your Data

Ensure your CSV files have the patient/subject ID in the 3rd column:

```
/path/to/video1.mp4 0 patient_001
/path/to/video2.mp4 1 patient_001
/path/to/video3.mp4 0 patient_002
/path/to/video4.mp4 1 patient_002
```

**Format:** `<video_path> <label> <patient_id>` (space or :: delimited)

## Step 2: Standard Training (No Changes Needed!)

Your existing training pipeline automatically uses subject-level aggregation:

```bash
python evals/video_classification_frozen/eval.py \
  --config your_eval_config.yaml
```

**Output in logs:**
```
[epoch 1] train: 95.2%  val(max-head): 89.3% (Best: 89.3%)
Subject-level accuracy: 92.1%  # <-- NEW!
```

## Step 3: Run Inference with Confusion Matrix

After training, use the inference script to generate full metrics and visualizations:

```bash
python notebooks/inference_with_subject_aggregation.py \
  --eval_config configs/eval/vitl/config.yaml \
  --probe_checkpoint /path/to/best.pt \
  --test_data_path /path/to/test.csv \
  --output_dir ./results
```

**Outputs generated:**
```
results/
├── confusion_matrix.png          # Visual confusion matrix
├── subject_predictions.csv       # Subject-level predictions
└── video_predictions.csv        # (optional) Video-level predictions
```

## Example 1: Cardiac Classification

**CSV Format:**
```
/data/echocardiograms/echo_001.mp4 0 patient_A
/data/echocardiograms/echo_002.mp4 0 patient_A
/data/echocardiograms/echo_003.mp4 1 patient_B
/data/echocardiograms/echo_004.mp4 1 patient_B
```

**Result:**
- Video-level accuracy: 95% (85/90 videos correct)
- Subject-level accuracy: 98% (49/50 subjects correct)

Subject aggregation reduces noise from multiple scans per patient!

## Example 2: Multi-View Classification  

**CSV Format:**
```
/data/views/apical_01.mp4 2 patient_001
/data/views/apical_02.mp4 2 patient_001
/data/views/parasternal_01.mp4 2 patient_001
/data/views/apical_03.mp4 3 patient_002
```

Multiple views of patient → averaged prediction → more robust accuracy

## Inference Script Options

```bash
python notebooks/inference_with_subject_aggregation.py \
  --eval_config configs/eval/config.yaml \
  --probe_checkpoint /path/to/probe.pt \
  --test_data_path /path/to/test.csv \
  --output_dir ./results \
  --gpu_id 0 \
  --save_video_level
```

**Arguments:**
- `--eval_config`: Path to evaluation config YAML
- `--probe_checkpoint`: Path to trained probe checkpoint (.pt)
- `--test_data_path`: Path to test data CSV (optional, uses config default)
- `--output_dir`: Where to save results
- `--gpu_id`: GPU device ID (default: 0)
- `--save_video_level`: Also save video-level predictions (optional)

## Output Files Explained

### confusion_matrix.png
Visual confusion matrix showing:
- True labels (y-axis) vs predicted labels (x-axis)
- Number of correct/incorrect predictions per class
- Easy identification of which classes are confused

### subject_predictions.csv
```
subject_id,true_label,predicted_label,correct
patient_001,0,0,True
patient_002,1,1,True
patient_003,0,1,False
...
```

Useful for:
- Identifying problematic subjects
- Domain-specific error analysis
- Fine-tuning per-subject quality metrics

### video_predictions.csv (optional)
```
video_path,true_label,predicted_label,confidence,patient_id
/path/to/video1.mp4,0,0,0.95,patient_001
/path/to/video2.mp4,0,0,0.87,patient_001
...
```

## Understanding the Metrics

```
Subject-level Accuracy: 92.1%
Classification Report:
              precision    recall  f1-score   support

       Class 0       0.93      0.91      0.92        45
       Class 1       0.91      0.93      0.92        55

    accuracy                           0.92       100
   macro avg       0.92      0.92      0.92       100
weighted avg       0.92      0.92      0.92       100
```

**Interpretation:**
- **Precision**: Of predicted class-0, how many are actually class-0? (93%)
- **Recall**: Of all actual class-0, how many were predicted correctly? (91%)
- **F1-score**: Harmonic mean of precision and recall

## Troubleshooting

### "KeyError: 'patient_id'" during training?
→ Your CSV file might not have the 3rd column. Add patient IDs or use the old format.

### "Subject-level accuracy" not in logs?
→ Your CSV might not have patient_ids (all None). This is fine—subject aggregation is skipped.

### Confusion matrix shows all zeros?
→ Check that class predictions are in range [0, num_classes-1].

### Poor subject-level accuracy compared to video-level?
→ May indicate inconsistent predictions across videos. Consider:
  - Insufficient frames per video
  - Augmentation too strong
  - Model needs more training

## Advanced Usage

### Custom Class Names in Confusion Matrix

To add class names instead of numbers, modify the inference script:

```python
class_names = ["Normal", "Abnormal", "Artifact"]
plot_confusion_matrix(subject_targets, subject_preds, class_names, output_path)
```

### Weighted Aggregation (Future Enhancement)

Currently uses equal weighting for all videos. Future version could support:
- Quality-weighted averaging
- Confidence-weighted predictions
- Temporal weighting

## Next Steps

1. ✅ Prepare CSV with patient_id column
2. ✅ Run training (automatic subject-level metrics)
3. ✅ Generate inference results with confusion matrix
4. ✅ Analyze per-subject errors
5. ✅ (Optional) Fine-tune model based on error analysis

## More Information

For detailed information, see:
- `IMPLEMENTATION_SUMMARY.md`: Technical details of changes
- `SUBJECT_AGGREGATION_GUIDE.md`: Comprehensive documentation
- `notebooks/inference_with_subject_aggregation.py`: Full source code with docstrings

