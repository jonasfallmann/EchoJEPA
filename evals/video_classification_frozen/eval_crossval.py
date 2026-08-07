# evals/video_classification_frozen/eval_crossval.py
"""
Nested 5-4 cross-validation for video classification probes.

Outer loop (5 folds):
  For each outer fold k (the held-out test fold):
    1. Partition: fold_k = outer_test, remaining 4 folds = inner_pool
    2. Inner 4-fold CV over inner_pool:
       For each inner val fold:
         - Train ALL heads from multihead_kwargs in parallel on remaining 3 folds
         - Evaluate per-head subject accuracy on inner val fold
       - Average per-head subject accuracy across 4 inner folds
       - Select best hyperparameter config
    3. Outer retrain (fresh):
       - Train SINGLE head with best config on ALL 4 inner_pool folds
       - Evaluate on outer_test fold
       - Save last-epoch checkpoint + predictions CSV
"""

import logging
import os
import pprint
import tempfile
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

import wandb

from evals.video_classification_frozen.eval import (
    CosineWDSchedule,
    DEFAULT_NORMALIZATION,
    FocalLoss,
    WarmupCosineLRSchedule,
    init_opt,
    load_checkpoint,
    make_dataloader,
    make_probe,
    run_one_epoch,
)
from evals.video_classification_frozen.models import init_module
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger

# Fix for "AF_UNIX path too long" error
short_tmp = "/tmp/vjepa_run"
os.makedirs(short_tmp, exist_ok=True)
tempfile.tempdir = short_tmp
os.environ["TMPDIR"] = short_tmp

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 42
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


# --------------------------------------------------------------------------- #
#  Temp CSV helpers
# --------------------------------------------------------------------------- #

def _df_to_dataset_csv(df, path):
    """Write a subset DataFrame in the space-delimited format VideoDataset expects.

    Format (space-delimited, no header):  video_path label patient_id
    """
    cols = ["video_filename", "label", "patient_id"]
    subset = df[cols].copy()
    # Ensure patient_id is written as string; None → empty
    subset["patient_id"] = subset["patient_id"].apply(
        lambda x: "" if pd.isna(x) else str(int(x)) if isinstance(x, float) else str(x)
    )
    subset.to_csv(path, sep=" ", header=False, index=False)
    return path


def _write_temp_csv(df, fold_ids, tmp_dir):
    """Write a temporary CSV for given fold_ids (int or list of ints)."""
    if isinstance(fold_ids, (int, np.integer)):
        subset = df[df["fold_id"] == fold_ids]
    else:
        subset = df[df["fold_id"].isin(fold_ids)]
    path = os.path.join(tmp_dir, f"fold_{uuid.uuid4().hex[:8]}.csv")
    _df_to_dataset_csv(subset, path)
    return path


# --------------------------------------------------------------------------- #
#  Inner cross-validation
# --------------------------------------------------------------------------- #

def _inner_cv(
    *,
    encoder,
    inner_pool_df,
    inner_fold_ids,
    world_size,
    rank,
    device,
    args_data,
    args_classifier,
    args_opt,
    opt_kwargs,
    use_ddp,
    probe_kwargs,
    inner_num_epochs,
    use_bfloat16,
    use_focal_loss,
    task_type,
    target_mean,
    target_std,
    tmp_dir,
    num_workers=8,
):
    """Run 4-fold inner CV over inner_pool_df. Returns (best_head_idx, per_head_avg_acc).

    Each inner fold:
      - Train ALL heads from opt_kwargs in parallel on 3 training folds
      - Validate on 1 held-out inner fold
      - Record per-head SUBJECT-LEVEL accuracy

    Result: average per-head subject accuracy across all 4 inner folds.
    """
    num_heads = len(opt_kwargs)
    # per_head_accs[h] = list of subject accuracies from each inner fold
    per_head_accs = defaultdict(list)

    for inner_val_fold in sorted(inner_fold_ids):
        inner_train_folds = [f for f in inner_fold_ids if f != inner_val_fold]

        train_csv = _write_temp_csv(inner_pool_df, inner_train_folds, tmp_dir)
        val_csv = _write_temp_csv(inner_pool_df, inner_val_fold, tmp_dir)

        logger.info(
            f"[Inner CV] val_fold={inner_val_fold}  "
            f"train_folds={inner_train_folds}  "
            f"train_csv={train_csv}  val_csv={val_csv}"
        )

        # ---- Data loaders ----
        dataset_type = args_data.get("dataset_type", "VideoDataset")
        resolution = args_data.get("resolution", 224)
        frames_per_clip = args_data.get("frames_per_clip", 16)
        frame_step = args_data.get("frame_step", 4)
        duration = args_data.get("clip_duration", None)
        num_segments = args_data.get("num_segments", 1)
        num_views_per_segment = args_data.get("num_views_per_segment", 1)
        normalization = args_data.get("normalization", None)
        batch_size = args_opt.get("batch_size", 1)

        train_loader, train_sampler = make_dataloader(
            dataset_type=dataset_type,
            root_path=[train_csv],
            img_size=resolution,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            eval_duration=duration,
            num_segments=num_segments,
            num_views_per_segment=1,
            allow_segment_overlap=True,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            training=True,
            num_workers=num_workers,
            normalization=normalization,
        )
        val_loader, _val_sampler = make_dataloader(
            dataset_type=dataset_type,
            root_path=[val_csv],
            img_size=resolution,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            eval_duration=duration,
            num_segments=num_segments,
            num_views_per_segment=num_views_per_segment,
            allow_segment_overlap=True,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            training=False,
            num_workers=num_workers,
            normalization=normalization,
        )
        ipe = len(train_loader)
        logger.info(f"[Inner CV] fold={inner_val_fold}  ipe={ipe}")

        # ---- Build classifiers (all heads, fresh init) ----
        classifiers = [
            make_probe(
                task=task_type,
                probe=probe_kwargs["probe_type"],
                embed_dim=encoder.embed_dim,
                num_classes=probe_kwargs["num_classes"],
                num_targets=probe_kwargs["num_targets"],
                num_heads=probe_kwargs["num_heads"],
                depth=probe_kwargs["num_probe_blocks"],
                use_ln=probe_kwargs["use_layernorm"],
                dropout=probe_kwargs["probe_dropout"],
            ).to(device)
            for _ in opt_kwargs
        ]

        if use_ddp:
            classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]

        if rank == 0:
            for h_idx, c in enumerate(classifiers):
                n_trainable = sum(p.numel() for p in c.parameters() if p.requires_grad)
                n_total = sum(p.numel() for p in c.parameters())
                logger.info(
                    "[Inner CV fold=%d] head %d (%s): total_params=%d  trainable=%d",
                    inner_val_fold, h_idx, c.__class__.__name__, n_total, n_trainable,
                )

        # ---- Optimizer & schedulers ----
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            classifiers=classifiers,
            iterations_per_epoch=ipe,
            opt_kwargs=opt_kwargs,
            num_epochs=inner_num_epochs,
            use_bfloat16=use_bfloat16,
        )

        # ---- Training loop ----
        best_per_head_subj = None
        for epoch in range(inner_num_epochs):
            train_sampler.set_epoch(epoch)

            train_acc_scalar, _, _ = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                use_focal_loss=use_focal_loss,
                task_type=task_type,
                target_mean=target_mean,
                target_std=target_std,
                return_per_head_subject=False,
            )

            val_acc_scalar, _, per_head_subj = run_one_epoch(
                device=device,
                training=False,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=val_loader,
                use_bfloat16=use_bfloat16,
                use_focal_loss=use_focal_loss,
                task_type=task_type,
                target_mean=target_mean,
                target_std=target_std,
                return_per_head_subject=True,
            )

            # Track best per-head subject accuracy across epochs
            if per_head_subj is not None:
                per_head_arr = np.array(per_head_subj)
                if best_per_head_subj is None:
                    best_per_head_subj = per_head_arr.copy()
                else:
                    best_per_head_subj = np.maximum(best_per_head_subj, per_head_arr)

            if rank == 0:
                logger.info(
                    "[Inner CV fold=%d epoch=%d] val_acc=%.2f  per_head_subj=%s",
                    inner_val_fold,
                    epoch + 1,
                    val_acc_scalar,
                    [f"{a:.2f}" for a in (per_head_subj or [])],
                )

        # Record best per-head subject accuracy for this inner fold
        if best_per_head_subj is not None:
            for h_idx, acc in enumerate(best_per_head_subj):
                per_head_accs[h_idx].append(float(acc))

        # ---- Cleanup temp CSVs ----
        for p in [train_csv, val_csv]:
            try:
                os.remove(p)
            except OSError:
                pass

        # ---- Cleanup GPU memory ----
        del classifiers, optimizer, scaler, scheduler, wd_scheduler
        del train_loader, val_loader, train_sampler
        torch.cuda.empty_cache()

    # ---- Average across inner folds ----
    if not per_head_accs:
        raise RuntimeError(
            "No per-head subject accuracies collected during inner CV. "
            "Ensure the crossval CSV has valid patient_id values."
        )
    avg_accs = {}
    for h_idx, acc_list in per_head_accs.items():
        avg_accs[h_idx] = float(np.mean(acc_list))
        logger.info(
            "[Inner CV summary] head=%d  per_fold=%s  avg=%.2f",
            h_idx,
            [f"{a:.2f}" for a in acc_list],
            avg_accs[h_idx],
        )

    best_head_idx = int(max(avg_accs, key=avg_accs.get))
    logger.info("[Inner CV] best_head=%d (avg_subj_acc=%.2f)", best_head_idx, avg_accs[best_head_idx])

    return best_head_idx, avg_accs


# --------------------------------------------------------------------------- #
#  Outer retraining with best config
# --------------------------------------------------------------------------- #

def _outer_retrain_and_eval(
    *,
    encoder,
    outer_train_df,
    outer_test_df,
    outer_fold_id,
    best_config,
    world_size,
    rank,
    device,
    args_data,
    args_classifier,
    args_opt,
    use_ddp,
    probe_kwargs,
    num_epochs,
    use_bfloat16,
    use_focal_loss,
    task_type,
    target_mean,
    target_std,
    folder,
    tmp_dir,
    num_workers=8,
):
    """Train a SINGLE head with best_config on all outer training data, eval on outer test."""
    train_csv = _write_temp_csv(outer_train_df, list(outer_train_df["fold_id"].unique()), tmp_dir)
    test_csv = _write_temp_csv(outer_test_df, list(outer_test_df["fold_id"].unique()), tmp_dir)

    logger.info(
        "[Outer fold=%d] train_csv=%s  test_csv=%s  config=%s",
        outer_fold_id, train_csv, test_csv, best_config,
    )

    # ---- Data loaders ----
    batch_size = args_opt.get("batch_size", 1)
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    resolution = args_data.get("resolution", 224)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_segments = args_data.get("num_segments", 1)
    normalization = args_data.get("normalization", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=[train_csv],
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
    )
    test_loader, _test_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=[test_csv],
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
    )
    ipe = len(train_loader)

    # ---- Single head classifier ----
    opt_kwargs_single = [best_config]

    classifier = make_probe(
        task=task_type,
        probe=probe_kwargs["probe_type"],
        embed_dim=encoder.embed_dim,
        num_classes=probe_kwargs["num_classes"],
        num_targets=probe_kwargs["num_targets"],
        num_heads=probe_kwargs["num_heads"],
        depth=probe_kwargs["num_probe_blocks"],
        use_ln=probe_kwargs["use_layernorm"],
        dropout=probe_kwargs["probe_dropout"],
    ).to(device)

    classifiers = [classifier]
    if use_ddp:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]

    if rank == 0:
        for c in classifiers:
            n_trainable = sum(p.numel() for p in c.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in c.parameters())
            logger.info(
                "[Outer fold=%d] %s: total_params=%d  trainable=%d",
                outer_fold_id, c.__class__.__name__, n_total, n_trainable,
            )

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        iterations_per_epoch=ipe,
        opt_kwargs=opt_kwargs_single,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # ---- Training ----
    best_val = float("-inf") if task_type == "classification" else float("inf")
    last_val_heads = None

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        train_acc_scalar, _, _ = run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
            use_focal_loss=use_focal_loss,
            task_type=task_type,
            target_mean=target_mean,
            target_std=target_std,
            return_per_head_subject=False,
        )

        val_acc_scalar, val_heads, _ = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=test_loader,
            use_bfloat16=use_bfloat16,
            use_focal_loss=use_focal_loss,
            task_type=task_type,
            target_mean=target_mean,
            target_std=target_std,
            return_per_head_subject=False,
        )

        if rank == 0:
            metric_name = "acc" if task_type == "classification" else "mae"
            logger.info(
                "[Outer fold=%d epoch=%d] train_%s=%.3f  val_%s=%.3f",
                outer_fold_id, epoch + 1, metric_name, train_acc_scalar, metric_name, val_acc_scalar,
            )

        last_val_heads = val_heads

    # ---- Save predictions on outer test fold ----
    fold_dir = os.path.join(folder, f"fold_{outer_fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    predictions_path = os.path.join(fold_dir, "predictions.csv")

    _final_val, _, _ = run_one_epoch(
        device=device,
        training=False,
        encoder=encoder,
        classifiers=classifiers,
        scaler=scaler,
        optimizer=optimizer,
        scheduler=scheduler,
        wd_scheduler=wd_scheduler,
        data_loader=test_loader,
        use_bfloat16=use_bfloat16,
        use_focal_loss=use_focal_loss,
        task_type=task_type,
        target_mean=target_mean,
        target_std=target_std,
        val_only=True,
        predictions_save_path=predictions_path,
        return_per_head_subject=False,
    )

    # ---- Save last-epoch checkpoint ----
    checkpoint_path = os.path.join(fold_dir, "last.pt")
    save_dict = {
        "classifiers": [c.state_dict() for c in classifiers],
        "opt": [o.state_dict() for o in optimizer],
        "scaler": None if scaler is None else [None if s is None else s.state_dict() for s in scaler],
        "epoch": num_epochs,
        "batch_size": batch_size,
        "world_size": world_size,
        "opt_grid": opt_kwargs_single,
        "outer_fold_id": outer_fold_id,
        "best_config": best_config,
        "val_acc_per_head": np.asarray(last_val_heads, dtype=float).tolist() if last_val_heads is not None else [],
    }
    if rank == 0:
        torch.save(save_dict, checkpoint_path)
        logger.info("[Outer fold=%d] saved checkpoint → %s", outer_fold_id, checkpoint_path)

    # ---- Cleanup ----
    for p in [train_csv, test_csv]:
        try:
            os.remove(p)
        except OSError:
            pass

    del classifiers, optimizer, scaler, scheduler, wd_scheduler
    del train_loader, test_loader, train_sampler
    torch.cuda.empty_cache()

    return last_val_heads


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

def main_crossval(args_eval, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  Config parsing (mirrors eval.py main)
    # ----------------------------------------------------------------------- #
    import os as _os

    def set_override(env_var, target_dict, key, type_func=str):
        val = _os.environ.get(env_var)
        if val is not None:
            if type_func == bool:
                val = val.lower() in ("true", "1", "t", "yes")
            else:
                val = type_func(val)
            print(f"!!! MANUAL OVERRIDE: {key} -> {val}")
            target_dict[key] = val

    set_override("OVERRIDE_TAG", args_eval, "tag")
    set_override("OVERRIDE_VAL_ONLY", args_eval, "val_only", bool)
    set_override("OVERRIDE_PRED_PATH", args_eval, "predictions_save_path")
    set_override("OVERRIDE_CKPT", args_eval, "probe_checkpoint")

    exp = args_eval.setdefault("experiment", {})
    clf = exp.setdefault("classifier", {})
    data = exp.setdefault("data", {})
    opt = exp.setdefault("optimization", {})

    set_override("OVERRIDE_NUM_HEADS", clf, "num_heads", int)
    set_override("OVERRIDE_NUM_BLOCKS", clf, "num_probe_blocks", int)
    set_override("OVERRIDE_NUM_CLASSES", data, "num_classes", int)
    set_override("OVERRIDE_RES", data, "resolution", int)
    set_override("OVERRIDE_TARGET_MEAN", data, "target_mean", float)
    set_override("OVERRIDE_TARGET_STD", data, "target_std", float)
    set_override("OVERRIDE_EPOCHS", opt, "num_epochs", int)
    set_override("OVERRIDE_FOCAL_LOSS", opt, "use_focal_loss", bool)
    set_override("OVERRIDE_BATCH", opt, "batch_size", int)

    # -- Validate mutual exclusion --
    has_train_val = data.get("dataset_train") or data.get("dataset_val")
    has_crossval = bool(data.get("dataset_crossval"))
    if has_train_val and has_crossval:
        raise ValueError(
            "dataset_train/dataset_val and dataset_crossval are mutually exclusive. "
            "Specify only one mode."
        )
    if not has_crossval:
        raise ValueError("dataset_crossval must be specified for cross-validation mode.")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    task_type = args_classifier.get("task_type", "classification")
    num_targets = args_classifier.get("num_targets", None)
    probe_type = args_classifier.get("probe_type", "attentive")
    use_layernorm = args_classifier.get("use_layernorm", True)
    probe_dropout = args_classifier.get("dropout", 0.0)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_crossval_path = args_data.get("dataset_crossval")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)
    target_mean = args_data.get("target_mean", None)
    target_std = args_data.get("target_std", None)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    use_focal_loss = args_opt.get("use_focal_loss", False)
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    inner_num_epochs = args_opt.get("inner_num_epochs", num_epochs)
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]

    # ----------------------------------------------------------------------- #
    #  Distributed setup
    # ----------------------------------------------------------------------- #
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # ----------------------------------------------------------------------- #
    #  Wandb
    # ----------------------------------------------------------------------- #
    if rank == 0:
        wandb_run_name = eval_tag if eval_tag else "crossval_video_classification"
        wandb.init(
            project="miracle-video-classification-crossval",
            name=wandb_run_name,
            config=args_eval,
            resume="allow" if resume_preempt else None,
        )

    # ----------------------------------------------------------------------- #
    #  Folder setup
    # ----------------------------------------------------------------------- #
    folder = os.path.join(pretrain_folder, "video_classification_frozen_crossval/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)

    use_ddp = dist.is_available() and dist.is_initialized() and world_size > 1

    # ----------------------------------------------------------------------- #
    #  Load crossval CSV
    # ----------------------------------------------------------------------- #
    logger.info(f"Loading cross-validation CSV: {dataset_crossval_path}")
    cv_df = pd.read_csv(dataset_crossval_path)  # comma-delimited with header
    required_cols = {"patient_id", "video_filename", "label", "fold_id"}
    missing = required_cols - set(cv_df.columns)
    if missing:
        raise ValueError(f"Crossval CSV missing required columns: {missing}")
    logger.info(f"Loaded {len(cv_df)} samples with columns: {list(cv_df.columns)}")

    all_fold_ids = sorted(cv_df["fold_id"].unique())

    # Optional: restrict to a single outer fold (e.g. --fold 4)
    outer_fold_override = args_eval.get("outer_fold")
    if outer_fold_override is not None:
        if outer_fold_override not in all_fold_ids:
            raise ValueError(
                f"--fold {outer_fold_override} not found in crossval CSV. "
                f"Available folds: {all_fold_ids}"
            )
        all_fold_ids = [outer_fold_override]
        logger.info(f"Running single outer fold only: {outer_fold_override}")
    elif len(all_fold_ids) != 5:
        raise ValueError(f"Expected 5 fold_ids in crossval CSV, got {len(all_fold_ids)}: {all_fold_ids}")

    logger.info(f"Fold IDs: {all_fold_ids}")
    for fid in all_fold_ids:
        n = (cv_df["fold_id"] == fid).sum()
        logger.info(f"  fold {fid}: {n} samples")

    # ----------------------------------------------------------------------- #
    #  Load frozen encoder (once)
    # ----------------------------------------------------------------------- #
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    logger.info("Encoder loaded and frozen.")

    # ----------------------------------------------------------------------- #
    #  Probe kwargs dict for make_probe
    # ----------------------------------------------------------------------- #
    probe_kwargs = dict(
        probe_type=probe_type,
        num_classes=num_classes,
        num_targets=num_targets,
        num_heads=num_heads,
        num_probe_blocks=num_probe_blocks,
        use_layernorm=use_layernorm,
        probe_dropout=probe_dropout,
    )

    # ----------------------------------------------------------------------- #
    #  Outer cross-validation loop
    # ----------------------------------------------------------------------- #
    cv_results = []  # list of per-fold summaries

    for outer_fold in all_fold_ids:
        logger.info("=" * 70)
        logger.info(f"[Outer CV] fold {outer_fold} / {all_fold_ids}")
        logger.info("=" * 70)

        outer_test_df = cv_df[cv_df["fold_id"] == outer_fold]
        inner_pool_df = cv_df[cv_df["fold_id"] != outer_fold]
        inner_fold_ids = sorted(inner_pool_df["fold_id"].unique())

        assert len(inner_fold_ids) == 4, f"Expected 4 inner folds, got {len(inner_fold_ids)}"

        logger.info(
            "Outer fold %d: test=%d samples, inner_pool=%d samples (folds=%s)",
            outer_fold,
            len(outer_test_df),
            len(inner_pool_df),
            inner_fold_ids,
        )

        # ---- Step 1: Inner CV to find best config ----
        best_head_idx, per_head_avg = _inner_cv(
            encoder=encoder,
            inner_pool_df=inner_pool_df,
            inner_fold_ids=inner_fold_ids,
            world_size=world_size,
            rank=rank,
            device=device,
            args_data=args_data,
            args_classifier=args_classifier,
            args_opt=args_opt,
            opt_kwargs=opt_kwargs,
            use_ddp=use_ddp,
            probe_kwargs=probe_kwargs,
            inner_num_epochs=inner_num_epochs,
            use_bfloat16=use_bfloat16,
            use_focal_loss=use_focal_loss,
            task_type=task_type,
            target_mean=target_mean,
            target_std=target_std,
            tmp_dir=short_tmp,
            num_workers=num_workers,
        )

        best_config = opt_kwargs[best_head_idx]
        logger.info(
            "[Outer fold=%d] Selected best config (head=%d): %s",
            outer_fold, best_head_idx, best_config,
        )

        # ---- Step 2: Outer retraining with best config ----
        val_heads = _outer_retrain_and_eval(
            encoder=encoder,
            outer_train_df=inner_pool_df,
            outer_test_df=outer_test_df,
            outer_fold_id=outer_fold,
            best_config=best_config,
            world_size=world_size,
            rank=rank,
            device=device,
            args_data=args_data,
            args_classifier=args_classifier,
            args_opt=args_opt,
            use_ddp=use_ddp,
            probe_kwargs=probe_kwargs,
            num_epochs=num_epochs,
            use_bfloat16=use_bfloat16,
            use_focal_loss=use_focal_loss,
            task_type=task_type,
            target_mean=target_mean,
            target_std=target_std,
            folder=folder,
            tmp_dir=short_tmp,
            num_workers=num_workers,
        )

        # Record results
        fold_result = {
            "outer_fold": outer_fold,
            "best_head_idx": best_head_idx,
            "best_config": best_config,
            "inner_per_head_avg": per_head_avg,
            "val_heads": val_heads.tolist() if val_heads is not None else [],
        }
        cv_results.append(fold_result)

        if rank == 0:
            wandb.log(
                {
                    f"fold_{outer_fold}/best_head_idx": best_head_idx,
                    f"fold_{outer_fold}/inner_avg_subj_acc_best": per_head_avg.get(best_head_idx, -1),
                },
                step=outer_fold,
            )

    # ----------------------------------------------------------------------- #
    #  Summary
    # ----------------------------------------------------------------------- #
    if rank == 0:
        summary_path = os.path.join(folder, "cv_summary.csv")
        summary_rows = []
        for r in cv_results:
            summary_rows.append({
                "outer_fold": r["outer_fold"],
                "best_head_idx": r["best_head_idx"],
                "best_lr": r["best_config"]["ref_lr"],
                "best_wd": r["best_config"]["ref_wd"],
                "inner_avg_subj_acc": r["inner_per_head_avg"].get(r["best_head_idx"], -1),
            })
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info("Cross-validation summary saved to %s", summary_path)

        # Log per-head averages across folds
        all_inner_avgs = {}
        for r in cv_results:
            for h, acc in r["inner_per_head_avg"].items():
                all_inner_avgs.setdefault(h, []).append(acc)
        for h, accs in sorted(all_inner_avgs.items()):
            logger.info(
                "[CV Summary] head=%d inner_avg_subj_acc = %.2f ± %.2f (across %d folds)",
                h, np.mean(accs), np.std(accs), len(accs),
            )

        wandb.finish()

    logger.info("Cross-validation complete.")
