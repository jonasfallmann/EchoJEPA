#!/usr/bin/env python3
"""EchoJEPA video embedding PCA visualization.

Standalone companion script for `video_embedding_pca_temporal.ipynb`.
It loads a trained checkpoint, extracts patch tokens from a video clip,
and visualizes them with a DINO-style PCA color mapping plus temporal views.
"""

# %%
from __future__ import annotations

from pathlib import Path
import argparse
import tempfile

import matplotlib

# Force matplotlib to not use any Xwindows backend since this runs on a server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

try:
    from src.datasets.video_dataset import VideoDataset
except Exception as exc:  # pragma: no cover - optional dependency path
    VideoDataset = None
    print(f"VideoDataset import skipped: {exc}")

from src.datasets.utils.video.transforms import CenterCrop, Compose, Normalize, Resize
from src.datasets.utils.video.volume_transforms import ClipToTensor
from src.models import vision_transformer as vit
from src.utils.checkpoint_loader import robust_checkpoint_loader

plt.style.use("seaborn-v0_8-whitegrid")
np.random.seed(0)
torch.manual_seed(0)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULTS = dict(
    CHECKPOINT_PATH="/system/user/publicwork/fallmann/miracle/EchoJEPA/checkpoints/echojepa_vitg.pt",
    CHECKPOINT_KEY="target_encoder",
    MODEL_NAME="vit_giant_xformers_rope",
    IMG_SIZE=512,
    NUM_FRAMES=16,
    TUBELET_SIZE=2,
    PATCH_SIZE=16,
    VIDEO_SOURCE="/restricteddata/miracle/demo.mp4",
    CONE_MASK="/restricteddata/miracle/demo_mask.png",
    SAMPLE_INDEX=0,
    FRAME_STEP=2,
    NUM_CLIPS=1,
    RANDOM_CLIP_SAMPLING=False,
    ALLOW_CLIP_OVERLAP=False,
    OUT_DIR="/system/user/publicwork/fallmann/miracle"
)


# %%
def build_pt_video_transform(img_size: int):
    short_side_size = int(256.0 / 224 * img_size)
    return Compose(
        [
            Resize(short_side_size, interpolation="bilinear"),
            CenterCrop(size=(img_size, img_size)),
            ClipToTensor(),
            Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def strip_prefixes(state_dict):
    for prefix in ("module.", "backbone."):
        state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items()}
    return state_dict


def load_encoder_from_checkpoint(model, checkpoint_path: str, checkpoint_key: str):
    ckpt = robust_checkpoint_loader(checkpoint_path, map_location="cpu")
    print("checkpoint keys:", list(ckpt.keys()))
    if checkpoint_key not in ckpt:
        for key in ("target_encoder", "encoder"):
            if key in ckpt:
                checkpoint_key = key
                break
    if checkpoint_key not in ckpt:
        raise KeyError(
            f"Could not find encoder weights in checkpoint. Tried {checkpoint_key!r}. "
            f"Available keys: {list(ckpt.keys())}"
        )

    state_dict = strip_prefixes(ckpt[checkpoint_key])
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {checkpoint_key!r} with msg: {msg}")
    return ckpt


def make_dataset(
        video_source: str,
        frames_per_clip: int,
        frame_step: int,
        num_clips: int = 1,
        random_clip_sampling: bool = False,
        allow_clip_overlap: bool = False,
):
    if VideoDataset is None:
        return None

    if video_source.endswith((".csv", ".npy")):
        data_paths = video_source
        temp_csv_path = None
    else:
        tmp_dir = Path(tempfile.gettempdir())
        temp_csv_path = tmp_dir / "echojepa_single_video.csv"
        temp_csv_path.write_text(f"{video_source} 0\n", encoding="utf-8")
        data_paths = str(temp_csv_path)

    ds = VideoDataset(
        data_paths=data_paths,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        transform=None,
        shared_transform=None,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=False,
        filter_long_videos=int(10 ** 9),
        duration=None,
        fps=None,
    )
    return ds


def load_raw_clip_from_source(video_source: str, sample_index: int, *, num_frames: int, frame_step: int,
                              num_clips: int, random_clip_sampling: bool, allow_clip_overlap: bool):
    ds = make_dataset(
        video_source=video_source,
        frames_per_clip=num_frames,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
    )
    if ds is not None:
        buffer, label, clip_indices = ds[sample_index]
        raw_clip = buffer[0] if isinstance(buffer, list) else buffer
        clip_indices = clip_indices[0] if isinstance(clip_indices, list) else clip_indices
        return raw_clip, clip_indices, label

    vr = VideoReader(video_source, num_threads=-1, ctx=cpu(0))
    frame_idx = np.arange(0, min(len(vr), num_frames * frame_step), frame_step)[:num_frames]
    raw_clip = vr.get_batch(frame_idx).asnumpy()
    return raw_clip, frame_idx, 0


def prepare_clip_for_model(raw_clip: np.ndarray, img_size: int):
    video = torch.from_numpy(raw_clip).permute(0, 3, 1, 2)  # T, C, H, W
    pt_transform = build_pt_video_transform(img_size)
    return pt_transform(video).unsqueeze(0)  # B, C, T, H, W


def reshape_tokens_to_grid(tokens: torch.Tensor, clip: torch.Tensor, patch_size: int, tubelet_size: int):
    _, _, T, H, W = clip.shape
    t = T // tubelet_size
    h = H // patch_size
    w = W // patch_size
    assert tokens.shape[1] == t * h * w, (tokens.shape, t, h, w)
    return tokens.reshape(tokens.shape[0], t, h, w, tokens.shape[-1])


def fit_rgb_pca(token_bank: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(token_bank)
    return pca


def pca_to_rgb(token_grid: np.ndarray, pca: PCA, components: tuple[int, int, int] = (0, 1, 2),
               mask: np.ndarray | None = None):
    """
    Transforms specific PCA components into DINO-style RGB heatmaps.
    Applies optional mask logic for percentile calculation and background zeroing.
    """
    flat = token_grid.reshape(-1, token_grid.shape[-1])
    all_components = pca.transform(flat)
    selected_components = all_components[:, components]
    rgb = selected_components.reshape(*token_grid.shape[:-1], 3)

    if mask is None:
        mask = np.ones(token_grid.shape[1:3], dtype=bool)

    # Normalize each channel independently using percentiles *only* from masked regions
    for i in range(3):
        channel = rgb[..., i]
        masked_channel = channel[:, mask]

        if masked_channel.size > 0:
            vmin, vmax = np.percentile(masked_channel, 1), np.percentile(masked_channel, 99)
            channel = np.clip((channel - vmin) / (vmax - vmin + 1e-6), 0, 1)
        else:
            channel = np.zeros_like(channel)

        channel[:, ~mask] = 0.0  # Zero out the background
        rgb[..., i] = channel

    return rgb


def upsample_grid(grid: np.ndarray, out_size: int):
    x = torch.from_numpy(grid).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def normalize01(x: np.ndarray):
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return x


def overlay_rgb_on_frame(frame: np.ndarray, rgb: np.ndarray, alpha: float = 0.45):
    frame = frame.astype(np.float32) / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    # Only apply overlay where rgb is non-zero (respecting our mask logic)
    mask_active = (rgb.sum(axis=-1) > 0)[..., None]

    out = frame.copy()
    out = np.where(mask_active, np.clip((1 - alpha) * frame + alpha * rgb, 0.0, 1.0), out)
    return out


def colorize_heatmap(heatmap: np.ndarray, cmap_name: str = "magma"):
    cmap = plt.get_cmap(cmap_name)
    return cmap(normalize01(heatmap))[..., :3]


def pca_component_to_heatmap(token_grid: np.ndarray, pca: PCA, component: int, mask: np.ndarray | None = None):
    """
    Extracts a single PCA component and returns a normalized heatmap.
    """
    flat = token_grid.reshape(-1, token_grid.shape[-1])
    all_components = pca.transform(flat)
    selected_component = all_components[:, component]
    heatmap = selected_component.reshape(*token_grid.shape[:-1])

    if mask is None:
        mask = np.ones(token_grid.shape[1:3], dtype=bool)

    masked_heatmap = heatmap[:, mask]
    if masked_heatmap.size > 0:
        vmin, vmax = np.percentile(masked_heatmap, 1), np.percentile(masked_heatmap, 99)
        heatmap = np.clip((heatmap - vmin) / (vmax - vmin + 1e-6), 0, 1)
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap[:, ~mask] = 0.0

    return heatmap


# %%
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULTS["CHECKPOINT_PATH"], help="Path to EchoJEPA checkpoint")
    parser.add_argument("--checkpoint-key", default=DEFAULTS["CHECKPOINT_KEY"], help="Checkpoint state-dict key")
    parser.add_argument("--model-name", default=DEFAULTS["MODEL_NAME"], help="Encoder constructor name")
    parser.add_argument("--img-size", type=int, default=DEFAULTS["IMG_SIZE"], help="Square input resolution")
    parser.add_argument("--num-frames", type=int, default=DEFAULTS["NUM_FRAMES"], help="Frames per clip")
    parser.add_argument("--tubelet-size", type=int, default=DEFAULTS["TUBELET_SIZE"], help="Temporal tubelet size")
    parser.add_argument("--patch-size", type=int, default=DEFAULTS["PATCH_SIZE"], help="Spatial patch size")
    parser.add_argument("--video-source", default=DEFAULTS["VIDEO_SOURCE"], help="MP4 path or CSV/NPY annotation file")
    parser.add_argument("--cone-mask", default=DEFAULTS["CONE_MASK"], help="Path to echocardiogram cone mask image (optional)")
    parser.add_argument("--sample-index", type=int, default=DEFAULTS["SAMPLE_INDEX"], help="Dataset sample index")
    parser.add_argument("--frame-step", type=int, default=DEFAULTS["FRAME_STEP"], help="Frame sampling stride")
    parser.add_argument("--num-clips", type=int, default=DEFAULTS["NUM_CLIPS"], help="Number of clips to sample")
    parser.add_argument("--random-clip-sampling", action="store_true", default=DEFAULTS["RANDOM_CLIP_SAMPLING"])
    parser.add_argument("--allow-clip-overlap", action="store_true", default=DEFAULTS["ALLOW_CLIP_OVERLAP"])
    parser.add_argument("--out-dir", default=DEFAULTS["OUT_DIR"], help="Directory to save generated plots")
    parser.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pca-components", type=int, nargs=3, default=(0,1,2),
                        help="Three integers indicating which PCA components to map to RGB (e.g., 1 2 3 to drop PC1)")
    parser.add_argument("--out-layer", type=int, nargs=1, default=1, help="Output layer to use")
    return parser.parse_args()


# %%
def main():
    args = parse_args()

    # Ensure output directory exists
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("device:", args.device)
    print("checkpoint:", args.checkpoint)
    print("video source:", args.video_source)
    print("output directory:", out_dir)

    encoder_ctor = vit.__dict__[args.model_name]
    encoder = encoder_ctor(
        img_size=(args.img_size, args.img_size),
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        patch_size=args.patch_size,
        uniform_power=True,
    ).to(args.device).eval()

    ckpt = load_encoder_from_checkpoint(encoder, args.checkpoint, args.checkpoint_key)
    print("encoder layers:", encoder.get_num_layers())
    print("embed dim:", encoder.embed_dim)
    print("pos embed exists:", encoder.pos_embed is not None)

    raw_clip, clip_indices, label = load_raw_clip_from_source(
        args.video_source,
        args.sample_index,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
        num_clips=args.num_clips,
        random_clip_sampling=args.random_clip_sampling,
        allow_clip_overlap=args.allow_clip_overlap,
    )
    print("raw_clip shape:", raw_clip.shape)  # T, H, W, C
    print("label:", label)
    print("clip_indices:", np.asarray(clip_indices))

    clip = prepare_clip_for_model(raw_clip, args.img_size).to(args.device)
    print("model clip shape:", tuple(clip.shape))  # B, C, T, H, W

    frame_indices = np.asarray(clip_indices)
    tubelet_frame_indices = frame_indices.reshape(-1, args.tubelet_size).mean(axis=1) if len(
        frame_indices) >= args.tubelet_size else frame_indices

    if args.out_layer is not None:
        layers_back = args.out_layer
        target_layer = encoder.get_num_layers() - 1 - layers_back
        print(f"Using output from layer {target_layer} (which is {layers_back} layers back from the top)")
        encoder.out_layers = [target_layer]

    with torch.inference_mode():
        tokens = encoder(clip)
        if args.out_layer is not None:
            tokens = tokens[0]

    print("tokens shape:", tuple(tokens.shape))  # B, N, D
    token_grid = reshape_tokens_to_grid(tokens, clip, args.patch_size, args.tubelet_size)[0].detach().cpu().numpy()
    print("token_grid shape:", token_grid.shape)  # T', H', W', D

    T_p, H_p, W_p, D = token_grid.shape
    print({"tubelets": T_p, "patch_h": H_p, "patch_w": W_p, "dim": D})

    # --- MASK LOGIC ---
    if args.cone_mask:
        mask_img = np.array(Image.open(args.cone_mask).convert('L'))
        mask_t = torch.from_numpy(mask_img).float().unsqueeze(0).unsqueeze(0)
        # Resize mask to patch resolution
        mask_down = F.interpolate(mask_t, size=(H_p, W_p), mode="nearest").squeeze().numpy()
        token_mask = mask_down > 0  # Assuming 0 is background, >0 is mask
        print(f"Loaded cone mask from {args.cone_mask}. Active tokens: {token_mask.sum()} / {H_p * W_p}")

        # Isolate masked tokens for computing PCA
        token_bank_for_fit = token_grid[:, token_mask, :].reshape(-1, D)
    else:
        token_mask = np.ones((H_p, W_p), dtype=bool)
        token_bank_for_fit = token_grid.reshape(-1, D)

    max_comp = max(args.pca_components)
    pca = fit_rgb_pca(token_bank_for_fit, n_components=max_comp + 1)

    # Pass the mask so that mapping visualizes ONLY the mask and normalizes properly
    rgb_grid = pca_to_rgb(token_grid, pca, components=args.pca_components, mask=token_mask)
    rgb_grid = gaussian_filter(rgb_grid, sigma=(0, 0.2, 0.2, 0))
    # After smoothing, we re-apply the zeroing out to prevent gaussian blur leaking outside the mask
    rgb_grid[:, ~token_mask] = 0.0

    rgb_up = upsample_grid(rgb_grid, args.img_size)
    frames = raw_clip.astype(np.float32) / 255.0
    rgb_frames = np.repeat(rgb_up, args.tubelet_size, axis=0)[: len(frames)]
    overlay_frames = np.stack([overlay_rgb_on_frame(frames[t], rgb_frames[t], alpha=0.5) for t in range(len(frames))])

    n_show = min(len(frames), 8)
    show_idx = np.linspace(0, len(frames) - 1, n_show).astype(int)

    # Plot 1: PCA Overlay and Individual Components
    fig, axes = plt.subplots(5, n_show, figsize=(2.8 * n_show, 15))
    if n_show == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]], [axes[3]], [axes[4]]])

    for col, t in enumerate(show_idx):
        axes[0, col].imshow(frames[t])
        axes[0, col].set_title(f"raw t={t}")
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay_frames[t])
        axes[1, col].set_title(f"PCA overlay t={t}")
        axes[1, col].axis("off")

        tubelet_idx = t // args.tubelet_size

        comp0_heatmap = pca_component_to_heatmap(token_grid, pca, args.pca_components[0], mask=token_mask)
        axes[2, col].imshow(comp0_heatmap[tubelet_idx], cmap="viridis")
        axes[2, col].set_title(f"PC{args.pca_components[0]} t={t}")
        axes[2, col].axis("off")

        comp1_heatmap = pca_component_to_heatmap(token_grid, pca, args.pca_components[1], mask=token_mask)
        axes[3, col].imshow(comp1_heatmap[tubelet_idx], cmap="viridis")
        axes[3, col].set_title(f"PC{args.pca_components[1]} t={t}")
        axes[3, col].axis("off")

        comp2_heatmap = pca_component_to_heatmap(token_grid, pca, args.pca_components[2], mask=token_mask)
        axes[4, col].imshow(comp2_heatmap[tubelet_idx], cmap="viridis")
        axes[4, col].set_title(f"PC{args.pca_components[2]} t={t}")
        axes[4, col].axis("off")

    fig.suptitle(
        f"PCA Overlay and Individual Components of Tubelet Tokens\nPCA components mapped to RGB: {args.pca_components}\nOutput layer: {encoder.out_layers[0] if encoder.out_layers else 'top'}",
        fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    fig.savefig(out_dir / "pca_overlay.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved PCA overlay to {out_dir / 'pca_overlay.png'}")

    # Plot 2: Change Map
    delta_grid = np.linalg.norm(np.diff(token_grid, axis=0), axis=-1) if T_p > 1 else np.zeros((1, H_p, W_p))
    delta_grid[:, ~token_mask] = 0.0  # Suppress background changes
    delta_grid = np.concatenate([delta_grid[:1], delta_grid], axis=0)
    delta_up = upsample_grid(delta_grid[..., None], args.img_size)[..., 0]
    delta_frames = np.repeat(delta_up, args.tubelet_size, axis=0)[: len(frames)]

    fig, axes = plt.subplots(2, n_show, figsize=(2.8 * n_show, 6))
    if n_show == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, t in enumerate(show_idx):
        axes[0, col].imshow(frames[t])
        axes[0, col].set_title(f"raw t={t}")
        axes[0, col].axis("off")
        axes[1, col].imshow(frames[t])

        # Only show change map heatmap in masked region
        active_map = delta_frames[t]
        active_map = np.ma.masked_where(active_map == 0, active_map)
        axes[1, col].imshow(active_map, cmap="magma", alpha=0.45)

        axes[1, col].set_title(f"change map t={t}")
        axes[1, col].axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "change_map.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved Change Map to {out_dir / 'change_map.png'}")

    # Plot 3: Tubelet Trajectory
    # Average ONLY the valid cone tokens when tracking overall representation changes
    tubelet_repr = token_grid[:, token_mask, :].mean(axis=1) if args.cone_mask else token_grid.mean(
        axis=(1, 2))  # T', D
    traj_pca = PCA(n_components=2, random_state=0).fit_transform(tubelet_repr)
    time_color = np.linspace(0, 1, len(traj_pca))

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(traj_pca[:, 0], traj_pca[:, 1], c=time_color, cmap="viridis", s=70)
    ax.plot(traj_pca[:, 0], traj_pca[:, 1], color="gray", linewidth=1, alpha=0.7)
    for i in range(len(traj_pca) - 1):
        ax.annotate("", xy=traj_pca[i + 1], xytext=traj_pca[i], arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    for i, (x, y) in enumerate(traj_pca):
        ax.text(x, y, str(int(tubelet_frame_indices[i]) if i < len(tubelet_frame_indices) else i), fontsize=8,
                alpha=0.85)
    ax.set_title("Tubelet-level embedding trajectory (Masked Region)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="normalized time")
    plt.tight_layout()
    fig.savefig(out_dir / "trajectory.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved Trajectory to {out_dir / 'trajectory.png'}")

    # Plot 4: Motion Scalar
    motion_scalar = np.linalg.norm(np.diff(tubelet_repr, axis=0), axis=1) if len(tubelet_repr) > 1 else np.zeros(1)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(len(motion_scalar)), motion_scalar, marker="o")
    ax.set_title("Representation change between consecutive tubelets (Masked Region)")
    ax.set_xlabel("tubelet index")
    ax.set_ylabel("L2 distance")
    plt.tight_layout()
    fig.savefig(out_dir / "motion_scalar.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved Motion Scalar to {out_dir / 'motion_scalar.png'}")

    return {
        "checkpoint": ckpt,
        "raw_clip": raw_clip,
        "tokens": tokens,
        "token_grid": token_grid,
        "pca": pca,
        "trajectory": traj_pca,
    }


# %%
if __name__ == "__main__":
    main()