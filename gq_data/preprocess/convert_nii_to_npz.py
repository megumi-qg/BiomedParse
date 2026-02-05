"""
convert_nii_to_npz.py

通用脚本：将 nii.gz 格式的医学图像转换为 BiomedParse 期望的 npz 格式。

输出格式：
- 每个 npz 文件包含: imgs (D,H,W), gts (D,H,W), spacing (3,)
- imgs 强度范围 [0, 255], dtype uint8

强度归一化策略：
- CT 图像：根据解剖部位使用窗宽窗位 (Hounsfield -> [0,255])
  - soft_tissues: W:400, L:40
  - lung: W:1500, L:-160
  - brain: W:80, L:40
  - bone: W:1800, L:400
- MRI/其他：裁剪到 0.5-99.5 百分位，再缩放到 [0,255]
- 若原始强度已在 [0,255]，则跳过预处理

支持数据集：ACDC (MRI), BTCV (CT-soft_tissues), PROMISE12 (MRI)

Usage:
    # ACDC 示例
    python gq_data/preprocess/convert_nii_to_npz.py \
        --dataset acdc \
        --img-dir /home/gaoqi/dataset/using/acdc1/test/img \
        --gt-dir /home/gaoqi/dataset/using/acdc1/test/gt \
        --output-dir /home/gaoqi/BiomedParse/gq_data/acdc/test

    # BTCV 示例
    python gq_data/preprocess/convert_nii_to_npz.py \
        --dataset btcv \
        --img-dir /path/to/btcv/imagesTr \
        --gt-dir /path/to/btcv/labelsTr \
        --output-dir /home/gaoqi/BiomedParse/gq_data/btcv/train

    # PROMISE12 示例
    python gq_data/preprocess/convert_nii_to_npz.py \
        --dataset promise12 \
        --img-dir /path/to/promise12/img \
        --gt-dir /path/to/promise12/gt \
        --output-dir /home/gaoqi/BiomedParse/gq_data/promise12/train
"""
import os
import argparse
import glob
import re
import numpy as np
import nibabel as nib


# CT 窗宽窗位预设 (Window, Level)，单位 HU
CT_WINDOW_PRESETS = {
    "soft_tissues": (400, 40),
    "lung": (1500, -160),
    "brain": (80, 40),
    "bone": (1800, 400),
}


def detect_depth_axis(shape):
    """深度轴通常为最小维度（临床 MR/CT 惯例）"""
    if len(shape) == 2:
        return None
    if min(shape) == 1:
        return int(np.argmin(shape))
    return int(np.argmin(shape))


def reorder_spacing(zooms, depth_axis):
    """返回 (depth, height, width) 顺序的 spacing"""
    if depth_axis is None:
        if len(zooms) >= 2:
            return (1.0, float(zooms[0]), float(zooms[1]))
        return (1.0, 1.0, 1.0)
    axes = list(range(len(zooms)))
    new_axes = [depth_axis] + [ax for ax in axes if ax != depth_axis]
    return tuple(float(zooms[ax]) for ax in new_axes)


def normalize_ct(imgs: np.ndarray, window: str = "soft_tissues") -> np.ndarray:
    """
    CT 图像：使用窗宽窗位将 HU 值归一化到 [0, 255]
    """
    w, l = CT_WINDOW_PRESETS[window]
    # HU 范围: [L - W/2, L + W/2]
    low = l - w / 2
    high = l + w / 2
    imgs = np.clip(imgs.astype(np.float32), low, high)
    imgs = (imgs - low) / (high - low) * 255.0
    return np.clip(imgs, 0, 255).astype(np.uint8)


def normalize_non_ct(imgs: np.ndarray) -> np.ndarray:
    """
    MRI/其他：裁剪到 0.5-99.5 百分位，缩放到 [0, 255]
    若已在 [0,255] 则跳过
    """
    imgs = imgs.astype(np.float32)
    mn, mx = float(np.min(imgs)), float(np.max(imgs))

    # 若已在 [0,255] 且为整数，直接转为 uint8
    if mn >= 0 and mx <= 255 and np.allclose(imgs, imgs.astype(np.int32)):
        return np.clip(imgs.round(), 0, 255).astype(np.uint8)

    # 百分位裁剪
    p_low = np.percentile(imgs, 0.5)
    p_high = np.percentile(imgs, 99.5)
    if p_high > p_low:
        imgs = np.clip(imgs, p_low, p_high)
        imgs = (imgs - p_low) / (p_high - p_low) * 255.0
    else:
        imgs = np.zeros_like(imgs)
    return np.clip(imgs, 0, 255).astype(np.uint8)


def process_case(
    img_path: str,
    gt_path: str,
    modality: str = "mri",
    ct_window: str = "soft_tissues",
    remove_empty_slices: bool = True,
) -> dict | None:
    """
    处理单例：加载 nii.gz，归一化强度，输出 (imgs, gts, spacing)
    """
    img_nii = nib.load(img_path)
    gt_nii = nib.load(gt_path)

    imgs = img_nii.get_fdata()
    gts = gt_nii.get_fdata()

    # 统一为 3D (D, H, W)，并记录原始 depth 轴（用于后续 spacing 重排）
    original_depth_axis = None
    if imgs.ndim == 2:
        imgs = imgs[np.newaxis, :, :]
        gts = gts[np.newaxis, :, :]
    elif imgs.ndim == 3:
        original_depth_axis = detect_depth_axis(imgs.shape)
        if original_depth_axis is not None:
            imgs = np.moveaxis(imgs, original_depth_axis, 0)
            gts = np.moveaxis(gts, original_depth_axis, 0)
        if imgs.ndim == 2:
            imgs = imgs[np.newaxis, :, :]
            gts = gts[np.newaxis, :, :]
    else:
        raise ValueError(f"Expected 2D or 3D, got imgs={imgs.shape}, gts={gts.shape}")

    if imgs.ndim != 3 or gts.ndim != 3:
        raise ValueError(f"After processing: imgs={imgs.shape}, gts={gts.shape}")

    gts = gts.astype(np.int32)

    # 可选：移除全背景切片
    if remove_empty_slices:
        mask_nonbg = np.any(gts != 0, axis=(1, 2))
        if not np.any(mask_nonbg):
            return None
        imgs = imgs[mask_nonbg]
        gts = gts[mask_nonbg]

    # 强度归一化
    if modality.lower() == "ct":
        imgs = normalize_ct(imgs, ct_window)
    else:
        imgs = normalize_non_ct(imgs)

    # Spacing：zooms 为原始 NIfTI 轴顺序，需用 original_depth_axis 重排为 (D,H,W)
    zooms = img_nii.header.get_zooms()
    if len(zooms) >= 3:
        zooms3 = zooms[:3]
    elif len(zooms) == 2:
        zooms3 = (1.0, zooms[0], zooms[1])
        original_depth_axis = None  # 2D 无 depth
    else:
        zooms3 = tuple(zooms) + (1.0,) * (3 - len(zooms))

    spacing = reorder_spacing(zooms3, original_depth_axis)

    return {"imgs": imgs, "gts": gts, "spacing": np.array(spacing, dtype=np.float64)}


def get_img_gt_pairs(img_dir: str, gt_dir: str, dataset: str) -> list[tuple[str, str]]:
    """
    根据数据集类型，匹配 img 与 gt 文件对。
    返回 [(img_path, gt_path), ...]
    """
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    if not img_files:
        raise FileNotFoundError(f"No .nii.gz files in {img_dir}")

    pairs = []
    for img_path in img_files:
        base = os.path.basename(img_path)
        name = base.replace(".nii.gz", "")

        if dataset == "acdc":
            # ACDC: patient001_frame01.nii.gz -> patient001_frame01_gt.nii.gz
            gt_name = f"{name}_gt.nii.gz"
        elif dataset == "btcv":
            # BTCV (nnU-Net): imagesTr/case_00001_00000.nii.gz -> labelsTr/case_00001.nii.gz
            match = re.match(r"^(.+)_\d{5}\.nii\.gz$", base)
            if match:
                gt_name = match.group(1) + ".nii.gz"
            else:
                gt_name = base
        elif dataset == "promise12":
            # PROMISE12: 通常同名或 caseXX.nii.gz -> caseXX_gt.nii.gz
            if os.path.exists(os.path.join(gt_dir, f"{name}_gt.nii.gz")):
                gt_name = f"{name}_gt.nii.gz"
            else:
                gt_name = base
        else:
            gt_name = f"{name}_gt.nii.gz"

        gt_path = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path) and gt_name != base:
            # 回退：尝试同名文件
            gt_path = os.path.join(gt_dir, base)
            gt_name = base
        if os.path.exists(gt_path):
            pairs.append((img_path, gt_path))
        else:
            print(f"  [skip] GT not found: {gt_name}")

    return pairs


def run(
    img_dir: str,
    gt_dir: str,
    output_dir: str,
    dataset: str = "acdc",
    modality: str | None = None,
    ct_window: str = "soft_tissues",
    remove_empty_slices: bool = True,
):
    """主流程"""
    if modality is None:
        modality = "ct" if dataset == "btcv" else "mri"

    os.makedirs(output_dir, exist_ok=True)

    pairs = get_img_gt_pairs(img_dir, gt_dir, dataset)
    if not pairs:
        print("No valid img-gt pairs found.")
        return

    print(f"Processing {len(pairs)} cases (modality={modality}, ct_window={ct_window})")

    n_saved = 0
    n_skipped = 0

    for img_path, gt_path in pairs:
        base = os.path.basename(img_path)
        name = base.replace(".nii.gz", "")

        try:
            processed = process_case(
                img_path,
                gt_path,
                modality=modality,
                ct_window=ct_window,
                remove_empty_slices=remove_empty_slices,
            )
        except Exception as e:
            print(f"  [error] {base}: {e}")
            n_skipped += 1
            continue

        if processed is None:
            print(f"  [skip] {base}: all slices background")
            n_skipped += 1
            continue

        out_path = os.path.join(output_dir, f"{name}.npz")
        np.savez_compressed(
            out_path,
            imgs=processed["imgs"],
            gts=processed["gts"],
            spacing=processed["spacing"],
        )
        print(f"  [saved] {out_path}: imgs {processed['imgs'].shape}, gts {processed['gts'].shape}")
        n_saved += 1

    print(f"Done. Saved {n_saved}, skipped {n_skipped}.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert nii.gz to BiomedParse npz format (imgs, gts, spacing)"
    )
    p.add_argument("--dataset", type=str, default="acdc", choices=["acdc", "btcv", "promise12"],
                   help="Dataset preset for img-gt pairing")
    p.add_argument("--img-dir", type=str, required=True, help="Directory of input images (.nii.gz)")
    p.add_argument("--gt-dir", type=str, required=True, help="Directory of ground truth (.nii.gz)")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for .npz files")
    p.add_argument("--modality", type=str, default=None, choices=["ct", "mri"],
                   help="Override modality (default: ct for btcv, mri for others)")
    p.add_argument("--ct-window", type=str, default="soft_tissues",
                   choices=list(CT_WINDOW_PRESETS.keys()),
                   help="CT window preset (only for modality=ct)")
    p.add_argument("--keep-empty-slices", action="store_true",
                   help="Do not remove slices with all-zero gt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        img_dir=args.img_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        modality=args.modality,
        ct_window=args.ct_window,
        remove_empty_slices=not args.keep_empty_slices,
    )
