"""
convert_mms2_to_npz.py

Convert MMs2 `.nii.gz` image and gt pairs to the BiomedParse `.npz` format.

Behavior:
- Read image and gt files from `<dataset_root>/<split>/img/` and `<dataset_root>/<split>/gt/`.
- For each case: load with nibabel, handle both 2D (LA) and 3D (SA) images,
  move slice axis to first dimension (D,H,W), remove slices with all-zero ground truth,
  scale image intensity to [0,255] (uint8), keep gt as integer labels,
  include `spacing` (voxel zooms) reordered to match axes,
  save to `<output_root>/<split>/*.npz` containing keys: `imgs`, `gts`, `spacing`.

Usage example:
    python biomedparse_datasets/convert_mms2_to_npz.py \
      --dataset-root data/MMs2 --split test --output-root data/MMs2

The script is conservative: it skips a case if all slices are background after removal.
"""
import os
import argparse
import glob
import numpy as np
import nibabel as nib


def detect_depth_axis(shape):
    # Heuristic: depth is typically the smallest axis for clinical MR/CT volumes
    # For MMs2: LA images are 2D (H, W, 1), SA images are 3D (H, W, D)
    # If 2D image or shape with one dimension = 1, return None (no depth axis)
    if len(shape) == 2:
        return None
    # If one dimension is 1, it's likely a 2D image with singleton dimension
    if min(shape) == 1:
        # Find the axis with size 1 (likely depth for 2D images)
        return int(np.argmin(shape))
    return int(np.argmin(shape))


def reorder_spacing(zooms, depth_axis):
    # Return spacing ordered as (depth, height, width)
    # For 2D images, return (1.0, height, width)
    if depth_axis is None:
        # 2D image: add depth dimension with spacing 1.0
        if len(zooms) >= 2:
            return (1.0, float(zooms[0]), float(zooms[1]))
        else:
            return (1.0, 1.0, 1.0)
    
    axes = list(range(len(zooms)))
    new_axes = [depth_axis] + [ax for ax in axes if ax != depth_axis]
    return tuple(float(zooms[ax]) for ax in new_axes)


def process_case(img_path, gt_path):
    img_nii = nib.load(img_path)
    gt_nii = nib.load(gt_path)

    imgs = img_nii.get_fdata()
    gts = gt_nii.get_fdata()

    # Handle different image dimensions
    # LA images: (H, W, 1) -> (1, H, W)
    # SA images: (H, W, D) -> (D, H, W)
    if imgs.ndim == 2:
        # Pure 2D: (H, W) -> (1, H, W)
        imgs = imgs[np.newaxis, :, :]
        gts = gts[np.newaxis, :, :]
    elif imgs.ndim == 3:
        # 3D volume: determine depth axis and move it to axis 0 -> (D,H,W)
        # For LA: (H, W, 1) -> (1, H, W)
        # For SA: (H, W, D) -> (D, H, W)
        depth_axis = detect_depth_axis(imgs.shape)
        if depth_axis is not None:
            imgs = np.moveaxis(imgs, depth_axis, 0)
            gts = np.moveaxis(gts, depth_axis, 0)
        # If depth_axis is None, it's already 2D-like, add dimension
        if imgs.ndim == 2:
            imgs = imgs[np.newaxis, :, :]
            gts = gts[np.newaxis, :, :]
    else:
        raise ValueError(f"Expect 2D or 3D volumes. Got shapes imgs={imgs.shape}, gts={gts.shape}")

    # Ensure arrays are 3D now
    if imgs.ndim != 3 or gts.ndim != 3:
        raise ValueError(f"Expect 3D volumes after processing. Got shapes imgs={imgs.shape}, gts={gts.shape}")

    # Convert types: gts -> int32, imgs -> float for scaling
    gts = gts.astype(np.int32)
    imgs = imgs.astype(np.float32)

    # Remove slices that are pure background (gts slice all zero)
    mask_nonbg = np.any(gts != 0, axis=(1, 2))  # shape (D,)
    if not np.any(mask_nonbg):
        # nothing left after removal
        return None

    imgs = imgs[mask_nonbg]
    gts = gts[mask_nonbg]

    # Scale image intensities to [0,255] and cast to uint8
    mn = float(np.min(imgs))
    mx = float(np.max(imgs))
    if mx > mn:
        imgs = (imgs - mn) / (mx - mn)
    else:
        imgs = np.zeros_like(imgs)
    imgs = np.clip((imgs * 255.0), 0, 255).astype(np.uint8)

    # Reorder spacing to (D,H,W)
    zooms = img_nii.header.get_zooms()
    # If header has >3 entries, take first 3
    if len(zooms) >= 3:
        zooms3 = zooms[:3]
    elif len(zooms) == 2:
        # 2D image: add depth spacing
        zooms3 = (1.0, zooms[0], zooms[1])
    else:
        zooms3 = zooms + (1.0,) * (3 - len(zooms))
    
    # Determine depth axis for spacing reordering
    # If only 1 slice, depth_axis is None
    depth_axis = detect_depth_axis(imgs.shape) if imgs.shape[0] > 1 else None
    spacing = reorder_spacing(zooms3, depth_axis)

    return {"imgs": imgs, "gts": gts.astype(np.int32), "spacing": spacing}


def run(dataset_root, split, output_root, pattern="*.nii.gz"):
    img_dir = os.path.join(dataset_root, split, "img")
    gt_dir = os.path.join(dataset_root, split, "gt")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    out_dir = os.path.join(output_root, split)
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_dir, pattern)))
    if len(img_files) == 0:
        print(f"No image files found in {img_dir} matching {pattern}")
        return

    n_saved = 0
    n_skipped = 0

    for img_path in img_files:
        base = os.path.basename(img_path)
        # Remove .nii.gz extension to get base name
        name = os.path.splitext(os.path.splitext(base)[0])[0]
        
        # MMs2 naming: 261_LA_ED.nii.gz -> 261_LA_ED_gt.nii.gz
        gt_path = os.path.join(gt_dir, f"{name}_gt.nii.gz")
        
        if not os.path.exists(gt_path):
            print(f"GT not found for {base} (expected {os.path.basename(gt_path)}), skipping")
            n_skipped += 1
            continue

        try:
            processed = process_case(img_path, gt_path)
        except Exception as e:
            print(f"Error processing {base}: {e}")
            n_skipped += 1
            continue

        if processed is None:
            print(f"All slices are background after filtering for {base}, skipping")
            n_skipped += 1
            continue

        out_path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(out_path, imgs=processed["imgs"], gts=processed["gts"], spacing=processed["spacing"])
        print(f"Saved {out_path}: imgs shape {processed['imgs'].shape}, gts shape {processed['gts'].shape}")
        n_saved += 1

    print(f"Done. Saved {n_saved} files, skipped {n_skipped} files.")


def make_prompts_template(output_data_root, dataset_name="MMs2"):
    # create a minimal class_prompts.json template for MMs2
    # MMs2 typically has: 0=background, 1=Left Ventricle Cavity, 2=Myocardium, 3=Right Ventricle
    # Note: MMs2 labels may vary, adjust based on actual dataset documentation
    import json
    
    # Check if class_prompts.json already exists
    prompts_path = os.path.join(output_data_root, "class_prompts.json")
    prompts = {}
    if os.path.exists(prompts_path):
        with open(prompts_path, "r") as f:
            prompts = json.load(f)
    
    # Add or update MMs2 prompts
    # Common cardiac MR segmentation labels:
    # 1 = Left Ventricle Cavity, 2 = Myocardium, 3 = Right Ventricle
    prompts[dataset_name] = {
        "1": ["Left ventricle cavity", "Left ventricle cavity in cardiac MR"],
        "2": ["Myocardium", "Myocardium in cardiac MR"],
        "3": ["Right ventricle", "Right ventricle in cardiac MR"],
        "instance_label": 0
    }
    
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Wrote/updated class_prompts.json at {prompts_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MMs2 .nii.gz files to BiomedParse .npz format")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to dataset root, e.g. data/MMs2")
    parser.add_argument("--split", type=str, default="test", help="Which split to process: train/test")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Output root folder; defaults to dataset root")
    parser.add_argument("--write-prompts", action="store_true", 
                        help="Write/update class_prompts.json template in the parent data folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_root = args.output_root if args.output_root is not None else args.dataset_root
    run(args.dataset_root, args.split, out_root)
    if args.write_prompts:
        # place prompts in parent data folder (one level up from dataset_root)
        data_root = os.path.dirname(args.dataset_root)
        make_prompts_template(data_root, os.path.basename(args.dataset_root))

