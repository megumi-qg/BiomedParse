"""
convert_acdc_to_npz.py

Convert ACDC `.nii.gz` image and gt pairs to the BiomedParse `.npz` format.

Behavior:
- Read image and gt files from `<dataset_root>/<split>/img/` and `<dataset_root>/<split>/gt/`.
- For each case: load with nibabel, move slice axis to first dimension (D,H,W),
  remove slices with all-zero ground truth, scale image intensity to [0,255] (uint8),
  keep gt as integer labels, include `spacing` (voxel zooms) reordered to match axes,
  save to `<output_root>/<split>/*.npz` containing keys: `imgs`, `gts`, `spacing`.

Usage example:
    python biomedparse_datasets/convert_acdc_to_npz.py \
      --dataset-root data/ACDC --split test --output-root data/ACDC

The script is conservative: it skips a case if all slices are background after removal.
"""
import os
import argparse
import glob
import numpy as np
import nibabel as nib


def detect_depth_axis(shape):
    # Heuristic: depth is typically the smallest axis for clinical MR/CT volumes
    return int(np.argmin(shape))


def reorder_spacing(zooms, depth_axis):
    # Return spacing ordered as (depth, height, width)
    axes = list(range(len(zooms)))
    new_axes = [depth_axis] + [ax for ax in axes if ax != depth_axis]
    return tuple(float(zooms[ax]) for ax in new_axes)


def process_case(img_path, gt_path):
    img_nii = nib.load(img_path)
    gt_nii = nib.load(gt_path)

    imgs = img_nii.get_fdata()
    gts = gt_nii.get_fdata()

    # Ensure arrays are 3D
    if imgs.ndim != 3 or gts.ndim != 3:
        raise ValueError(f"Expect 3D volumes. Got shapes imgs={imgs.shape}, gts={gts.shape}")

    # Determine depth axis (heuristic) and move it to axis 0 -> (D,H,W)
    depth_axis = detect_depth_axis(imgs.shape)
    imgs = np.moveaxis(imgs, depth_axis, 0)
    gts = np.moveaxis(gts, depth_axis, 0)

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
    else:
        zooms3 = zooms + (1.0,) * (3 - len(zooms))
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
        name = os.path.splitext(os.path.splitext(base)[0])[0]
        # Try to find matching GT file. common pattern: same basename or basename + '_gt'
        gt_path = os.path.join(gt_dir, base)
        if not os.path.exists(gt_path):
            base_no_ext = name
            alt1 = os.path.join(gt_dir, f"{base_no_ext}_gt.nii.gz")
            alt2 = os.path.join(gt_dir, f"{base_no_ext}_gt.nii")
            found = None
            if os.path.exists(alt1):
                found = alt1
            elif os.path.exists(alt2):
                found = alt2
            else:
                # search for any file in gt_dir containing the base and '_gt'
                for p in sorted(glob.glob(os.path.join(gt_dir, "*"))):
                    fname = os.path.basename(p)
                    if base_no_ext in fname and "_gt" in fname and fname.endswith((".nii.gz", ".nii")):
                        found = p
                        break

            if found is None:
                print(f"GT not found for {base}, skipping")
                n_skipped += 1
                continue
            else:
                gt_path = found

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


def make_prompts_template(output_data_root, dataset_name="ACDC"):
    # create a minimal class_prompts.json template for ACDC
    import json
    prompts = {
        dataset_name: {
            "1": ["Right ventricle", "Right ventricle in cardiac MR"],
            "2": ["Myocardium", "Myocardium in cardiac MR"],
            "3": ["Left ventricle", "Left ventricle in cardiac MR"],
            "instance_label": 0
        }
    }
    out_path = os.path.join(output_data_root, "class_prompts.json")
    with open(out_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Wrote class_prompts.json to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ACDC .nii.gz files to BiomedParse .npz format")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to dataset root, e.g. data/ACDC")
    parser.add_argument("--split", type=str, default="test", help="Which split to process: train/test")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Output root folder; defaults to dataset root")
    parser.add_argument("--write-prompts", action="store_true", help="Write a minimal class_prompts.json template to the parent data folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_root = args.output_root if args.output_root is not None else args.dataset_root
    run(args.dataset_root, args.split, out_root)
    if args.write_prompts:
        # place prompts in parent data folder (one level up from dataset_root)
        data_root = os.path.dirname(args.dataset_root)
        make_prompts_template(data_root, os.path.basename(args.dataset_root))
