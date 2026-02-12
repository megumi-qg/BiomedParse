"""
BiomedParse v1 模型评估脚本（2D）

本脚本用于评估 BiomedParse v1 模型在 2D 医学图像分割数据集（如 CAMUS）上的性能。
支持计算 Dice、IoU、HD95、NSD 等指标。

注意：推理与评估已解耦。请先使用 inference_v1.py 进行推理，将预测结果保存到输出文件夹，
再使用本脚本加载预测结果进行评估。

使用方法：
    python gq_scripts/evaluate_v1.py --data-root <数据目录> --dataset-name <数据集名称> --inference-dir <推理结果目录> [--output-name <输出路径>]

参数说明：
    --data-root: 测试数据目录，包含 .npz 格式的 ground truth
    --dataset-name: 数据集名称，需在 class_prompts.json 中存在
    --inference-dir: 推理结果目录（由 inference_v1.py 生成）
    --output-name: 输出 JSON 路径，可为完整路径或仅文件名

示例：
    python gq_scripts/evaluate_v1.py --data-root data/CAMUS/test --dataset-name CAMUS --inference-dir inference_results/CAMUS
    python gq_scripts/evaluate_v1.py --data-root data/CAMUS/test --dataset-name CAMUS --inference-dir inference_results/CAMUS --output-name gq_data/camus/eval_results/camus_v1.json
"""

import os
import sys
import glob
import json
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_prompts(data_root, dataset_name="CAMUS"):
    """
    Load prompts from class_prompts.json file.
    
    Args:
        data_root: Root directory of the dataset
        dataset_name: Name of the dataset in class_prompts.json
    
    Returns:
        ids: List of class IDs
        prompts_dict: Dictionary mapping class ID to list of prompts
    """
    # Search upward from data_root for class_prompts.json (support file colocated with data/)
    cur = os.path.abspath(data_root)
    found = None
    # climb up to repo root (or max 5 levels)
    for _ in range(6):
        candidate = os.path.join(cur, "class_prompts.json")
        if os.path.exists(candidate):
            found = candidate
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    if found is None:
        raise FileNotFoundError(f"class_prompts.json not found near {data_root}; looked upward")

    with open(found, "r") as f:
        cj = json.load(f)
    if dataset_name not in cj:
        raise KeyError(f"Dataset {dataset_name} not in class_prompts.json")
    ds = cj[dataset_name]
    ids = [int(k) for k in ds.keys() if k.isdigit()]
    ids.sort()
    
    # Build prompts dictionary
    prompts_dict = {}
    for i in ids:
        prompts_dict[i] = ds[str(i)]
    
    return ids, prompts_dict


def compute_hd95_2d(pred_mask, gt_mask, spacing):
    """
    Compute 95th percentile Hausdorff Distance between two binary 2D masks.
    
    Args:
        pred_mask: np.ndarray, binary mask (H,W) or (D,H,W) with D=1
        gt_mask: np.ndarray, binary mask (H,W) or (D,H,W) with D=1
        spacing: tuple of 2 or 3 floats, voxel spacing in (H,W) or (D,H,W) order
    
    Returns:
        float: HD95 value in mm, or np.inf if one mask is empty
    """
    # Handle 3D format with D=1: squeeze to 2D
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]  # (H, W)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]  # (H, W)
    
    # Extract 2D spacing (H, W)
    if len(spacing) == 3:
        spacing_2d = (spacing[1], spacing[2])  # (H, W)
    else:
        spacing_2d = spacing[:2]  # (H, W)
    
    # If both masks are empty, return 0
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    
    # If one mask is empty, return inf
    if not pred_mask.any() or not gt_mask.any():
        return np.inf
    
    # Get surface points (boundary pixels) using binary erosion
    # A pixel is on the surface if it's in the mask but its erosion is not
    structure = np.ones((3, 3))  # 2D connectivity
    pred_eroded = ndimage.binary_erosion(pred_mask, structure=structure)
    gt_eroded = ndimage.binary_erosion(gt_mask, structure=structure)
    
    pred_surface = pred_mask & (~pred_eroded)
    gt_surface = gt_mask & (~gt_eroded)
    
    # If no surface points found, return 0 (masks are identical)
    if not pred_surface.any() and not gt_surface.any():
        return 0.0
    
    # Compute distance transform from each surface
    # Distance from GT surface to prediction boundary
    if gt_surface.any():
        # Distance transform: distance from each point to nearest point in ~pred_mask (i.e., to pred boundary)
        dist_gt_to_pred = ndimage.distance_transform_edt(~pred_mask, sampling=spacing_2d)
        distances_gt_to_pred = dist_gt_to_pred[gt_surface]
    else:
        distances_gt_to_pred = np.array([np.inf])
    
    # Distance from prediction surface to GT boundary
    if pred_surface.any():
        dist_pred_to_gt = ndimage.distance_transform_edt(~gt_mask, sampling=spacing_2d)
        distances_pred_to_gt = dist_pred_to_gt[pred_surface]
    else:
        distances_pred_to_gt = np.array([np.inf])
    
    # Compute 95th percentile for both directions
    if len(distances_gt_to_pred) > 0 and not np.all(np.isinf(distances_gt_to_pred)):
        valid_distances = distances_gt_to_pred[~np.isinf(distances_gt_to_pred)]
        if len(valid_distances) > 0:
            hd95_gt_to_pred = np.percentile(valid_distances, 95)
        else:
            hd95_gt_to_pred = np.inf
    else:
        hd95_gt_to_pred = np.inf
    
    if len(distances_pred_to_gt) > 0 and not np.all(np.isinf(distances_pred_to_gt)):
        valid_distances = distances_pred_to_gt[~np.isinf(distances_pred_to_gt)]
        if len(valid_distances) > 0:
            hd95_pred_to_gt = np.percentile(valid_distances, 95)
        else:
            hd95_pred_to_gt = np.inf
    else:
        hd95_pred_to_gt = np.inf
    
    # HD95 is the maximum of the two 95th percentiles
    hd95 = max(hd95_gt_to_pred, hd95_pred_to_gt)
    
    # If both are inf, return inf
    if np.isinf(hd95):
        return np.inf
    
    return float(hd95)


def compute_nsd_2d(pred_mask, gt_mask, spacing, tolerance=2.0):
    """
    Compute Normalized Surface Distance (NSD) at tolerance between two binary 2D masks.
    NSD is the Surface Dice at tolerance, which measures the surface overlap within tolerance.
    
    Args:
        pred_mask: np.ndarray, binary mask (H,W) or (D,H,W) with D=1
        gt_mask: np.ndarray, binary mask (H,W) or (D,H,W) with D=1
        spacing: tuple of 2 or 3 floats, voxel spacing in (H,W) or (D,H,W) order
        tolerance: float, tolerance in mm (default 2.0)
    
    Returns:
        float: NSD value (0-1), or np.nan if one mask is empty
    """
    # Handle 3D format with D=1: squeeze to 2D
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]  # (H, W)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]  # (H, W)
    
    # Extract 2D spacing (H, W)
    if len(spacing) == 3:
        spacing_2d = (spacing[1], spacing[2])  # (H, W)
    else:
        spacing_2d = spacing[:2]  # (H, W)
    
    # If both masks are empty, return 1.0 (perfect match)
    if not pred_mask.any() and not gt_mask.any():
        return 1.0
    
    # If one mask is empty, return 0.0 (no overlap)
    if not pred_mask.any() or not gt_mask.any():
        return 0.0
    
    try:
        # Get surface points (boundary pixels)
        structure = np.ones((3, 3))  # 2D connectivity
        pred_eroded = ndimage.binary_erosion(pred_mask, structure=structure)
        gt_eroded = ndimage.binary_erosion(gt_mask, structure=structure)
        
        pred_surface = pred_mask & (~pred_eroded)
        gt_surface = gt_mask & (~gt_eroded)
        
        # If no surface points found, return 1.0 (masks are identical)
        if not pred_surface.any() and not gt_surface.any():
            return 1.0
        
        # Compute distance transform from each surface
        # Distance from GT surface to prediction boundary
        if gt_surface.any():
            dist_gt_to_pred = ndimage.distance_transform_edt(~pred_mask, sampling=spacing_2d)
            distances_gt_to_pred = dist_gt_to_pred[gt_surface]
        else:
            distances_gt_to_pred = np.array([np.inf])
        
        # Distance from prediction surface to GT boundary
        if pred_surface.any():
            dist_pred_to_gt = ndimage.distance_transform_edt(~gt_mask, sampling=spacing_2d)
            distances_pred_to_gt = dist_pred_to_gt[pred_surface]
        else:
            distances_pred_to_gt = np.array([np.inf])
        
        # Count surface points within tolerance
        # GT surface points within tolerance of pred boundary
        if len(distances_gt_to_pred) > 0:
            valid_distances_gt = distances_gt_to_pred[~np.isinf(distances_gt_to_pred)]
            if len(valid_distances_gt) > 0:
                gt_within_tolerance = np.sum(valid_distances_gt <= tolerance)
                gt_total = len(valid_distances_gt)
            else:
                gt_within_tolerance = 0
                gt_total = 0
        else:
            gt_within_tolerance = 0
            gt_total = 0
        
        # Pred surface points within tolerance of GT boundary
        if len(distances_pred_to_gt) > 0:
            valid_distances_pred = distances_pred_to_gt[~np.isinf(distances_pred_to_gt)]
            if len(valid_distances_pred) > 0:
                pred_within_tolerance = np.sum(valid_distances_pred <= tolerance)
                pred_total = len(valid_distances_pred)
            else:
                pred_within_tolerance = 0
                pred_total = 0
        else:
            pred_within_tolerance = 0
            pred_total = 0
        
        # NSD is the average of the two ratios
        if gt_total + pred_total == 0:
            return 1.0
        
        nsd = (gt_within_tolerance + pred_within_tolerance) / (gt_total + pred_total)
        return float(nsd)
    except Exception as e:
        # If computation fails, return nan
        print(f"Warning: NSD computation failed: {e}")
        return np.nan


def compute_metrics(pred, gt, class_ids, spacing=None, nsd_tolerance=2.0):
    """
    Compute evaluation metrics for 2D segmentation.
    
    Args:
        pred: np.ndarray (D,H,W) or (H,W) integer labels
        gt: np.ndarray (D,H,W) or (H,W) integer labels
        class_ids: List of class IDs to evaluate
        spacing: tuple of 2 or 3 floats, voxel spacing, optional
        nsd_tolerance: float, tolerance for NSD in mm (default 2.0)
    
    Returns:
        per_class: Dictionary mapping class ID to metrics dict
        overall_dice: float, overall Dice coefficient
        overall_iou: float, overall IoU
        overall_hd95: float, overall HD95
        overall_nsd: float, overall NSD
    """
    # Handle 2D case: ensure we have (D,H,W) format with D=1
    if pred.ndim == 2:
        pred = pred[np.newaxis, :, :]
    if gt.ndim == 2:
        gt = gt[np.newaxis, :, :]
    
    per_class = {}
    total_inter = 0
    total_union = 0
    total_gt_sum = 0
    total_pred_sum = 0
    total_hd95_list = []  # for overall HD95 calculation
    total_nsd_list = []  # for overall NSD calculation
    
    for c in class_ids:
        pred_mask = (pred == c)
        gt_mask = (gt == c)
        inter = int((pred_mask & gt_mask).sum())
        pred_sum = int(pred_mask.sum())
        gt_sum = int(gt_mask.sum())
        union = int((pred_mask | gt_mask).sum())
        
        # dice (DSC)
        if pred_sum + gt_sum == 0:
            dice = 1.0
        else:
            dice = 2.0 * inter / (pred_sum + gt_sum)
        
        # iou
        if union == 0:
            iou = 1.0
        else:
            iou = inter / union
        
        # hd95
        if spacing is not None:
            hd95 = compute_hd95_2d(pred_mask, gt_mask, spacing)
            total_hd95_list.append(hd95)
        else:
            hd95 = np.nan
        
        # nsd
        if spacing is not None:
            nsd = compute_nsd_2d(pred_mask, gt_mask, spacing, tolerance=nsd_tolerance)
            if not np.isnan(nsd):
                total_nsd_list.append(nsd)
        else:
            nsd = np.nan
        
        per_class[c] = {
            "dice": float(dice), 
            "iou": float(iou), 
            "hd95": float(hd95) if spacing is not None else np.nan,
            "nsd": float(nsd) if spacing is not None else np.nan,
            "pred_sum": pred_sum, 
            "gt_sum": gt_sum
        }
        total_inter += inter
        total_union += union
        total_gt_sum += gt_sum
        total_pred_sum += pred_sum

    # overall micro metrics (considering only listed classes)
    if total_pred_sum + total_gt_sum == 0:
        overall_dice = 1.0
    else:
        overall_dice = 2.0 * total_inter / (total_pred_sum + total_gt_sum)
    if total_union == 0:
        overall_iou = 1.0
    else:
        overall_iou = total_inter / total_union
    
    # overall HD95: average of per-class HD95 values
    if spacing is not None and len(total_hd95_list) > 0:
        # Filter out inf values for mean calculation
        valid_hd95 = [h for h in total_hd95_list if not np.isinf(h)]
        if len(valid_hd95) > 0:
            overall_hd95 = float(np.mean(valid_hd95))
        else:
            overall_hd95 = np.inf
    else:
        overall_hd95 = np.nan
    
    # overall NSD: average of per-class NSD values
    if spacing is not None and len(total_nsd_list) > 0:
        valid_nsd = [n for n in total_nsd_list if not np.isnan(n)]
        if len(valid_nsd) > 0:
            overall_nsd = float(np.mean(valid_nsd))
        else:
            overall_nsd = np.nan
    else:
        overall_nsd = np.nan

    return per_class, float(overall_dice), float(overall_iou), float(overall_hd95), float(overall_nsd)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate BiomedParse v1 predictions on 2D dataset")
    parser.add_argument("--data-root", type=str, default="data/CAMUS/test",
                        help="Path to dataset directory containing .npz files (for ground truth)")
    parser.add_argument("--dataset-name", type=str, default="CAMUS",
                        help="Dataset name in class_prompts.json")
    parser.add_argument("--inference-dir", type=str, default="inference_results",
                        help="Directory containing inference results from inference_v1.py")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output JSON path; can be full path or filename (saved under data/)")
    args = parser.parse_args()

    data_root = args.data_root
    inference_dir = args.inference_dir
    dataset_name = args.dataset_name
    output_name = args.output_name if args.output_name else f"{dataset_name}_eval_summary.json"

    ids, _ = load_prompts(data_root, dataset_name=dataset_name)
    print("Using class ids:", ids)

    gt_files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
    if len(gt_files) == 0:
        print("No .npz files found in", data_root)
        return

    if not os.path.exists(inference_dir):
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    all_per_class = {c: {"dice": [], "iou": [], "hd95": [], "nsd": []} for c in ids}
    overall_dice_list = []
    overall_iou_list = []
    overall_hd95_list = []
    overall_nsd_list = []
    per_patient = {}

    for gt_file in gt_files:
        name = os.path.basename(gt_file)
        gt_data = np.load(gt_file, allow_pickle=True)
        gts = gt_data["gts"]
        if gts.ndim == 2:
            gts = gts[np.newaxis, :, :]

        spacing = None
        if "spacing" in gt_data:
            spacing_val = gt_data["spacing"]
            if isinstance(spacing_val, (np.ndarray, list, tuple)):
                if len(spacing_val) == 2:
                    spacing = (1.0, spacing_val[0], spacing_val[1])
                else:
                    spacing = tuple(spacing_val[:3])
            else:
                spacing = (1.0, spacing_val, spacing_val)

        pred_file = os.path.join(inference_dir, name)
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file not found: {pred_file}, skipping {name}")
            continue

        pred_data = np.load(pred_file, allow_pickle=True)
        if "pred_mask" not in pred_data:
            print(f"Warning: 'pred_mask' not found in {pred_file}, skipping {name}")
            continue

        pred = pred_data["pred_mask"]
        if pred.ndim == 2:
            pred = pred[np.newaxis, :, :]

        if spacing is None and "spacing" in pred_data:
            spacing_val = pred_data["spacing"]
            if isinstance(spacing_val, (np.ndarray, list, tuple)):
                if len(spacing_val) == 2:
                    spacing = (1.0, spacing_val[0], spacing_val[1])
                else:
                    spacing = tuple(spacing_val[:3])
            else:
                spacing = (1.0, spacing_val, spacing_val)

        gt = gts
        per_class, overall_dice, overall_iou, overall_hd95, overall_nsd = compute_metrics(
            pred, gt, ids, spacing=spacing
        )

        for c in ids:
            all_per_class[c]["dice"].append(per_class[c]["dice"])
            all_per_class[c]["iou"].append(per_class[c]["iou"])
            if not (np.isnan(per_class[c]["hd95"]) or np.isinf(per_class[c]["hd95"])):
                all_per_class[c]["hd95"].append(per_class[c]["hd95"])
            if not np.isnan(per_class[c]["nsd"]):
                all_per_class[c]["nsd"].append(per_class[c]["nsd"])
        overall_dice_list.append(overall_dice)
        overall_iou_list.append(overall_iou)
        if not (np.isnan(overall_hd95) or np.isinf(overall_hd95)):
            overall_hd95_list.append(overall_hd95)
        if not np.isnan(overall_nsd):
            overall_nsd_list.append(overall_nsd)

        per_patient[name] = {
            "per_class": {
                str(c): {
                    "dice": per_class[c]["dice"],
                    "iou": per_class[c]["iou"],
                    "hd95": per_class[c]["hd95"],
                    "nsd": per_class[c]["nsd"],
                    "pred_sum": per_class[c]["pred_sum"],
                    "gt_sum": per_class[c]["gt_sum"]
                } for c in ids
            },
            "overall": {
                "dice": overall_dice,
                "iou": overall_iou,
                "hd95": overall_hd95,
                "nsd": overall_nsd
            }
        }

        hd95_str = f", HD95={overall_hd95:.4f}" if not (np.isnan(overall_hd95) or np.isinf(overall_hd95)) else ", HD95=N/A"
        nsd_str = f", NSD={overall_nsd:.4f}" if not np.isnan(overall_nsd) else ", NSD=N/A"
        print(f"{name}: overall Dice={overall_dice:.4f}, IoU={overall_iou:.4f}{hd95_str}{nsd_str}")

    print(f"\nEvaluated {len(per_patient)} samples")

    # summarize
    summary = {}
    for c in ids:
        arr_d = np.array(all_per_class[c]["dice"])
        arr_i = np.array(all_per_class[c]["iou"])
        arr_h = np.array(all_per_class[c]["hd95"]) if len(all_per_class[c]["hd95"]) > 0 else np.array([np.nan])
        arr_n = np.array(all_per_class[c]["nsd"]) if len(all_per_class[c]["nsd"]) > 0 else np.array([np.nan])
        # Filter out inf values for HD95 mean
        valid_h = arr_h[~np.isinf(arr_h)]
        hd95_mean = float(np.nanmean(valid_h)) if len(valid_h) > 0 else np.nan
        # NSD mean
        nsd_mean = float(np.nanmean(arr_n)) if len(arr_n) > 0 else np.nan
        summary[c] = {
            "dice_mean": float(np.nanmean(arr_d)), 
            "iou_mean": float(np.nanmean(arr_i)), 
            "hd95_mean": hd95_mean,
            "nsd_mean": nsd_mean,
            "count": int(len(arr_d))
        }

    # Calculate overall HD95 mean (filter out inf and nan)
    valid_overall_hd95 = [h for h in overall_hd95_list if not (np.isinf(h) or np.isnan(h))]
    overall_hd95_mean = float(np.nanmean(valid_overall_hd95)) if len(valid_overall_hd95) > 0 else np.nan
    
    # Calculate overall NSD mean (filter out nan)
    valid_overall_nsd = [n for n in overall_nsd_list if not np.isnan(n)]
    overall_nsd_mean = float(np.nanmean(valid_overall_nsd)) if len(valid_overall_nsd) > 0 else np.nan
    
    overall = {
        "dice_mean": float(np.nanmean(np.array(overall_dice_list))), 
        "iou_mean": float(np.nanmean(np.array(overall_iou_list))),
        "hd95_mean": overall_hd95_mean,
        "nsd_mean": overall_nsd_mean
    }

    print("\nPer-class mean metrics:")
    for c in ids:
        hd95_val = summary[c]['hd95_mean']
        nsd_val = summary[c]['nsd_mean']
        hd95_str = f", HD95={hd95_val:.4f}" if not (np.isnan(hd95_val) or np.isinf(hd95_val)) else ", HD95=N/A"
        nsd_str = f", NSD={nsd_val:.4f}" if not np.isnan(nsd_val) else ", NSD=N/A"
        print(f"Class {c}: Dice={summary[c]['dice_mean']:.4f}, IoU={summary[c]['iou_mean']:.4f}{hd95_str}{nsd_str}, cases={summary[c]['count']}")

    hd95_overall_val = overall['hd95_mean']
    nsd_overall_val = overall['nsd_mean']
    hd95_overall_str = f", HD95={hd95_overall_val:.4f}" if not (np.isnan(hd95_overall_val) or np.isinf(hd95_overall_val)) else ", HD95=N/A"
    nsd_overall_str = f", NSD={nsd_overall_val:.4f}" if not np.isnan(nsd_overall_val) else ", NSD=N/A"
    print(f"\nOverall mean: Dice={overall['dice_mean']:.4f}, IoU={overall['iou_mean']:.4f}{hd95_overall_str}{nsd_overall_str}")

    # save summary including per-patient results
    if os.path.isabs(output_name) or os.path.dirname(output_name):
        outp = output_name
    else:
        outp = os.path.join("data", output_name)
    out_dir = os.path.dirname(outp)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    output_dict = {
        "per_class": summary,
        "overall": overall,
        "per_patient": per_patient,
        "n_cases": len(per_patient)
    }
    with open(outp, "w") as f:
        json.dump(output_dict, f, indent=2)
    print("Wrote summary to", outp)


if __name__ == '__main__':
    main()
