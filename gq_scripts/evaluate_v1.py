"""
BiomedParse v1 模型评估脚本（用于2D推理）

本脚本用于评估 BiomedParse v1 模型在2D医学图像分割数据集（如CAMUS）上的性能。
支持计算多种评估指标，包括 Dice 系数、IoU、HD95（95% Hausdorff 距离）和 NSD（归一化表面距离）。

主要功能：
- 加载预训练的 BiomedParse v1 模型
- 在指定数据集上进行2D推理
- 计算每个类别和整体的评估指标
- 生成详细的评估报告（包括每个样本的结果）

使用方法：
    python gq_scripts/evaluate_v1.py --data-root <数据目录> --dataset-name <数据集名称> --ckpt-path <检查点路径> [--output-name <输出文件名>]

参数说明：
    --data-root: 测试数据目录路径，包含 .npz 格式的数据文件
                 默认值: "data/CAMUS/test"
    
    --dataset-name: 数据集名称，需要在 class_prompts.json 中存在对应的配置
                    默认值: "CAMUS"
    
    --ckpt-path: 模型检查点文件路径
                 默认值: "/home/gaoqi/official_ckpt/biomedparse/biomedparse_v1.pt"
    
    --output-name: 输出评估摘要的 JSON 文件名（可选）
                   如果不指定，将使用 "{dataset_name}_eval_summary.json"
                   输出文件将保存在 data/ 目录下

输出说明：
    脚本会生成一个 JSON 格式的评估摘要文件，包含：
    - per_class: 每个类别的平均指标（Dice, IoU, HD95, NSD）
    - overall: 整体平均指标
    - per_patient: 每个样本的详细结果
    - n_cases: 评估的样本数量

注意事项：
    - 本脚本专门用于2D推理，适用于CAMUS等2D数据集
    - 如果数据文件中包含 spacing 信息，将计算 HD95 和 NSD 指标
    - 如果数据文件中没有 spacing 信息，HD95 和 NSD 将显示为 N/A
    - 默认情况下，如果某个类别有多个 prompt，脚本会使用第一个 prompt（所有样本使用相同的 prompt）
    - 使用 --random-prompt 参数可以启用随机 prompt 选择：每个样本的每个解剖区域都会独立地随机选择一个 prompt
    - 使用 --seed 参数可以设置随机种子，确保结果可复现
    - spacing的格式为[D,H,W], 对于camus 2d图像，d维度为1

示例：
    # 评估 CAMUS 数据集
    python gq_scripts/evaluate_v1.py --data-root data/CAMUS/test --dataset-name CAMUS --ckpt-path /home/gaoqi/official_ckpt/biomedparse/biomedparse_v1.pt
    python gq_scripts/evaluate_v1.py --data-root data/CAMUS/test --dataset-name CAMUS_mul --ckpt-path /home/gaoqi/official_ckpt/biomedparse/biomedparse_v1.pt --random-prompt --seed 42

"""

import os
import sys
import glob
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ensure repository root is on sys.path so top-level imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import v1 model components
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from scipy import ndimage


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


def select_prompts(prompts_dict, ids, random_select=False):
    """
    Select prompts for each class and combine them into a text string.
    
    Args:
        prompts_dict: Dictionary mapping class ID to list of prompts
        ids: List of class IDs
        random_select: If True, randomly select one prompt from multiple prompts for each class.
                       If False, use the first prompt (default behavior).
    
    Returns:
        text: Combined text prompt string
        selected_info: List of strings describing selected prompts (for logging)
    """
    texts = []
    selected_info = []
    
    for i in ids:
        prompts = prompts_dict[i]
        if random_select and len(prompts) > 1:
            # Randomly select one prompt from multiple prompts
            selected_prompt = random.choice(prompts)
            texts.append(selected_prompt)
            selected_info.append(f"Class {i}: selected prompt {prompts.index(selected_prompt)+1}/{len(prompts)}")
        else:
            # Use first prompt (default behavior)
            texts.append(prompts[0])
            if len(prompts) > 1 and random_select:
                selected_info.append(f"Class {i}: using first prompt (1/{len(prompts)})")
    
    text = "[SEP]".join(texts)
    return text, selected_info


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


def npz_to_pil_image(imgs):
    """
    Convert npz image array to PIL Image.
    
    Args:
        imgs: np.ndarray of shape (D, H, W) or (H, W), uint8 [0, 255]
    
    Returns:
        PIL.Image: RGB image
    """
    # Handle 2D case: (H, W) -> (1, H, W)
    if imgs.ndim == 2:
        imgs = imgs[np.newaxis, :, :]
    
    # For CAMUS, typically D=1 (single 2D image)
    # Take the first (and likely only) slice
    img_2d = imgs[0]  # (H, W)
    
    # Convert to RGB by repeating the channel
    img_rgb = np.stack([img_2d, img_2d, img_2d], axis=-1)  # (H, W, 3)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(img_rgb.astype(np.uint8))
    
    return pil_image


def inference_2d_slice(model, image_pil, prompts_list):
    """
    Run 2D inference on a single image slice using v1 model.
    
    Args:
        model: BiomedParse v1 model
        image_pil: PIL.Image, RGB image
        prompts_list: List of prompt strings for each class
    
    Returns:
        pred_masks: List of numpy arrays, one mask per prompt (probability maps)
    """
    # interactive_infer_image returns a single probability map for the best matching prompt
    # For multi-class segmentation, we need to run inference for each class separately
    pred_mask_probs_list = []
    for prompt in prompts_list:
        # Run inference for each prompt separately
        pred_mask_prob = interactive_infer_image(model, image_pil, [prompt])
        # pred_mask_prob might be (1, H, W) or (H, W), ensure it's (H, W)
        if pred_mask_prob.ndim == 3:
            # If shape is (1, H, W), squeeze the first dimension
            if pred_mask_prob.shape[0] == 1:
                pred_mask_prob = pred_mask_prob[0]  # (H, W)
            else:
                # If shape is (num_prompts, H, W), take the first one
                pred_mask_prob = pred_mask_prob[0]  # (H, W)
        # pred_mask_prob is now (H, W) probability map
        pred_mask_probs_list.append(pred_mask_prob)
    
    return pred_mask_probs_list


def merge_2d_masks(pred_mask_probs_list, class_ids, threshold=0.5):
    """
    Merge multiple 2D probability masks into a single multi-class segmentation.
    
    Args:
        pred_mask_probs_list: List of probability maps, one per class
        class_ids: List of class IDs corresponding to each mask
        threshold: Threshold for binarization
    
    Returns:
        merged_mask: np.ndarray of shape (H, W) with integer class labels
    """
    # Ensure all masks are 2D (H, W) - defensive check
    processed_masks = []
    for mask in pred_mask_probs_list:
        if mask.ndim == 3:
            # If shape is (1, H, W) or (num_prompts, H, W), take the first slice
            mask = mask[0]  # (H, W)
        elif mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {mask.shape}, expected (H, W) or (1, H, W)")
        processed_masks.append(mask)
    
    # Now all masks should be (H, W)
    H, W = processed_masks[0].shape
    merged_mask = np.zeros((H, W), dtype=np.int32)
    
    # Stack all probability maps
    prob_stack = np.stack(processed_masks, axis=0)  # (num_classes, H, W)
    
    # For each pixel, assign to the class with highest probability (if above threshold)
    max_probs = np.max(prob_stack, axis=0)  # (H, W)
    max_indices = np.argmax(prob_stack, axis=0)  # (H, W)
    
    # Only assign class if probability is above threshold
    valid_mask = max_probs > threshold
    merged_mask[valid_mask] = np.array(class_ids)[max_indices[valid_mask]]
    
    return merged_mask


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate BiomedParse v1 model on 2D dataset")
    parser.add_argument("--data-root", type=str, default="data/CAMUS/test",
                        help="Path to dataset test directory containing .npz files")
    parser.add_argument("--dataset-name", type=str, default="CAMUS",
                        help="Dataset name in class_prompts.json (e.g., CAMUS)")
    parser.add_argument("--ckpt-path", type=str, default="/home/gaoqi/official_ckpt/biomedparse/biomedparse_v1.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output summary JSON filename (default: {dataset_name}_eval_summary.json)")
    parser.add_argument("--random-prompt", action="store_true",
                        help="Randomly select one prompt from multiple prompts for each class for EACH sample (if available)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for prompt selection (for reproducibility)")
    args = parser.parse_args()
    
    # Set random seed if provided (for reproducibility)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # paths
    data_root = args.data_root
    ckpt_path = args.ckpt_path
    dataset_name = args.dataset_name
    output_name = args.output_name if args.output_name else f"{dataset_name}_eval_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load prompts dictionary
    ids, prompts_dict = load_prompts(data_root, dataset_name=dataset_name)
    print("Using class ids:", ids)
    
    # If random_select is False, select prompts once at the beginning
    if not args.random_prompt:
        text, _ = select_prompts(prompts_dict, ids, random_select=False)
        prompts_list = text.split("[SEP]")
        print("Using fixed prompts (first prompt for each class)")
    else:
        print("Using random prompt selection (each sample will use independently selected prompts)")

    # Build model config for v1
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)
    
    # Load model from pretrained weights
    model = BaseModel(opt, build_model(opt)).from_pretrained(ckpt_path).eval().to(device)
    
    # Pre-compute text embeddings for BIOMED_CLASSES
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )

    npz_files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
    
    if len(npz_files) == 0:
        print("No .npz files found in", data_root)
        return

    all_per_class = {c: {"dice": [], "iou": [], "hd95": [], "nsd": []} for c in ids}
    overall_dice_list = []
    overall_iou_list = []
    overall_hd95_list = []
    overall_nsd_list = []
    per_patient = {}

    for p in npz_files:
        name = os.path.basename(p)
        d = np.load(p, allow_pickle=True)
        imgs = d["imgs"]  # (D,H,W) or (H,W), typically D=1 for 2D images
        gts = d["gts"]  # (D,H,W) or (H,W)
        
        # Handle 2D case: ensure we have (D,H,W) format
        if imgs.ndim == 2:
            imgs = imgs[np.newaxis, :, :]
            gts = gts[np.newaxis, :, :]
        
        # For 2D images, typically D=1, take the first slice
        img_2d = imgs[0]  # (H, W)
        gt_2d = gts[0]  # (H, W)
        
        # Get spacing if available (for 2D, spacing is typically (H, W))
        spacing = None
        if "spacing" in d:
            spacing_val = d["spacing"]
            # Handle both array and tuple formats
            if isinstance(spacing_val, (np.ndarray, list, tuple)):
                if len(spacing_val) == 2:
                    # 2D spacing: add depth dimension
                    spacing = (1.0, spacing_val[0], spacing_val[1])  # (D, H, W)
                else:
                    spacing = tuple(spacing_val[:3])  # (D, H, W)
            else:
                spacing = (1.0, spacing_val, spacing_val)  # fallback: assume isotropic

        # Select prompts for this sample (if random_select is enabled)
        if args.random_prompt:
            sample_text, selected_info = select_prompts(prompts_dict, ids, random_select=True)
            sample_prompts_list = sample_text.split("[SEP]")
            if selected_info:
                print(f"  {name} prompt selection:")
                for info in selected_info:
                    print(f"    {info}")
        else:
            sample_prompts_list = prompts_list
        
        # Convert image to PIL Image
        image_pil = npz_to_pil_image(img_2d)
        
        # Run 2D inference for each class
        pred_mask_probs_list = inference_2d_slice(model, image_pil, sample_prompts_list)
        
        # Merge masks into multi-class segmentation
        pred_2d = merge_2d_masks(pred_mask_probs_list, ids, threshold=0.5)
        
        # Convert to 3D format (D, H, W) for metric computation
        pred = pred_2d[np.newaxis, :, :]  # (1, H, W)
        gt = gt_2d[np.newaxis, :, :]  # (1, H, W)

        per_class, overall_dice, overall_iou, overall_hd95, overall_nsd = compute_metrics(
            pred, gt, ids, spacing=spacing
        )

        for c in ids:
            all_per_class[c]["dice"].append(per_class[c]["dice"])
            all_per_class[c]["iou"].append(per_class[c]["iou"])
            # Only append HD95 if it's valid (not nan and not inf)
            if not (np.isnan(per_class[c]["hd95"]) or np.isinf(per_class[c]["hd95"])):
                all_per_class[c]["hd95"].append(per_class[c]["hd95"])
            # Only append NSD if it's valid (not nan)
            if not np.isnan(per_class[c]["nsd"]):
                all_per_class[c]["nsd"].append(per_class[c]["nsd"])
        overall_dice_list.append(overall_dice)
        overall_iou_list.append(overall_iou)
        # Only append HD95 if it's valid (not nan and not inf)
        if not (np.isnan(overall_hd95) or np.isinf(overall_hd95)):
            overall_hd95_list.append(overall_hd95)
        # Only append NSD if it's valid (not nan)
        if not np.isnan(overall_nsd):
            overall_nsd_list.append(overall_nsd)

        # record per-patient results
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
    outp = os.path.join("data", output_name)
    output_dict = {
        "per_class": summary,
        "overall": overall,
        "per_patient": per_patient,
        "n_cases": len(npz_files)
    }
    with open(outp, "w") as f:
        json.dump(output_dict, f, indent=2)
    print("Wrote summary to", outp)


if __name__ == '__main__':
    main()
