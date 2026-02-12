"""
BiomedParse v1 模型推理脚本（2D/3D）

本脚本用于在 2D 或 3D 医学图像分割数据集上进行推理。BiomedParse v1 仅支持 2D 推理，
因此对于 3D 数据（如 ACDC MRI），会对每个切片逐片进行 2D 推理，再将结果堆叠为 3D 体积。
评估时，请使用 evaluate_v2.py 脚本进行评估。

输入数据说明：
    - 输入为 .npz 文件，每个文件需包含 imgs (D, H, W)，可选 spacing。
    - 图像强度建议为 [0, 255] uint8（如由 convert_nii_to_npz.py 预处理得到）。
    - 无需在预处理阶段将图像 resize 到 1024×1024：推理时 inference_utils.inference
      会在内部将每帧 resize 到 1024×1024 送入模型，预测 mask 再插值回原始 (H, W) 后
      与输入尺寸一致。

主要功能：
- 加载预训练的 BiomedParse v1 模型
- 支持 2D 数据（D=1）和 3D 数据（D>1，逐切片推理）
- 将预测结果保存到输出文件夹（每个样本保存为 .npz 文件）

使用方法：
    python gq_scripts/inference_v1.py --data-root <数据目录> --dataset-name <数据集名称> --ckpt-path <检查点路径> --output-dir <输出目录>

参数说明：
    --data-root: 测试数据目录路径，包含 .npz 格式的数据文件
    --dataset-name: 数据集名称，需要在 class_prompts.json 中存在对应的配置
    --ckpt-path: 模型检查点文件路径
    --output-dir: 输出目录路径，推理结果将保存到此目录
    --random-prompt: 启用随机 prompt 选择
    --seed: 随机种子

输出说明：
    每个样本的推理结果保存为一个 .npz 文件，包含：
    - pred_mask: 预测的分割 mask (D, H, W)
    - class_ids: 类别 ID 列表
    - spacing: 体素间距（如果原始数据中有）
    - prompt_info: 使用的 prompt 信息

示例：
    # 2D 数据（CAMUS）
    python gq_scripts/inference_v1.py \
        --data-root data/CAMUS/test \
        --dataset-name CAMUS \
        --ckpt-path /path/to/biomedparse_v1.pt \
        --output-dir inference_results/CAMUS

    # 3D 数据（ACDC）
    python gq_scripts/inference_v1.py \
        --data-root gq_data/acdc/test \
        --dataset-name ACDC \
        --ckpt-path /home/gaoqi/official_ckpt/biomedparse/biomedparse_v1.pt \
        --output-dir gq_data/acdc/test/inference_results_v1

    # 3D数据, BTCV_cervix
    python gq_scripts/inference_v1.py \
    --data-root gq_data/btcv/test \
    --dataset-name BTCV_cervix \
    --ckpt-path /home/gaoqi/official_ckpt/biomedparse/biomedparse_v1.pt \
    --output-dir gq_data/btcv/test/inference_results_v1_1

    # 3D数据, PROMISE12
    python gq_scripts/inference_v1.py \
    --data-root gq_data/promise12/test \
    --dataset-name PROMISE12 \
    --ckpt-path /home/gaoqi/official_ckpt/biomedparse/biomedparse_v1.pt \
    --output-dir gq_data/promise12/test/inference_results_v1
"""

import os
import sys
import glob
import json
import random
import gc
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image


def load_prompts(data_root, dataset_name="CAMUS"):
    """Load prompts from class_prompts.json."""
    cur = os.path.abspath(data_root)
    found = None
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
        raise FileNotFoundError(f"class_prompts.json not found near {data_root}")

    with open(found, "r") as f:
        cj = json.load(f)
    if dataset_name not in cj:
        raise KeyError(f"Dataset {dataset_name} not in class_prompts.json")
    ds = cj[dataset_name]
    ids = [int(k) for k in ds.keys() if k.isdigit()]
    ids.sort()
    prompts_dict = {i: ds[str(i)] for i in ids}
    return ids, prompts_dict


def select_prompts(prompts_dict, ids, random_select=False):
    """Select prompts for each class and return as list."""
    texts = []
    selected_info = []
    for i in ids:
        prompts = prompts_dict[i]
        if random_select and len(prompts) > 1:
            selected_prompt = random.choice(prompts)
            texts.append(selected_prompt)
            selected_info.append(f"Class {i}: selected prompt {prompts.index(selected_prompt)+1}/{len(prompts)}")
        else:
            texts.append(prompts[0])
    return texts, selected_info


def slice_to_pil_image(slice_2d):
    """Convert a 2D slice (H, W) to PIL Image (RGB)."""
    if slice_2d.ndim == 3:
        slice_2d = slice_2d[0]
    img_rgb = np.stack([slice_2d, slice_2d, slice_2d], axis=-1)
    return Image.fromarray(img_rgb.astype(np.uint8))


def inference_2d_slice(model, image_pil, prompts_list):
    """Run 2D inference for each prompt."""
    pred_mask_probs_list = []
    for prompt in prompts_list:
        pred_mask_prob = interactive_infer_image(model, image_pil, [prompt])
        if pred_mask_prob.ndim == 3:
            pred_mask_prob = pred_mask_prob[0] if pred_mask_prob.shape[0] == 1 else pred_mask_prob[0]
        pred_mask_probs_list.append(pred_mask_prob)
    return pred_mask_probs_list


def merge_2d_masks(pred_mask_probs_list, class_ids, threshold=0.5):
    """Merge probability masks into multi-class segmentation."""
    processed_masks = []
    for mask in pred_mask_probs_list:
        if mask.ndim == 3:
            mask = mask[0]
        processed_masks.append(mask)

    H, W = processed_masks[0].shape
    merged_mask = np.zeros((H, W), dtype=np.int32)
    prob_stack = np.stack(processed_masks, axis=0)
    max_probs = np.max(prob_stack, axis=0)
    max_indices = np.argmax(prob_stack, axis=0)
    valid_mask = max_probs > threshold
    merged_mask[valid_mask] = np.array(class_ids)[max_indices[valid_mask]]
    return merged_mask


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run BiomedParse v1 model inference on 2D dataset")
    parser.add_argument("--data-root", type=str, default="data/CAMUS/test",
                        help="Path to dataset directory containing .npz files")
    parser.add_argument("--dataset-name", type=str, default="CAMUS",
                        help="Dataset name in class_prompts.json")
    parser.add_argument("--ckpt-path", type=str, default="/home/gaoqi/official_ckpt/biomedparse/biomedparse_v1.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="inference_results",
                        help="Output directory for inference results")
    parser.add_argument("--random-prompt", action="store_true",
                        help="Randomly select prompt for each sample")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ids, prompts_dict = load_prompts(args.data_root, dataset_name=args.dataset_name)
    print("Using class ids:", ids)

    if not args.random_prompt:
        prompts_list, _ = select_prompts(prompts_dict, ids, random_select=False)
        print("Using fixed prompts (first prompt for each class)")
    else:
        print("Using random prompt selection per sample")

    # Build and load model
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt_path).eval().to(device)
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )

    npz_files = sorted(glob.glob(os.path.join(args.data_root, "*.npz")))
    if len(npz_files) == 0:
        print("No .npz files found in", args.data_root)
        return

    print(f"\nStarting inference on {len(npz_files)} files...")
    print(f"Results will be saved to: {args.output_dir}\n")

    for idx, p in enumerate(npz_files, 1):
        name = os.path.basename(p)
        print(f"[{idx}/{len(npz_files)}] Processing: {name}")
        try:
            d = np.load(p, allow_pickle=True)
            imgs = d["imgs"]
            if imgs.ndim == 2:
                imgs = imgs[np.newaxis, :, :]

            D, H, W = imgs.shape
            is_3d = D > 1

            spacing = None
            if "spacing" in d:
                spacing_val = d["spacing"]
                if isinstance(spacing_val, (np.ndarray, list, tuple)):
                    if len(spacing_val) == 2:
                        spacing = (1.0, spacing_val[0], spacing_val[1])
                    else:
                        spacing = tuple(spacing_val[:3])
                else:
                    spacing = (1.0, spacing_val, spacing_val)

            if args.random_prompt:
                prompts_list, selected_info = select_prompts(prompts_dict, ids, random_select=True)
                if selected_info:
                    print(f"  Prompt selection:")
                    for info in selected_info:
                        print(f"    {info}")
            else:
                prompts_list, _ = select_prompts(prompts_dict, ids, random_select=False)

            # 逐切片进行 2D 推理（支持 3D 体积如 ACDC）
            pred_slices = []
            for slice_idx in range(D):
                if is_3d and (slice_idx + 1) % 5 == 0:
                    print(f"    Slice {slice_idx + 1}/{D}...")
                slice_2d = imgs[slice_idx]
                image_pil = slice_to_pil_image(slice_2d)
                pred_mask_probs_list = inference_2d_slice(model, image_pil, prompts_list)
                pred_2d = merge_2d_masks(pred_mask_probs_list, ids, threshold=0.5)
                pred_slices.append(pred_2d)

            pred_mask = np.stack(pred_slices, axis=0).astype(np.int32)  # (D, H, W)

            output_path = os.path.join(args.output_dir, name)
            save_dict = {
                "pred_mask": pred_mask,
                "class_ids": np.array(ids),
                "prompt_info": "[SEP]".join(prompts_list) if not args.random_prompt else "random_selected"
            }
            if spacing is not None:
                save_dict["spacing"] = np.array(spacing)

            np.savez_compressed(output_path, **save_dict)
            print(f"  Saved to: {output_path}")

        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            continue
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nInference completed! All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
