"""
BiomedParse 模型推理脚本

本脚本用于在医学图像分割数据集上进行推理，并将预测结果保存到指定文件夹。

主要功能：
- 加载预训练的 BiomedParse 模型
- 在指定数据集上进行推理
- 将预测结果保存到输出文件夹（每个样本保存为 .npz 文件）

使用方法：
    python gq_scripts/inference_v2.py --data-root <数据目录> --dataset-name <数据集名称> --ckpt-path <检查点路径> --output-dir <输出目录> [--random-prompt] [--seed <随机种子>]

参数说明：
    --data-root: 测试数据目录路径，包含 .npz 格式的数据文件
                 默认值: "data/MMs2/test"
    
    --dataset-name: 数据集名称，需要在 class_prompts.json 中存在对应的配置
                    可选值: "ACDC", "CAMUS", "MMs2" 等
                    默认值: "MMs2"
    
    --ckpt-path: 模型检查点文件路径
                 默认值: "checkpoint/biomedparse_v2.ckpt"
    
    --output-dir: 输出目录路径，推理结果将保存到此目录
                  默认值: "inference_results"
    
    --random-prompt: 启用随机 prompt 选择：每个样本的每个解剖区域都会独立地随机选择一个 prompt
    --seed: 设置随机种子，确保结果可复现

输出说明：
    每个样本的推理结果将保存为一个 .npz 文件，包含：
    - pred_mask: 预测的分割mask (D, H, W) 整数标签
    - spacing: 体素间距信息（如果原始数据中有）
    - class_ids: 类别ID列表
    - prompt_info: 使用的prompt信息（用于记录）

注意事项：
    - 对于 MMs2 数据集，脚本会自动过滤，只处理 SA（短轴）数据，跳过 LA（长轴）数据
    - 输出文件名将与输入文件名相同（例如：input.npz -> input.npz）
    
示例：
    # 推理 ACDC 数据集
    python gq_scripts/inference_v2.py \
        --data-root gq_data/acdc/test \
        --dataset-name ACDC \
        --ckpt-path /home/gaoqi/official_ckpt/biomedparse/biomedparse_v2.ckpt \
        --output-dir gq_data/acdc/test/inference_results_v2

    # 使用随机 prompt 选择
    python gq_scripts/inference_v2.py \
        --data-root gq_data/acdc/test \
        --dataset-name ACDC \
        --ckpt-path /home/gaoqi/official_ckpt/biomedparse/biomedparse_v2.ckpt \
        --output-dir inference_results/ACDC \
        --random-prompt \
        --seed 42
"""

import os
import sys
import glob
import json
import random
import gc
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra

# ensure repository root is on sys.path so top-level imports like `utils` work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import process_input, process_output
from inference import postprocess, merge_multiclass_masks


def load_prompts(data_root, dataset_name="ACDC"):
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run BiomedParse model inference on dataset")
    parser.add_argument("--data-root", type=str, default="data/MMs2/test",
                        help="Path to dataset test directory containing .npz files")
    parser.add_argument("--dataset-name", type=str, default="MMs2",
                        help="Dataset name in class_prompts.json (e.g., ACDC, CAMUS)")
    parser.add_argument("--ckpt-path", type=str, default="checkpoint/biomedparse_v2.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="inference_results",
                        help="Output directory for inference results")
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
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load prompts dictionary
    ids, prompts_dict = load_prompts(data_root, dataset_name=dataset_name)
    print("Using class ids:", ids)
    
    # If random_select is False, select prompts once at the beginning
    if not args.random_prompt:
        text, _ = select_prompts(prompts_dict, ids, random_select=False)
        print("Using fixed prompts (first prompt for each class)")
    else:
        print("Using random prompt selection (each sample will use independently selected prompts)")

    # instantiate model via hydra
    GlobalHydra.instance().clear()
    # use absolute path to repo's configs/model to avoid relative lookup from scripts/
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg_path = os.path.join(repo_root, 'configs', 'model')
    # hydra.initialize requires a relative config_path; for an absolute path use initialize_config_dir
    if os.path.isabs(cfg_path):
        hydra.initialize_config_dir(config_dir=cfg_path, job_name="inference")
    else:
        hydra.initialize(config_path=cfg_path, job_name="inference")
    cfg = compose(config_name="biomedparse_3D")

    # Defensive: if tokenizer path in config is missing/None, fallback to local checkpoint folder
    try:
        tokval = cfg.sem_seg_head.predictor.language_encoder.tokenizer.pretrained_model_name_or_path
    except Exception:
        tokval = None
    if tokval is None:
        fallback = os.path.abspath(os.path.join(repo_root, 'checkpoint', 'clip-vit-base-patch32'))
        print(f"Warning: tokenizer pretrained_model_name_or_path is None in config; falling back to {fallback}")
        try:
            cfg.sem_seg_head.predictor.language_encoder.tokenizer.pretrained_model_name_or_path = fallback
        except Exception:
            # last resort: set an environment variable that some code may respect
            os.environ['PRETRAINED_TOKENIZER_FALLBACK'] = fallback

    # Try to proactively load the tokenizer from the local path and monkeypatch AutoTokenizer
    # This helps when transformers' CLIP tokenizer expects vocab files that the AutoTokenizer
    # resolution path might not provide during Hydra instantiation.
    try:
        import transformers
        from transformers import AutoTokenizer
        local_tok_path = None
        try:
            local_tok_path = cfg.sem_seg_head.predictor.language_encoder.tokenizer.pretrained_model_name_or_path
        except Exception:
            local_tok_path = os.environ.get('PRETRAINED_TOKENIZER_FALLBACK', None)

        if local_tok_path:
            try:
                # attempt local-only load
                tok = AutoTokenizer.from_pretrained(local_tok_path, local_files_only=True)
                print(f"Loaded tokenizer from local path: {local_tok_path}")

                # monkeypatch AutoTokenizer.from_pretrained to return our tokenizer when called
                _orig_atfp = transformers.AutoTokenizer.from_pretrained
                def _patched_atfp(pretrained_model_name_or_path, *a, **kw):
                    if pretrained_model_name_or_path in (local_tok_path, 'openai/clip-vit-base-patch32'):
                        return tok
                    return _orig_atfp(pretrained_model_name_or_path, *a, **kw)

                transformers.AutoTokenizer.from_pretrained = _patched_atfp
            except Exception as e:
                print(f"Warning: failed to load tokenizer locally from {local_tok_path}: {e}")
    except Exception as e:
        print(f"Warning: transformers not available or failed to prepare local tokenizer patch: {e}")

    model = hydra.utils.instantiate(cfg, _convert_="object")
    model.load_pretrained(ckpt_path)
    model = model.to(device).eval()

    npz_files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
    
    # For MMs2 dataset, only process SA (Short Axis) data, skip LA (Long Axis) data
    if dataset_name.startswith("MMs2"):
        npz_files = [f for f in npz_files if "_SA_" in os.path.basename(f)]
        print(f"Filtered to SA files only: {len(npz_files)} files remaining")
    
    if len(npz_files) == 0:
        print("No .npz files found in", data_root)
        return

    print(f"\nStarting inference on {len(npz_files)} files...")
    print(f"Results will be saved to: {output_dir}\n")

    for idx, p in enumerate(npz_files, 1):
        name = os.path.basename(p)
        print(f"[{idx}/{len(npz_files)}] Processing: {name}")
        try:
            d = np.load(p, allow_pickle=True)
            imgs = d["imgs"]  # (D,H,W)

            # Get spacing if available
            spacing = None
            if "spacing" in d:
                spacing_val = d["spacing"]
                if isinstance(spacing_val, (np.ndarray, list, tuple)):
                    spacing = tuple(spacing_val)
                else:
                    spacing = tuple([spacing_val] * 3)

            # Select prompts for this sample (if random_select is enabled)
            if args.random_prompt:
                sample_text, selected_info = select_prompts(prompts_dict, ids, random_select=True)
                if selected_info:
                    print(f"  Prompt selection:")
                    for info in selected_info:
                        print(f"    {info}")
            else:
                sample_text = text

            # prepare input
            imgs_proc, pad_width, padded_size, valid_axis = process_input(imgs, 512)
            imgs_proc = imgs_proc.to(device).int()
            input_tensor = {"image": imgs_proc.unsqueeze(0), "text": [sample_text]}

            with torch.no_grad():
                output = model(input_tensor, mode="eval", slice_batch_size=4)

            mask_preds = output["predictions"]["pred_gmasks"]
            mask_preds = F.interpolate(mask_preds, size=(512, 512), mode="bicubic", align_corners=False, antialias=True)
            mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
            mask_preds = merge_multiclass_masks(mask_preds, ids)
            mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

            pred_mask = mask_preds  # expected (D,H,W) int map

            # Save inference result
            output_path = os.path.join(output_dir, name)
            save_dict = {
                "pred_mask": pred_mask,
                "class_ids": np.array(ids),
                "prompt_info": sample_text if not args.random_prompt else "random_selected"
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

    print(f"\nInference completed! All results saved to: {output_dir}")


if __name__ == '__main__':
    main()
