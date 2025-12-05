import os
import sys
import glob
import json
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
    texts = [ds[str(i)][0] for i in ids]  # use first prompt for each class
    text = "[SEP]".join(texts)
    return ids, text


def compute_metrics(pred, gt, class_ids):
    # pred, gt: np.ndarray (D,H,W) integer labels
    per_class = {}
    total_inter = 0
    total_union = 0
    total_gt_sum = 0
    total_pred_sum = 0
    for c in class_ids:
        pred_mask = (pred == c)
        gt_mask = (gt == c)
        inter = int((pred_mask & gt_mask).sum())
        pred_sum = int(pred_mask.sum())
        gt_sum = int(gt_mask.sum())
        union = int((pred_mask | gt_mask).sum())
        # dice
        if pred_sum + gt_sum == 0:
            dice = 1.0
        else:
            dice = 2.0 * inter / (pred_sum + gt_sum)
        # iou
        if union == 0:
            iou = 1.0
        else:
            iou = inter / union

        per_class[c] = {"dice": float(dice), "iou": float(iou), "pred_sum": pred_sum, "gt_sum": gt_sum}
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

    return per_class, float(overall_dice), float(overall_iou)


def main():
    # paths
    data_root = "data/ACDC/test"
    ckpt_path = "checkpoint/biomedparse_v2.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load prompts
    ids, text = load_prompts(data_root, dataset_name="ACDC")
    print("Using class ids:", ids)

    # instantiate model via hydra
    GlobalHydra.instance().clear()
    # use absolute path to repo's configs/model to avoid relative lookup from scripts/
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg_path = os.path.join(repo_root, 'configs', 'model')
    # hydra.initialize requires a relative config_path; for an absolute path use initialize_config_dir
    if os.path.isabs(cfg_path):
        hydra.initialize_config_dir(config_dir=cfg_path, job_name="eval_acdc")
    else:
        hydra.initialize(config_path=cfg_path, job_name="eval_acdc")
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
    if len(npz_files) == 0:
        print("No .npz files found in", data_root)
        return

    all_per_class = {c: {"dice": [], "iou": []} for c in ids}
    overall_dice_list = []
    overall_iou_list = []
    per_patient = {}

    for p in npz_files:
        name = os.path.basename(p)
        d = np.load(p, allow_pickle=True)
        imgs = d["imgs"]  # (D,H,W)
        gts = d["gts"]

        # prepare input
        imgs_proc, pad_width, padded_size, valid_axis = process_input(imgs, 512)
        imgs_proc = imgs_proc.to(device).int()
        input_tensor = {"image": imgs_proc.unsqueeze(0), "text": [text]}

        with torch.no_grad():
            output = model(input_tensor, mode="eval", slice_batch_size=4)

        mask_preds = output["predictions"]["pred_gmasks"]
        # resize to 512
        mask_preds = F.interpolate(mask_preds, size=(512, 512), mode="bicubic", align_corners=False, antialias=True)
        mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

        pred = mask_preds  # expected (D,H,W) int map
        gt = gts

        per_class, overall_dice, overall_iou = compute_metrics(pred, gt, ids)

        for c in ids:
            all_per_class[c]["dice"].append(per_class[c]["dice"])
            all_per_class[c]["iou"].append(per_class[c]["iou"])
        overall_dice_list.append(overall_dice)
        overall_iou_list.append(overall_iou)

        # record per-patient results
        per_patient[name] = {
            "per_class": {str(c): {"dice": per_class[c]["dice"], "iou": per_class[c]["iou"], "pred_sum": per_class[c]["pred_sum"], "gt_sum": per_class[c]["gt_sum"]} for c in ids},
            "overall": {"dice": overall_dice, "iou": overall_iou}
        }

        print(f"{name}: overall Dice={overall_dice:.4f}, IoU={overall_iou:.4f}")

    # summarize
    summary = {}
    for c in ids:
        arr_d = np.array(all_per_class[c]["dice"])
        arr_i = np.array(all_per_class[c]["iou"])
        summary[c] = {"dice_mean": float(np.nanmean(arr_d)), "iou_mean": float(np.nanmean(arr_i)), "count": int(len(arr_d))}

    overall = {"dice_mean": float(np.nanmean(np.array(overall_dice_list))), "iou_mean": float(np.nanmean(np.array(overall_iou_list)))}

    print("\nPer-class mean metrics:")
    for c in ids:
        print(f"Class {c}: Dice={summary[c]['dice_mean']:.4f}, IoU={summary[c]['iou_mean']:.4f}, cases={summary[c]['count']}")

    print(f"\nOverall mean: Dice={overall['dice_mean']:.4f}, IoU={overall['iou_mean']:.4f}")

    # save summary including per-patient results
    outp = os.path.join("data", "ACDC_eval_summary.json")
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
