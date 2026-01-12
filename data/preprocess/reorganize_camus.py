"""
reorganize_camus.py

将CAMUS数据集从patient文件夹结构重组为img/gt文件夹结构。

将每个patient文件夹中的：
- 图像文件（*.nii.gz，不包含_gt，排除half_sequence）移动到 data/CAMUS/test/img/
- GT文件（*_gt.nii.gz，排除half_sequence）移动到 data/CAMUS/test/gt/

保持文件名不变。
注意：half_sequence文件不会被移动，保留在patient文件夹中。

Usage:
    python biomedparse_datasets/reorganize_camus.py --dataset-root data/CAMUS --split test
"""
import os
import argparse
import glob
import shutil


def reorganize_camus(dataset_root, split="test"):
    """
    重组CAMUS数据集结构
    
    Args:
        dataset_root: 数据集根目录，如 data/CAMUS
        split: 数据集分割，如 test
    """
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # 创建img和gt目录
    img_dir = os.path.join(split_dir, "img")
    gt_dir = os.path.join(split_dir, "gt")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    # 获取所有patient文件夹
    patient_dirs = [d for d in os.listdir(split_dir) 
                   if os.path.isdir(os.path.join(split_dir, d)) and d.startswith("patient")]
    patient_dirs.sort()
    
    if len(patient_dirs) == 0:
        print(f"No patient directories found in {split_dir}")
        return
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    n_img_moved = 0
    n_gt_moved = 0
    n_skipped = 0
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(split_dir, patient_dir)
        
        # 获取该patient文件夹中的所有.nii.gz文件
        nii_files = glob.glob(os.path.join(patient_path, "*.nii.gz"))
        
        for nii_file in nii_files:
            filename = os.path.basename(nii_file)
            
            # 跳过half_sequence文件
            if "half_sequence" in filename:
                continue
            
            # 判断是GT文件还是图像文件
            if filename.endswith("_gt.nii.gz"):
                # GT文件：移动到gt目录
                dest_path = os.path.join(gt_dir, filename)
                if os.path.exists(dest_path):
                    print(f"Warning: {filename} already exists in gt/, skipping")
                    n_skipped += 1
                    continue
                shutil.move(nii_file, dest_path)
                n_gt_moved += 1
            else:
                # 图像文件：移动到img目录
                dest_path = os.path.join(img_dir, filename)
                if os.path.exists(dest_path):
                    print(f"Warning: {filename} already exists in img/, skipping")
                    n_skipped += 1
                    continue
                shutil.move(nii_file, dest_path)
                n_img_moved += 1
        
        print(f"Processed {patient_dir}: moved {len([f for f in nii_files if not f.endswith('_gt.nii.gz')])} images, "
              f"{len([f for f in nii_files if f.endswith('_gt.nii.gz')])} GT files")
    
    print(f"\nDone!")
    print(f"  Images moved to img/: {n_img_moved}")
    print(f"  GT files moved to gt/: {n_gt_moved}")
    print(f"  Skipped (duplicates): {n_skipped}")
    print(f"\nImage directory: {img_dir}")
    print(f"GT directory: {gt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize CAMUS dataset from patient folders to img/gt structure"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to dataset root, e.g. data/CAMUS"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to process: train/test"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reorganize_camus(args.dataset_root, args.split)

