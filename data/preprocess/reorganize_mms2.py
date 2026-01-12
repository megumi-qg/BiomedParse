"""
reorganize_mms2.py

将MMs2数据集从patient文件夹结构重组为img/gt文件夹结构。

将每个patient文件夹中的：
- 图像文件（*_LA_ED.nii.gz, *_LA_ES.nii.gz, *_SA_ED.nii.gz, *_SA_ES.nii.gz）移动到 data/MMs2/test/img/
- GT文件（*_LA_ED_gt.nii.gz, *_LA_ES_gt.nii.gz, *_SA_ED_gt.nii.gz, *_SA_ES_gt.nii.gz）移动到 data/MMs2/test/gt/

保持文件名不变。
注意：CINE文件不会被移动，保留在patient文件夹中。
"""
import os
import argparse
import glob
import shutil


def reorganize_mms2(dataset_root, split="test"):
    """
    重组MMs2数据集结构
    
    Args:
        dataset_root: 数据集根目录，如 data/MMs2
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
    
    # 获取所有patient文件夹（数字命名的文件夹）
    patient_dirs = [d for d in os.listdir(split_dir) 
                   if os.path.isdir(os.path.join(split_dir, d)) and d.isdigit()]
    patient_dirs.sort(key=int)  # 按数字排序
    
    if len(patient_dirs) == 0:
        print(f"No patient directories found in {split_dir}")
        return
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # 定义需要处理的文件模式
    img_patterns = ["*_LA_ED.nii.gz", "*_LA_ES.nii.gz", "*_SA_ED.nii.gz", "*_SA_ES.nii.gz"]
    gt_patterns = ["*_LA_ED_gt.nii.gz", "*_LA_ES_gt.nii.gz", "*_SA_ED_gt.nii.gz", "*_SA_ES_gt.nii.gz"]
    
    n_img_moved = 0
    n_gt_moved = 0
    n_skipped = 0
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(split_dir, patient_dir)
        
        # 处理图像文件
        for pattern in img_patterns:
            img_files = glob.glob(os.path.join(patient_path, pattern))
            for img_file in img_files:
                filename = os.path.basename(img_file)
                dest_path = os.path.join(img_dir, filename)
                if os.path.exists(dest_path):
                    print(f"Warning: {filename} already exists in img/, skipping")
                    n_skipped += 1
                    continue
                shutil.move(img_file, dest_path)
                n_img_moved += 1
        
        # 处理GT文件
        for pattern in gt_patterns:
            gt_files = glob.glob(os.path.join(patient_path, pattern))
            for gt_file in gt_files:
                filename = os.path.basename(gt_file)
                dest_path = os.path.join(gt_dir, filename)
                if os.path.exists(dest_path):
                    print(f"Warning: {filename} already exists in gt/, skipping")
                    n_skipped += 1
                    continue
                shutil.move(gt_file, dest_path)
                n_gt_moved += 1
        
        # 统计该patient处理的文件数
        img_count = sum(len(glob.glob(os.path.join(patient_path, p))) for p in img_patterns)
        gt_count = sum(len(glob.glob(os.path.join(patient_path, p))) for p in gt_patterns)
        print(f"Processed {patient_dir}: moved {img_count} images, {gt_count} GT files")
    
    print(f"\nDone!")
    print(f"  Images moved to img/: {n_img_moved}")
    print(f"  GT files moved to gt/: {n_gt_moved}")
    print(f"  Skipped (duplicates): {n_skipped}")
    print(f"\nImage directory: {img_dir}")
    print(f"GT directory: {gt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize MMs2 dataset from patient folders to img/gt structure"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to dataset root, e.g. data/MMs2"
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
    reorganize_mms2(args.dataset_root, args.split)

