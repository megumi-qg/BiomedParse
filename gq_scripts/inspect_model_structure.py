#!/usr/bin/env python3
"""
查看biomedparse v2模型权重中的层级结构
"""
import torch
from collections import defaultdict

def print_structure(weights_dict, prefix="", max_depth=3, current_depth=0):
    """递归打印权重字典的结构"""
    if current_depth >= max_depth:
        return
    
    # 收集当前层级的所有键
    current_level = defaultdict(list)
    
    for key in weights_dict.keys():
        if key.startswith(prefix):
            remaining = key[len(prefix):].lstrip('.')
            if '.' in remaining:
                next_level = remaining.split('.')[0]
                if next_level not in current_level:
                    current_level[next_level] = []
            else:
                # 这是叶子节点
                current_level[remaining] = []
    
    # 打印当前层级
    for key in sorted(current_level.keys()):
        indent = "  " * current_depth
        print(f"{indent}{key}")
        
        # 递归打印下一层
        next_prefix = f"{prefix}.{key}" if prefix else key
        print_structure(weights_dict, next_prefix, max_depth, current_depth + 1)

def inspect_checkpoint(checkpoint_path):
    """检查checkpoint文件的结构"""
    print(f"正在加载权重文件: {checkpoint_path}")
    
    # 加载权重文件
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 确定state_dict的位置
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("找到 'state_dict' 键")
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("找到 'model' 键")
    else:
        state_dict = checkpoint
        print("直接使用checkpoint作为state_dict")
    
    print(f"\n总共有 {len(state_dict)} 个权重键\n")
    
    # 提取model.backbone的键
    backbone_keys = [k for k in state_dict.keys() if k.startswith("model.backbone.")]
    sem_seg_head_keys = [k for k in state_dict.keys() if k.startswith("model.sem_seg_head.")]
    
    print("=" * 80)
    print("model.backbone 的层级结构:")
    print("=" * 80)
    
    if backbone_keys:
        # 创建backbone的子字典
        backbone_dict = {}
        for key in backbone_keys:
            sub_key = key.replace("model.backbone.", "")
            backbone_dict[sub_key] = state_dict[key]
        
        # 打印backbone的结构（最多3层）
        print_structure(backbone_dict, prefix="", max_depth=3, current_depth=0)
    else:
        print("未找到 model.backbone 的权重")
    
    print("\n" + "=" * 80)
    print("model.sem_seg_head 的层级结构:")
    print("=" * 80)
    
    if sem_seg_head_keys:
        # 创建sem_seg_head的子字典
        sem_seg_head_dict = {}
        for key in sem_seg_head_keys:
            sub_key = key.replace("model.sem_seg_head.", "")
            sem_seg_head_dict[sub_key] = state_dict[key]
        
        # 打印sem_seg_head的结构（最多3层）
        print_structure(sem_seg_head_dict, prefix="", max_depth=3, current_depth=0)
    else:
        print("未找到 model.sem_seg_head 的权重")
    
    # 额外：显示第一层的所有键（用于验证）
    print("\n" + "=" * 80)
    print("backbone 第一层键列表（前20个）:")
    print("=" * 80)
    if backbone_keys:
        first_level = set()
        for key in backbone_keys:
            sub_key = key.replace("model.backbone.", "")
            first_part = sub_key.split('.')[0]
            first_level.add(first_part)
        for key in sorted(list(first_level))[:20]:
            print(f"  {key}")
        if len(first_level) > 20:
            print(f"  ... 还有 {len(first_level) - 20} 个键")
    
    print("\n" + "=" * 80)
    print("sem_seg_head 第一层键列表（前20个）:")
    print("=" * 80)
    if sem_seg_head_keys:
        first_level = set()
        for key in sem_seg_head_keys:
            sub_key = key.replace("model.sem_seg_head.", "")
            first_part = sub_key.split('.')[0]
            first_level.add(first_part)
        for key in sorted(list(first_level))[:20]:
            print(f"  {key}")
        if len(first_level) > 20:
            print(f"  ... 还有 {len(first_level) - 20} 个键")

if __name__ == "__main__":
    checkpoint_path = "/home/gaoqi/official_ckpt/biomedparse/biomedparse_v2.ckpt"
    inspect_checkpoint(checkpoint_path)
