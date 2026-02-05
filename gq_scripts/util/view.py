"""
查看 .npz 文件信息的工具脚本

用法:
    python gq_scripts/util/view.py <npz_file_path>
    
示例:
    python gq_scripts/util/view.py gq_data/acdc/test/patient101_frame01.npz
"""

import os
import sys
import numpy as np
import argparse


def print_array_info(name, arr):
    """打印数组的详细信息"""
    print(f"\n{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    
    # 如果是对象数组（object dtype），特殊处理
    if arr.dtype == np.dtype('object'):
        print(f"  Type: Object array (contains Python objects)")
        
        # 处理标量对象数组（0维数组）
        if arr.shape == ():
            elem = arr.item()  # 提取标量值
            print(f"  Value type: {type(elem).__name__}")
            if isinstance(elem, dict):
                print(f"  Keys: {list(elem.keys())}")
                # 打印字典的所有键值对
                for k, v in elem.items():
                    v_str = str(v)
                    if len(v_str) > 100:
                        v_str = v_str[:100] + "..."
                    print(f"    {k}: {type(v).__name__} {v_str}")
            elif isinstance(elem, (list, tuple)):
                print(f"  Length: {len(elem)}")
                if len(elem) > 0:
                    print(f"  Elements:")
                    for i, item in enumerate(elem[:10]):  # 最多显示10个元素
                        item_str = str(item)
                        if len(item_str) > 100:
                            item_str = item_str[:100] + "..."
                        print(f"    [{i}]: {type(item).__name__} {item_str}")
                    if len(elem) > 10:
                        print(f"    ... ({len(elem) - 10} more elements)")
            elif isinstance(elem, str):
                print(f"  Value: {elem}")
            else:
                elem_str = str(elem)
                if len(elem_str) > 200:
                    elem_str = elem_str[:200] + "..."
                print(f"  Value: {elem_str}")
        else:
            # 处理多维对象数组
            print(f"  Length: {len(arr)}")
            # 打印前几个元素的信息
            preview_count = min(3, len(arr))
            for i in range(preview_count):
                elem = arr[i]
                print(f"  Element {i}: {type(elem).__name__}")
                if isinstance(elem, dict):
                    print(f"    Keys: {list(elem.keys())}")
                    # 打印字典的前几个键值对
                    for j, (k, v) in enumerate(list(elem.items())[:3]):
                        print(f"      {k}: {type(v).__name__} {str(v)[:50]}")
                    if len(elem) > 3:
                        print(f"      ... ({len(elem) - 3} more keys)")
                elif isinstance(elem, (list, tuple)):
                    print(f"    Length: {len(elem)}")
                    if len(elem) > 0:
                        print(f"    First element: {type(elem[0]).__name__} {str(elem[0])[:50]}")
                else:
                    print(f"    Value preview: {str(elem)[:100]}")
            if len(arr) > preview_count:
                print(f"  ... ({len(arr) - preview_count} more elements)")
        
        # 打印内存占用
        size_mb = arr.nbytes / (1024 * 1024)
        print(f"  Memory size: {size_mb:.2f} MB")
        return
    
    # 对于数值类型数组，计算统计信息
    try:
        print(f"  Min: {np.min(arr)}")
        print(f"  Max: {np.max(arr)}")
        print(f"  Mean: {np.mean(arr):.4f}")
        print(f"  Std: {np.std(arr):.4f}")
    except (TypeError, ValueError) as e:
        print(f"  Warning: Cannot compute statistics: {e}")
    
    # 如果是整数类型，打印唯一值信息
    if np.issubdtype(arr.dtype, np.integer):
        try:
            unique_values = np.unique(arr)
            if len(unique_values) <= 256:  # 只显示前256个唯一值
                print(f"  Unique values: {unique_values}")
            else:
                print(f"  Unique values: {unique_values[:256]} ... ({len(unique_values)} total)")
            print(f"  Number of unique values: {len(unique_values)}")
        except (TypeError, ValueError):
            pass
    
    # 打印内存占用
    size_mb = arr.nbytes / (1024 * 1024)
    print(f"  Memory size: {size_mb:.2f} MB")


def print_spacing_info(spacing):
    """打印 spacing 信息"""
    print(f"\nspacing:")
    if isinstance(spacing, np.ndarray):
        print(f"  Type: numpy.ndarray")
        print(f"  Shape: {spacing.shape}")
        print(f"  Dtype: {spacing.dtype}")
        print(f"  Values: {spacing}")
        if len(spacing) == 3:
            print(f"  Format: (D={spacing[0]:.4f}, H={spacing[1]:.4f}, W={spacing[2]:.4f})")
    elif isinstance(spacing, (list, tuple)):
        print(f"  Type: {type(spacing).__name__}")
        print(f"  Length: {len(spacing)}")
        print(f"  Values: {spacing}")
        if len(spacing) == 3:
            print(f"  Format: (D={spacing[0]:.4f}, H={spacing[1]:.4f}, W={spacing[2]:.4f})")
    else:
        print(f"  Type: {type(spacing).__name__}")
        print(f"  Value: {spacing}")


def view_npz(npz_path):
    """查看 npz 文件的信息"""
    if not os.path.exists(npz_path):
        print(f"错误: 文件不存在: {npz_path}")
        return
    
    print(f"=" * 60)
    print(f"文件: {npz_path}")
    print(f"=" * 60)
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # 打印所有键
        keys = list(data.keys())
        print(f"\nKeys: {keys}")
        
        # 遍历每个键并打印信息
        for key in keys:
            value = data[key]
            
            if key == 'imgs':
                print_array_info("imgs", value)
            elif key == 'gts':
                print_array_info("gts", value)
            elif key == 'spacing':
                print_spacing_info(value)
            else:
                # 处理其他可能的键
                print(f"\n{key}:")
                if isinstance(value, np.ndarray):
                    print_array_info(f"  {key}", value)
                else:
                    print(f"  Type: {type(value).__name__}")
                    if isinstance(value, (list, tuple, dict)):
                        print(f"  Value: {value}")
                    else:
                        print(f"  Value: {value}")
        
        # 打印文件大小
        file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
        print(f"\n文件大小: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"错误: 无法读取文件: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="查看 .npz 文件的信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python gq_scripts/util/view.py data/MMs2/test/201_SA_ED.npz
        """
    )
    parser.add_argument("npz_file", type=str, help=".npz 文件路径")
    
    args = parser.parse_args()
    view_npz(args.npz_file)


if __name__ == "__main__":
    main()
