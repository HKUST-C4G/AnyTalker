import json
import numpy as np
import subprocess
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.stats

jsonl_list = [
    "/nfs/datasets/abaka-0601/batch5_53h/metadata_wan_fps24.jsonl",
    "/nfs/datasets/abaka-0601/batch5_125h/metadata_wan_fps24.jsonl",
    "/nfs/datasets/abaka-0601/batch5_145h/metadata_wan_fps24.jsonl",
    "/nfs/datasets/abaka-0601/batch5_98h/metadata_wan_fps24.jsonl",
    "/nfs/datasets/new_abaka-0505/metadata_wan_fps24.jsonl",
    "/nfs/datasets/abaka-0509/metadata_wan_fps24.jsonl",
    "/nfs/datasets/abaka-0511/metadata_wan_fps24.jsonl",
    "/nfs/datasets/batch3/metadata_wan_fps24.jsonl",
    "/nfs/datasets/batch4/metadata_wan_fps24.jsonl",
    "/nfs/datasets/hallo3/metadata_wan_fps24.jsonl",
    "/nfs/datasets/CelebV-HQ/metadata_wan_fps24.jsonl ",
    "/nfs/datasets/batch6/metadata_wan_fps24.jsonl",
    "/nfs/datasets/abaka-sing-0703/metadata_wan_fps24.jsonl",
    "/nfs/datasets/datatang/metadata_wan_fps24.jsonl"
]

def get_video_dimensions(video_path):
    """使用ffmpeg获取视频的宽度和高度"""
    try:
        # 构建ffmpeg命令来获取视频信息
        cmd = [
            'ffprobe', 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_streams', 
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        
        # 查找视频流
        for stream in video_info['streams']:
            if stream['codec_type'] == 'video':
                width = int(stream['width'])
                height = int(stream['height'])
                return width, height
                
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print(f"获取视频尺寸失败 {video_path}: {e}")
        return None, None
    
    return None, None

def read_face_bbox(
        bboxs_path, 
        h, 
        w, 
        video_length = None, 
        start_idx = None, 
        end_idx = None,
        bbox_type = "xywh", 
    ):
    face_mask_start = None
    face_mask_end = None
    face_center = None
    bboxs = None
    bbox_infos = None
    if bboxs_path is not None:
        bboxs = np.load(bboxs_path)
        
        if start_idx is not None and end_idx is not None:
            # 计算视频选取的帧数
            video_frames = end_idx - start_idx
            
            # 将视频的起点和终点映射到bbox序列
            if len(bboxs) == 1:
                # 如果只有一个bbox，起点和终点都用这个
                bbox_start_idx = 0
                bbox_end_idx = 0
            else:
                # 均匀映射：将视频起点终点映射到bbox序列
                bbox_start_idx = int(start_idx * (len(bboxs) - 1) / (video_length - 1)) if video_length > 1 else 0
                bbox_end_idx = int(end_idx * (len(bboxs) - 1) / (video_length - 1)) if video_length > 1 else 0
                bbox_start_idx = min(bbox_start_idx, len(bboxs) - 1)
                bbox_end_idx = min(bbox_end_idx, len(bboxs) - 1)
            
            # 获取序列中所有相关帧的bbox
            relevant_start_idx = 0
            relevant_end_idx = len(bboxs) - 1 
            # 提取相关的bbox序列
            relevant_bboxs = bboxs[relevant_start_idx:relevant_end_idx + 1]
            
            # 使用高效的方式计算全局边界（并集）
            global_x_min = relevant_bboxs[:, 0].min()
            global_y_min = relevant_bboxs[:, 1].min()
            if bbox_type == "xywh":
                global_x_max = (relevant_bboxs[:, 2] + relevant_bboxs[:, 0]).max()
                global_y_max = (relevant_bboxs[:, 3] + relevant_bboxs[:, 1]).max()
            elif bbox_type == "xxyy":
                global_x_max = relevant_bboxs[:, 2].max()
                global_y_max = relevant_bboxs[:, 3].max()
            
            # 不对全局bbox进行扩展
            global_width = global_x_max - global_x_min
            global_height = global_y_max - global_y_min
            global_center_x = (global_x_min + global_x_max) / 2
            global_center_y = (global_y_min + global_y_max) / 2
            
            # 计算全局bbox
            global_x_min = max(0, global_center_x - global_width / 2)
            global_x_max = min(w, global_center_x + global_width / 2)
            global_y_min = max(0, global_center_y - global_height / 2)
            global_y_max = min(h, global_center_y + global_height / 2)
            
            # 创建全局bbox信息
            global_face_center = [(global_x_min + global_x_max)/2, (global_y_min + global_y_max)/2]
            global_bbox_info = {
                'center': [global_face_center[0] / w, global_face_center[1] / h],  # 相对坐标
                'width': (global_x_max - global_x_min) / w,  # 相对宽度
                'height': (global_y_max - global_y_min) / h,  # 相对高度
                'bbox': [global_x_min/w, global_y_min/h, global_x_max/w, global_y_max/h]  # 相对bbox
            }
    
    return bboxs, bbox_infos

def plot_probability_density_distributions(all_widths, all_heights, all_areas, all_relative_widths, all_relative_heights, all_relative_areas):
    """Plot probability density distributions"""
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('BBox Probability Density Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Absolute size distributions
    # Width distribution
    axes[0, 0].hist(all_widths, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    kde_x = np.linspace(min(all_widths), max(all_widths), 1000)
    kde = scipy.stats.gaussian_kde(all_widths)
    axes[0, 0].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
    axes[0, 0].set_title('Absolute Width Distribution')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Height distribution
    axes[0, 1].hist(all_heights, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    kde_x = np.linspace(min(all_heights), max(all_heights), 1000)
    kde = scipy.stats.gaussian_kde(all_heights)
    axes[0, 1].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
    axes[0, 1].set_title('Absolute Height Distribution')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Area distribution
    axes[0, 2].hist(all_areas, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    kde_x = np.linspace(min(all_areas), max(all_areas), 1000)
    kde = scipy.stats.gaussian_kde(all_areas)
    axes[0, 2].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
    axes[0, 2].set_title('Absolute Area Distribution')
    axes[0, 2].set_xlabel('Area (pixels²)')
    axes[0, 2].set_ylabel('Probability Density')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 2. Relative size distributions
    # Relative width distribution
    axes[1, 0].hist(all_relative_widths, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    kde_x = np.linspace(min(all_relative_widths), max(all_relative_widths), 1000)
    kde = scipy.stats.gaussian_kde(all_relative_widths)
    axes[1, 0].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
    axes[1, 0].set_title('Relative Width Distribution')
    axes[1, 0].set_xlabel('Relative Width (ratio)')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Relative height distribution
    axes[1, 1].hist(all_relative_heights, bins=50, density=True, alpha=0.7, color='plum', edgecolor='black')
    kde_x = np.linspace(min(all_relative_heights), max(all_relative_heights), 1000)
    kde = scipy.stats.gaussian_kde(all_relative_heights)
    axes[1, 1].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
    axes[1, 1].set_title('Relative Height Distribution')
    axes[1, 1].set_xlabel('Relative Height (ratio)')
    axes[1, 1].set_ylabel('Probability Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Relative area distribution
    axes[1, 2].hist(all_relative_areas, bins=50, density=True, alpha=0.7, color='gold', edgecolor='black')
    kde_x = np.linspace(min(all_relative_areas), max(all_relative_areas), 1000)
    kde = scipy.stats.gaussian_kde(all_relative_areas)
    axes[1, 2].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
    axes[1, 2].set_title('Relative Area Distribution')
    axes[1, 2].set_xlabel('Relative Area (ratio)')
    axes[1, 2].set_ylabel('Probability Density')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bbox_probability_density_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_bbox_distribution():
    """分析所有jsonl文件中bbox的分布情况"""
    all_widths = []
    all_heights = []
    all_areas = []
    all_relative_widths = []
    all_relative_heights = []
    all_relative_areas = []
    
    total_processed = 0
    total_errors = 0
    
    for jsonl_path in tqdm(jsonl_list, desc="处理数据集文件"):
        if not os.path.exists(jsonl_path):
            print(f"文件不存在: {jsonl_path}")
            continue
            
        # 先计算文件行数
        with open(jsonl_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        with open(jsonl_path, 'r') as f:
            for line_num, line in tqdm(enumerate(f, 1), total=total_lines, desc="处理行", leave=False):
                try:
                    data = json.loads(line.strip())
                    
                    # 获取视频路径和bbox路径
                    video_path = data.get('video')
                    bboxs_path = data.get('bboxs')
                    width = data.get('width')
                    height = data.get('height')
                    
                    if not all([video_path, bboxs_path]):
                        continue
                    
                    # 如果jsonl中没有width/height信息，使用ffmpeg获取
                    if width is None or height is None:
                        full_video_path = os.path.join(os.path.dirname(jsonl_path), video_path)
                        width, height = get_video_dimensions(full_video_path)
                        if width is None or height is None:
                            print(f"无法获取视频尺寸: {full_video_path}")
                            total_errors += 1
                            continue
                    
                    # 加载bbox数据
                    full_bbox_path = os.path.join(os.path.dirname(jsonl_path), bboxs_path)
                    if not os.path.exists(full_bbox_path):
                        print(f"bbox文件不存在: {full_bbox_path}")
                        total_errors += 1
                        continue
                    
                    bboxs = np.load(full_bbox_path)
                    
                    # 计算每个bbox的统计信息
                    for bbox in bboxs:
                        if len(bbox) >= 4:
                            x, y, w_bbox, h_bbox = bbox[:4]
                            
                            # 绝对尺寸（像素）
                            abs_width = w_bbox
                            abs_height = h_bbox
                            abs_area = abs_width * abs_height
                            
                            # 相对尺寸（占图像的比例）
                            rel_width = abs_width / width
                            rel_height = abs_height / height
                            rel_area = rel_width * rel_height
                            
                            # 添加到全局统计
                            all_widths.append(abs_width)
                            all_heights.append(abs_height)
                            all_areas.append(abs_area)
                            all_relative_widths.append(rel_width)
                            all_relative_heights.append(rel_height)
                            all_relative_areas.append(rel_area)
                    
                    total_processed += 1
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误 {jsonl_path}:{line_num}: {e}")
                    total_errors += 1
                except Exception as e:
                    print(f"处理错误 {jsonl_path}:{line_num}: {e}")
                    total_errors += 1
    
    # 打印统计结果
    print(f"\n=== 总体统计 ===")
    print(f"总处理样本数: {total_processed}")
    print(f"总错误数: {total_errors}")
    print(f"总bbox数: {len(all_widths)}")
    
    if all_widths:
        print(f"\n=== 绝对尺寸统计（像素） ===")
        print(f"宽度 - 均值: {np.mean(all_widths):.2f}, 中位数: {np.median(all_widths):.2f}, 标准差: {np.std(all_widths):.2f}")
        print(f"高度 - 均值: {np.mean(all_heights):.2f}, 中位数: {np.median(all_heights):.2f}, 标准差: {np.std(all_heights):.2f}")
        print(f"面积 - 均值: {np.mean(all_areas):.2f}, 中位数: {np.median(all_areas):.2f}, 标准差: {np.std(all_areas):.2f}")
        
        print(f"\n=== 相对尺寸统计（占图像比例） ===")
        print(f"相对宽度 - 均值: {np.mean(all_relative_widths):.4f}, 中位数: {np.median(all_relative_widths):.4f}, 标准差: {np.std(all_relative_widths):.4f}")
        print(f"相对高度 - 均值: {np.mean(all_relative_heights):.4f}, 中位数: {np.median(all_relative_heights):.4f}, 标准差: {np.std(all_relative_heights):.4f}")
        print(f"相对面积 - 均值: {np.mean(all_relative_areas):.6f}, 中位数: {np.median(all_relative_areas):.6f}, 标准差: {np.std(all_relative_areas):.6f}")
    
    # 绘制概率密度分布图
    print(f"\n=== 绘制概率密度分布图 ===")
    if all_widths:
        plot_probability_density_distributions(all_widths, all_heights, all_areas, all_relative_widths, all_relative_heights, all_relative_areas)
    
    # 保存统计结果
    results = {
        'total_samples': total_processed,
        'total_errors': total_errors,
        'total_bboxes': len(all_widths),
        'absolute_stats': {
            'widths': {'mean': float(np.mean(all_widths)), 'median': float(np.median(all_widths)), 'std': float(np.std(all_widths))},
            'heights': {'mean': float(np.mean(all_heights)), 'median': float(np.median(all_heights)), 'std': float(np.std(all_heights))},
            'areas': {'mean': float(np.mean(all_areas)), 'median': float(np.median(all_areas)), 'std': float(np.std(all_areas))}
        },
        'relative_stats': {
            'widths': {'mean': float(np.mean(all_relative_widths)), 'median': float(np.median(all_relative_widths)), 'std': float(np.std(all_relative_widths))},
            'heights': {'mean': float(np.mean(all_relative_heights)), 'median': float(np.median(all_relative_heights)), 'std': float(np.std(all_relative_heights))},
            'areas': {'mean': float(np.mean(all_relative_areas)), 'median': float(np.median(all_relative_areas)), 'std': float(np.std(all_relative_areas))}
        }
    }
    
    print(f"\n保存统计结果...")
    with open('bbox_distribution_stats.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"统计结果已保存到: bbox_distribution_stats.json")
    print(f"概率密度分布图已保存到: bbox_probability_density_distributions.png")

if __name__ == "__main__":
    # 运行完整的分析（包括概率密度分布图）
    analyze_bbox_distribution()