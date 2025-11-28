"""
将2个视频的面部图像拼接成一个视频
"""

import os
import cv2
import numpy as np
import argparse
import random
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from decord import VideoReader
from einops import rearrange

def load_video_using_decord(video_path, bbox_path, max_num_frames=81, target_fps=24):
    reader = VideoReader(video_path)
    video_length = len(reader)

    num_frames = (video_length - 1) // 4 * 4 + 1 if video_length < max_num_frames else max_num_frames
    start_idx = random.randint(0,video_length - num_frames - 1,) if video_length > num_frames else 0

    batch_idx = np.arange(start_idx, start_idx + num_frames)
    
    # 显式管理内存：分步处理避免临时变量积累
    raw_frames = reader.get_batch(batch_idx).asnumpy()
    frames = torch.from_numpy(raw_frames).permute(0, 3, 1, 2).contiguous()
    del raw_frames  # 立即释放numpy数组
    frames = frames / 255.0  # (f, c, h, w)

    h, w = frames.shape[-2:]
    face_mask_start, face_mask_end, face_center, bboxs, bbox_infos, face_mask_global = read_face_bbox(
        bbox_path, 
        h, 
        w, 
        video_length, 
        start_idx, 
        start_idx + num_frames,
        expand_ratio_w = random.uniform(0.3, 0.5), # 防止bbox过小, 做一些aug
        expand_ratio_h = random.uniform(0.2, 0.4) # 防止bbox过小, 做一些aug
    )
    # print(f"video length: {video_length}, bboxs length: {len(bboxs)}")
    # 保持原始视频尺寸，不进行裁剪
    # 裁剪将在concatenate_faces函数中进行
    print(f"视频原始尺寸: {frames.shape[-2]}x{frames.shape[-1]}")
    
    # 如果视频的原始fps与目标fps不同，进行帧率调整
    if hasattr(reader, 'get_avg_fps'):
        original_fps = reader.get_avg_fps()
        if abs(original_fps - target_fps) > 0.1:  # 允许小的误差
            print(f"视频原始fps: {original_fps}, 目标fps: {target_fps}")
            # 这里可以添加帧率转换逻辑，但decord本身不支持直接改变fps
            # 我们保持原始帧率，在最终保存时使用目标fps
    frames = rearrange(frames, "T C H W -> C T H W")

    # 显式释放VideoReader资源
    del reader
    
    # 返回起点、终点和全局的face_mask，用字典组织
    face_masks = {
        'start': face_mask_start,
        'end': face_mask_end,
        'global': face_mask_global
    }
    return frames, face_masks, start_idx, num_frames, bboxs, bbox_infos

def read_face_bbox(
    bboxs_path, 
    h, 
    w, 
    video_length = None, 
    start_idx = None, 
    end_idx = None,
    bbox_type = "xywh", 
    expand_ratio_w = 0.3,  # 宽度方向扩大30%
    expand_ratio_h = 0.2   # 高度方向扩大20%
):
    face_mask_start = None
    face_mask_end = None
    face_center = None
    bboxs = None
    bbox_infos = None
    if bboxs_path is not None:
        bboxs = np.load(bboxs_path)
        # print(f"video length: {video_length}, bboxs length: {len(bboxs)}")
        
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
            
            # print(f"视频帧范围: [{start_idx}, {end_idx}], bbox索引范围: [{bbox_start_idx}, {bbox_end_idx}]")
            
            # 获取起点和终点的bbox
            start_bbox = bboxs[bbox_start_idx]
            end_bbox = bboxs[bbox_end_idx]
            
            # 处理起点bbox
            if bbox_type == "xywh":
                start_x_min, start_y_min = start_bbox[0], start_bbox[1]
                start_x_max = start_bbox[2] + start_bbox[0]
                start_y_max = start_bbox[3] + start_bbox[1]
            elif bbox_type == "xxyy":
                start_x_min, start_y_min = start_bbox[0], start_bbox[1]
                start_x_max, start_y_max = start_bbox[2], start_bbox[3]
            
            # 扩大bbox以获得更好的覆盖（特别是宽度）
            start_width = start_x_max - start_x_min
            start_height = start_y_max - start_y_min
            start_center_x = (start_x_min + start_x_max) / 2
            start_center_y = (start_y_min + start_y_max) / 2
            
            # 重新计算扩大后的bbox
            expanded_width = start_width * (1 + 2 * expand_ratio_w)
            expanded_height = start_height * (1 + 2 * expand_ratio_h)
            start_x_min = max(0, start_center_x - expanded_width / 2)
            start_x_max = min(w, start_center_x + expanded_width / 2)
            start_y_min = max(0, start_center_y - expanded_height / 2)
            start_y_max = min(h, start_center_y + expanded_height / 2)
            
            start_ori_mask = torch.zeros((h, w))
            start_ori_mask[int(start_y_min):int(start_y_max), int(start_x_min):int(start_x_max)] = 1
            start_face_center = [(start_x_min + start_x_max)/2, (start_y_min + start_y_max)/2]
            # 保存相对坐标（比例），避免crop_and_resize后坐标系统不匹配
            start_bbox_info = {
                'center': [start_face_center[0] / w, start_face_center[1] / h],  # 相对坐标
                'width': (start_x_max - start_x_min) / w,  # 相对宽度
                'height': (start_y_max - start_y_min) / h,  # 相对高度
                'bbox': [start_x_min/w, start_y_min/h, start_x_max/w, start_y_max/h]  # 相对bbox
            }
            

            face_mask_start = crop_and_resize(start_ori_mask, face_center=start_face_center, normalize=False, interpolation=InterpolationMode.NEAREST).squeeze()
            del start_ori_mask  # 释放临时mask tensor
            
            # 处理终点bbox
            if bbox_type == "xywh":
                end_x_min, end_y_min = end_bbox[0], end_bbox[1]
                end_x_max = end_bbox[2] + end_bbox[0]
                end_y_max = end_bbox[3] + end_bbox[1]
            elif bbox_type == "xxyy":
                end_x_min, end_y_min = end_bbox[0], end_bbox[1]
                end_x_max, end_y_max = end_bbox[2], end_bbox[3]
            
            # 扩大end bbox（与start bbox相同的处理）
            end_width = end_x_max - end_x_min
            end_height = end_y_max - end_y_min
            end_center_x = (end_x_min + end_x_max) / 2
            end_center_y = (end_y_min + end_y_max) / 2
            
            # 重新计算扩大后的bbox
            expanded_width = end_width * (1 + 2 * expand_ratio_w)
            expanded_height = end_height * (1 + 2 * expand_ratio_h)
            end_x_min = max(0, end_center_x - expanded_width / 2)
            end_x_max = min(w, end_center_x + expanded_width / 2)
            end_y_min = max(0, end_center_y - expanded_height / 2)
            end_y_max = min(h, end_center_y + expanded_height / 2)
            
            end_ori_mask = torch.zeros((h, w))
            end_ori_mask[int(end_y_min):int(end_y_max), int(end_x_min):int(end_x_max)] = 1
            end_face_center = [(end_x_min + end_x_max)/2, (end_y_min + end_y_max)/2]
            end_bbox_info = {
                'center': [end_face_center[0] / w, end_face_center[1] / h],  # 相对坐标
                'width': (end_x_max - end_x_min) / w,  # 相对宽度
                'height': (end_y_max - end_y_min) / h,  # 相对高度
                'bbox': [end_x_min/w, end_y_min/h, end_x_max/w, end_y_max/h]  # 相对bbox
            }
            

            face_mask_end = crop_and_resize(end_ori_mask, face_center=end_face_center, normalize=False, interpolation=InterpolationMode.NEAREST).squeeze()
            del end_ori_mask  # 释放临时mask tensor
            
            # 使用起点的face_center作为默认值（向后兼容）
            face_center = start_face_center
            print(f"人脸中心点: ({face_center[0]:.1f}, {face_center[1]:.1f})")
            print(f"原始图像尺寸: {w}x{h}")
            print(f"起点bbox: [{start_x_min:.1f}, {start_y_min:.1f}, {start_x_max:.1f}, {start_y_max:.1f}]")
            
            # 计算全局bbox（整个序列的并集）- 使用更高效的实现
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
            
            # 创建全局mask
            global_ori_mask = torch.zeros((h, w))
            global_ori_mask[int(global_y_min):int(global_y_max), int(global_x_min):int(global_x_max)] = 1
            face_mask_global = crop_and_resize(global_ori_mask, face_center=global_face_center, normalize=False, interpolation=InterpolationMode.NEAREST).squeeze()
            del global_ori_mask  # 释放临时mask tensor
            
            # 将bbox信息打包返回
            bbox_infos = {
                'start': start_bbox_info,
                'end': end_bbox_info,
                'global': global_bbox_info  # 新增全局bbox
            }
        else:
            # 如果没有提供start_idx和end_idx，bbox_infos保持为None
            bbox_infos = None

    return face_mask_start, face_mask_end, face_center, bboxs, bbox_infos, face_mask_global if 'face_mask_global' in locals() else None


def crop_and_resize(
    image, 
    face_center=None, 
    normalize=True, 
    interpolation = InterpolationMode.BICUBIC,
    height = None,
    width = None
):

    if not isinstance(image, torch.Tensor):
        image = torchvision.transforms.functional.to_tensor(image)

    ori_width, ori_height = image.shape[-1], image.shape[-2]
    if image.ndim != 4:
        image = image.view(1, -1, ori_height, ori_width)

    # 如果没有指定目标尺寸，使用原始尺寸
    if height is None:
        height = ori_height
    if width is None:
        width = ori_width

    scale = max(width / ori_width, height / ori_height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(ori_height*scale), round(ori_width*scale)),
        interpolation=interpolation
    )
    if face_center is not None:
        cx, cy = face_center[0] * scale, face_center[1] * scale
        image = torchvision.transforms.functional.crop(
            image,
            top = min(max(0, round(cy - height/2)), image.shape[-2] - height),
            left = min(max(0, round(cx - width/2)), image.shape[-1] - width),
            height = height,
            width = width
        )
    else:
        image = torchvision.transforms.functional.center_crop(image, (height, width))
    
    if normalize:
        # 对于视频张量 (C, T, H, W)，需要分别对每一帧进行normalize
        if image.shape[1] > 3:  # 如果第二维是时间维度
            # 重新排列为 (T, C, H, W) 以便逐帧处理
            image = image.permute(1, 0, 2, 3)  # (T, C, H, W)
            
            # 对每一帧进行normalize
            for t in range(image.shape[0]):
                image[t] = torchvision.transforms.functional.normalize(
                    image[t],
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                )
            
            # 重新排列回 (C, T, H, W)
            image = image.permute(1, 0, 2, 3)  # (C, T, H, W)
        else:
            # 对于单帧图像，直接normalize
            image = torchvision.transforms.functional.normalize(
                image,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )

    return image

def debug_save_frame(video_tensor, output_path, frame_idx=0):
    """
    调试函数：保存单帧图像用于检查裁剪效果
    """
    import matplotlib.pyplot as plt
    
    # 获取指定帧
    frame = video_tensor[:, frame_idx, :, :].permute(1, 2, 0).cpu().numpy()
    
    # 反标准化
    frame = (frame + 1) * 127.5
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    # 保存图像
    plt.imsave(output_path, frame)
    print(f"调试帧已保存到: {output_path}")

def save_video_tensor_to_file(video_tensor, output_path, fps=24):
    """
    将视频tensor保存为视频文件，使用ffmpeg
    
    Args:
        video_tensor: 形状为 (C, T, H, W) 的视频tensor
        output_path: 输出视频路径
        fps: 帧率
    """
    import subprocess
    import tempfile
    import os
    
    # 确保tensor在CPU上并转换为numpy
    if video_tensor.is_cuda:
        video_tensor = video_tensor.cpu()
    
    # 转换为 (T, H, W, C) 格式
    video_np = video_tensor.permute(1, 2, 3, 0).numpy()
    
    # 反标准化（从[-1,1]转换到[0,255]）
    video_np = (video_np + 1) * 127.5
    video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    
    # 获取视频参数
    num_frames, height, width, channels = video_np.shape
    
    print(f"准备保存视频: 尺寸({width}x{height}), 帧数({num_frames}), fps({fps})")
    
    # 创建临时目录用于存储帧图像
    temp_dir = tempfile.mkdtemp(prefix="video_frames_")
    
    try:
        # 保存每一帧为PNG图像
        frame_paths = []
        for i in range(num_frames):
            frame = video_np[i]
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            
            # 使用matplotlib保存图像（保持RGB格式）
            import matplotlib.pyplot as plt
            plt.imsave(frame_path, frame)
            frame_paths.append(frame_path)
        
        print(f"已保存 {num_frames} 帧到临时目录: {temp_dir}")
        
        # 使用ffmpeg将帧序列转换为视频
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',  # 使用H.264编码
            '-preset', 'medium',  # 编码预设
            '-crf', '23',  # 恒定质量因子（18-28之间，数值越小质量越好）
            '-pix_fmt', 'yuv420p',  # 像素格式，确保兼容性
            '-movflags', '+faststart',  # 优化网络播放
            output_path
        ]
        
        print(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")
        
        # 执行ffmpeg命令
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"视频已成功保存到: {output_path}")
        
        # 可选：显示ffmpeg输出信息
        if result.stdout:
            print("ffmpeg输出:", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        raise
    except Exception as e:
        print(f"保存视频时发生错误: {e}")
        raise
    finally:
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"已清理临时目录: {temp_dir}")
        except Exception as e:
            print(f"清理临时文件时发生错误: {e}")

def save_video_tensor_to_file_efficient(video_tensor, output_path, fps=24):
    """
    将视频tensor保存为视频文件，使用ffmpeg管道方式（更高效）
    
    Args:
        video_tensor: 形状为 (C, T, H, W) 的视频tensor
        output_path: 输出视频路径
        fps: 帧率
    """
    import subprocess
    import numpy as np
    
    # 确保tensor在CPU上并转换为numpy
    if video_tensor.is_cuda:
        video_tensor = video_tensor.cpu()
    
    # 转换为 (T, H, W, C) 格式
    video_np = video_tensor.permute(1, 2, 3, 0).numpy()
    
    # 反标准化（从[-1,1]转换到[0,255]）
    video_np = (video_np + 1) * 127.5
    video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    
    # 获取视频参数
    num_frames, height, width, channels = video_np.shape
    
    print(f"准备保存视频（高效模式）: 尺寸({width}x{height}), 帧数({num_frames}), fps({fps})")
    
    # 使用ffmpeg管道方式
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',  # 从stdin读取
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]
    
    print(f"执行ffmpeg命令: {' '.join(ffmpeg_cmd)}")
    
    try:
        # 启动ffmpeg进程
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 逐帧写入数据
        for i in range(num_frames):
            frame = video_np[i]
            # 确保数据是连续的
            frame_bytes = frame.tobytes()
            process.stdin.write(frame_bytes)
        
        # 等待进程完成（这会自动关闭stdin）
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print(f"视频已成功保存到: {output_path}")
        else:
            print(f"ffmpeg执行失败，返回码: {process.returncode}")
            print(f"错误输出: {stderr.decode()}")
            raise subprocess.CalledProcessError(process.returncode, ffmpeg_cmd)
            
    except Exception as e:
        print(f"保存视频时发生错误: {e}")
        raise

def concatenate_faces(
    video_path_1, 
    video_path_2, 
    bbox_path_1, 
    bbox_path_2, 
    output_path, 
    fps=24,
    target_width=832,
    target_height=480,
    use_efficient_save=True
):
    """
    将两个视频的面部图像拼接成一个双人视频
    
    Args:
        video_path_1: 第一个视频路径
        video_path_2: 第二个视频路径  
        bbox_path_1: 第一个视频的bbox路径
        bbox_path_2: 第二个视频的bbox路径
        output_path: 输出视频路径
        fps: 输出视频帧率
        target_width: 目标视频宽度
        target_height: 目标视频高度
    """
    # 加载两个视频（不进行裁剪，保持原始尺寸）
    frames_1, face_masks_1, start_idx_1, num_frames_1, bboxs_1, bbox_infos_1 = load_video_using_decord(
        video_path_1, 
        bbox_path_1,
        target_fps=fps
    )
    
    frames_2, face_masks_2, start_idx_2, num_frames_2, bboxs_2, bbox_infos_2 = load_video_using_decord(
        video_path_2, 
        bbox_path_2,
        target_fps=fps
    )
    
    # 验证两个视频的fps是否相同
    reader1 = VideoReader(video_path_1)
    reader2 = VideoReader(video_path_2)
    
    fps1 = reader1.get_avg_fps() if hasattr(reader1, 'get_avg_fps') else None
    fps2 = reader2.get_avg_fps() if hasattr(reader2, 'get_avg_fps') else None
    
    if fps1 is not None and fps2 is not None:
        if abs(fps1 - fps2) > 0.1:  # 允许小的误差
            print(f"警告: 两个视频的fps不同 - 视频1: {fps1}, 视频2: {fps2}")
        else:
            print(f"两个视频fps相同: {fps1}")
    
    del reader1, reader2  # 释放资源
    
    # 确保两个视频帧数一致（取较短的视频长度）
    min_frames = min(frames_1.shape[1], frames_2.shape[1])
    print(f"视频1帧数: {frames_1.shape[1]}, 视频2帧数: {frames_2.shape[1]}, 使用帧数: {min_frames}")
    frames_1 = frames_1[:, :min_frames, :, :]
    frames_2 = frames_2[:, :min_frames, :, :]
    
    # 计算每个人占据的宽度（左右各占一半）
    person_width = target_width // 2
    person_height = target_height
    
    print(f"目标尺寸: {target_height}x{target_width}")
    print(f"每个人占据: {person_height}x{person_width}")
    
    # 使用类似a2v_dataset.py的crop_and_resize逻辑处理两个视频
    # 处理第一个视频（左半边）
    processed_frames_1 = crop_and_resize(
        frames_1, 
        face_center=None,  # 使用中心裁剪
        normalize=True,
        height=person_height,
        width=person_width
    )
    
    # 处理第二个视频（右半边）
    processed_frames_2 = crop_and_resize(
        frames_2, 
        face_center=None,  # 使用中心裁剪
        normalize=True,
        height=person_height,
        width=person_width
    )
    
    # 创建拼接后的视频帧
    concatenated_frames = torch.zeros(frames_1.shape[0], min_frames, target_height, target_width)
    
    # 拼接视频帧 - 左半边放第一个视频，右半边放第二个视频
    concatenated_frames[:, :, :, :person_width] = processed_frames_1
    concatenated_frames[:, :, :, person_width:] = processed_frames_2
    
    print(f"拼接信息:")
    print(f"  目标尺寸: {target_height}x{target_width}")
    print(f"  视频1处理后尺寸: {processed_frames_1.shape[-2]}x{processed_frames_1.shape[-1]}")
    print(f"  视频2处理后尺寸: {processed_frames_2.shape[-2]}x{processed_frames_2.shape[-1]}")
    print(f"  拼接后尺寸: {concatenated_frames.shape}")
    
    # 处理face_mask的拼接
    concatenated_face_masks = {}
    
    # 处理全局mask
    if face_masks_1['global'] is not None and face_masks_2['global'] is not None:
        # 对mask也进行相同的crop_and_resize处理
        mask1_processed = crop_and_resize(
            face_masks_1['global'].unsqueeze(0).unsqueeze(0),  # 添加batch和channel维度
            face_center=None,
            normalize=False,
            height=person_height,
            width=person_width
        ).squeeze()
        
        mask2_processed = crop_and_resize(
            face_masks_2['global'].unsqueeze(0).unsqueeze(0),  # 添加batch和channel维度
            face_center=None,
            normalize=False,
            height=person_height,
            width=person_width
        ).squeeze()
        
        # 创建拼接后的mask
        concatenated_global_mask = torch.zeros(target_height, target_width)
        concatenated_global_mask[:, :person_width] = mask1_processed
        concatenated_global_mask[:, person_width:] = mask2_processed
        
        concatenated_face_masks['global'] = concatenated_global_mask
        concatenated_face_masks['person1'] = concatenated_global_mask[:, :person_width]
        concatenated_face_masks['person2'] = concatenated_global_mask[:, person_width:]
    
    # 处理start和end mask
    if face_masks_1['start'] is not None and face_masks_2['start'] is not None:
        start1_processed = crop_and_resize(
            face_masks_1['start'].unsqueeze(0).unsqueeze(0),
            face_center=None,
            normalize=False,
            height=person_height,
            width=person_width
        ).squeeze()
        
        start2_processed = crop_and_resize(
            face_masks_2['start'].unsqueeze(0).unsqueeze(0),
            face_center=None,
            normalize=False,
            height=person_height,
            width=person_width
        ).squeeze()
        
        concatenated_start_mask = torch.zeros(target_height, target_width)
        concatenated_start_mask[:, :person_width] = start1_processed
        concatenated_start_mask[:, person_width:] = start2_processed
        concatenated_face_masks['start'] = concatenated_start_mask
    
    if face_masks_1['end'] is not None and face_masks_2['end'] is not None:
        end1_processed = crop_and_resize(
            face_masks_1['end'].unsqueeze(0).unsqueeze(0),
            face_center=None,
            normalize=False,
            height=person_height,
            width=person_width
        ).squeeze()
        
        end2_processed = crop_and_resize(
            face_masks_2['end'].unsqueeze(0).unsqueeze(0),
            face_center=None,
            normalize=False,
            height=person_height,
            width=person_width
        ).squeeze()
        
        concatenated_end_mask = torch.zeros(target_height, target_width)
        concatenated_end_mask[:, :person_width] = end1_processed
        concatenated_end_mask[:, person_width:] = end2_processed
        concatenated_face_masks['end'] = concatenated_end_mask
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 保存调试帧（可选）
    debug_dir = os.path.join(os.path.dirname(output_path), "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    debug_save_frame(processed_frames_1, os.path.join(debug_dir, "person1_frame0.png"), 0)
    debug_save_frame(processed_frames_2, os.path.join(debug_dir, "person2_frame0.png"), 0)
    debug_save_frame(concatenated_frames, os.path.join(debug_dir, "concatenated_frame0.png"), 0)
    
    # 保存拼接后的视频，使用指定的fps
    print(f"使用fps: {fps} 保存视频")
    if use_efficient_save:
        save_video_tensor_to_file_efficient(concatenated_frames, output_path, fps)
    else:
        save_video_tensor_to_file(concatenated_frames, output_path, fps)
    
    # 返回拼接后的数据
    result = {
        'concatenated_frames': concatenated_frames,
        'concatenated_face_masks': concatenated_face_masks,
        'person1_data': {
            'frames': processed_frames_1,
            'face_masks': face_masks_1,
            'start_idx': start_idx_1,
            'num_frames': num_frames_1,
            'bboxs': bboxs_1,
            'bbox_infos': bbox_infos_1
        },
        'person2_data': {
            'frames': processed_frames_2,
            'face_masks': face_masks_2,
            'start_idx': start_idx_2,
            'num_frames': num_frames_2,
            'bboxs': bboxs_2,
            'bbox_infos': bbox_infos_2
        }
    }
    
    # 处理bbox信息的拼接
    concatenated_bbox_infos = {}
    if bbox_infos_1 is not None and bbox_infos_2 is not None:
        # 调整第一个人的bbox信息（左半边）
        person1_bbox_infos = {}
        for key in ['start', 'end', 'global']:
            if key in bbox_infos_1:
                bbox_info = bbox_infos_1[key].copy()
                # 调整中心点x坐标（从整个图像坐标系转换到左半边坐标系）
                # 计算相对位置：person_width / target_width
                relative_width = person_width / target_width
                bbox_info['center'][0] = bbox_info['center'][0] * relative_width
                # 调整bbox坐标
                bbox_info['bbox'][0] = bbox_info['bbox'][0] * relative_width  # x_min
                bbox_info['bbox'][2] = bbox_info['bbox'][2] * relative_width  # x_max
                person1_bbox_infos[key] = bbox_info
        
        # 调整第二个人的bbox信息（右半边）
        person2_bbox_infos = {}
        for key in ['start', 'end', 'global']:
            if key in bbox_infos_2:
                bbox_info = bbox_infos_2[key].copy()
                # 调整中心点x坐标（从整个图像坐标系转换到右半边坐标系）
                # 计算相对位置：person_width / target_width 作为偏移
                relative_offset = person_width / target_width
                relative_width = person_width / target_width
                bbox_info['center'][0] = relative_offset + bbox_info['center'][0] * relative_width
                # 调整bbox坐标
                bbox_info['bbox'][0] = relative_offset + bbox_info['bbox'][0] * relative_width  # x_min
                bbox_info['bbox'][2] = relative_offset + bbox_info['bbox'][2] * relative_width  # x_max
                person2_bbox_infos[key] = bbox_info
        
        concatenated_bbox_infos = {
            'person1': person1_bbox_infos,
            'person2': person2_bbox_infos
        }
    
    # 更新结果字典
    result['concatenated_bbox_infos'] = concatenated_bbox_infos
    
    return result



def random_concat_test(jsonl_path, num_pairs=100, save_dir="/nfs/zzzhong/data/temp/concat_test/videos", 
                      base_dir="/nfs/datasets/datatang_session2"):
    """
    随机抽取视频对进行拼接测试
    
    Args:
        jsonl_path: jsonl文件路径
        num_pairs: 要测试的视频对数量
        save_dir: 保存拼接结果的目录
        base_dir: 数据集基础目录
    """
    import json
    import random
    import os
    from pathlib import Path
    
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取jsonl文件
    print(f"正在读取jsonl文件: {jsonl_path}")
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    # 解析所有视频信息
    videos = []
    for line in lines:
        try:
            video_info = json.loads(line.strip())
            videos.append(video_info)
        except json.JSONDecodeError as e:
            print(f"解析jsonl行时出错: {e}")
            continue
    
    print(f"总共读取到 {len(videos)} 个视频")
    
    # 随机抽取视频对
    if len(videos) < 2:
        print("视频数量不足，无法进行拼接测试")
        return
    
    # 随机选择视频对
    selected_pairs = []
    for i in range(num_pairs):
        # 随机选择两个不同的视频
        pair = random.sample(videos, 2)
        selected_pairs.append(pair)
    
    print(f"已随机选择 {len(selected_pairs)} 对视频进行测试")
    
    # 进行拼接测试
    success_count = 0
    failed_pairs = []
    
    for i, (video1, video2) in enumerate(selected_pairs):
        try:
            print(f"\n正在处理第 {i+1}/{len(selected_pairs)} 对视频:")
            print(f"  视频1: {video1['video']}")
            print(f"  视频2: {video2['video']}")
            
            # 构建完整路径
            video_path_1 = os.path.join(base_dir, video1['video'])
            video_path_2 = os.path.join(base_dir, video2['video'])
            bbox_path_1 = os.path.join(base_dir, video1['bboxs'])
            bbox_path_2 = os.path.join(base_dir, video2['bboxs'])
            
            # 检查文件是否存在
            if not all(os.path.exists(path) for path in [video_path_1, video_path_2, bbox_path_1, bbox_path_2]):
                print("  文件不存在，跳过此对")
                failed_pairs.append((video1, video2, "文件不存在"))
                continue
            
            # 生成输出文件名
            video1_name = os.path.splitext(os.path.basename(video1['video']))[0]
            video2_name = os.path.splitext(os.path.basename(video2['video']))[0]
            output_name = f"{video1_name}_{video2_name}.mp4"
            output_path = os.path.join(save_dir, output_name)
            
            # 进行拼接
            result = concatenate_faces(
                video_path_1, 
                video_path_2, 
                bbox_path_1, 
                bbox_path_2, 
                output_path,
                fps=16,
                target_width=832,
                target_height=480,
                use_efficient_save=True
            )
            
            print(f"  拼接成功: {output_name}")
            success_count += 1
            
        except Exception as e:
            print(f"  拼接失败: {str(e)}")
            failed_pairs.append((video1, video2, str(e)))
    
    # 输出测试结果
    print(f"\n=== 拼接测试结果 ===")
    print(f"总测试对数: {len(selected_pairs)}")
    print(f"成功对数: {success_count}")
    print(f"失败对数: {len(failed_pairs)}")
    print(f"成功率: {success_count/len(selected_pairs)*100:.2f}%")
    
    if failed_pairs:
        print(f"\n失败的对子:")
        for i, (video1, video2, error) in enumerate(failed_pairs[:10]):  # 只显示前10个
            print(f"  {i+1}. {os.path.basename(video1['video'])} + {os.path.basename(video2['video'])}: {error}")
        if len(failed_pairs) > 10:
            print(f"  ... 还有 {len(failed_pairs) - 10} 个失败的对子")

if __name__ == "__main__":
    # 原有的命令行参数处理代码保持不变
    parser = argparse.ArgumentParser(description="将两个视频的面部图像拼接成双人视频")
    parser.add_argument("--video_path_1", type=str, default="/nfs/zzzhong/codes/virtual_human/portrait_wan_14B/data/test_data/images_w_bbox/1.mp4", help="第一个视频路径")
    parser.add_argument("--video_path_2", type=str, default="/nfs/zzzhong/codes/virtual_human/portrait_wan_14B/data/test_data/images_w_bbox/5.mp4", help="第二个视频路径")
    parser.add_argument("--bbox_path_1", type=str, default="/nfs/zzzhong/codes/virtual_human/portrait_wan_14B/data/test_data/images_w_bbox/1.npy", help="第一个视频bbox路径")
    parser.add_argument("--bbox_path_2", type=str, default="/nfs/zzzhong/codes/virtual_human/portrait_wan_14B/data/test_data/images_w_bbox/5.npy", help="第二个视频bbox路径")
    parser.add_argument("--output_path", type=str, default="/nfs/zzzhong/data/temp/concat_test/1-5.mp4", help="输出视频路径")
    parser.add_argument("--fps", type=int, default=24, help="输出视频帧率")
    parser.add_argument("--target_width", type=int, default=832, help="输出视频宽度")
    parser.add_argument("--target_height", type=int, default=480, help="输出视频高度")
    parser.add_argument("--use_efficient_save", action="store_true", help="使用高效的ffmpeg管道保存方式")
    parser.add_argument("--random_test", action="store_true", help="进行随机拼接测试")
    parser.add_argument("--jsonl_path", type=str, default="/nfs/datasets/datatang_session2/metadata_wan_fps24.jsonl", help="jsonl文件路径")
    parser.add_argument("--num_pairs", type=int, default=100, help="随机测试的视频对数量")
    parser.add_argument("--save_dir", type=str, default="/nfs/zzzhong/data/temp/concat_test/videos", help="保存拼接结果的目录")
    args = parser.parse_args()
    
    if args.random_test:
        # 进行随机拼接测试
        random_concat_test(
            jsonl_path=args.jsonl_path,
            num_pairs=args.num_pairs,
            save_dir=args.save_dir
        )
    else:
        # 原有的单对视频拼接逻辑
        try:
            result = concatenate_faces(
                args.video_path_1, 
                args.video_path_2, 
                args.bbox_path_1, 
                args.bbox_path_2, 
                args.output_path,
                args.fps,
                args.target_width,
                args.target_height,
                args.use_efficient_save
            )
            print("视频拼接完成！")
            print(f"拼接后视频尺寸: {result['concatenated_frames'].shape}")
            print(f"第一个人的数据包含: {list(result['person1_data'].keys())}")
            print(f"第二个人的数据包含: {list(result['person2_data'].keys())}")
        except Exception as e:
            print(f"视频拼接失败: {str(e)}")
            import traceback
            traceback.print_exc()
        