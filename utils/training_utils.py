import os
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from safetensors.torch import load_file
import random
import glob
import numpy as np
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image


def init_distributed(args):
    """初始化分布式训练环境"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    sp_size = 1
    dp_size = world_size

    if args.use_usp:
        sp_size = args.ulysses_degree * args.ring_degree
        dp_size = world_size // sp_size
        assert sp_size <= world_size, f"sequence parallel size ({sp_size}) must be less than or equal to world size ({world_size})."
        assert world_size % sp_size == 0, f"world size ({world_size}) must be divisible by sequence parallel size ({sp_size})."
        
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment, get_data_parallel_rank
        init_distributed_environment(rank=global_rank, world_size=world_size)
        initialize_model_parallel(
            data_parallel_degree=dp_size,
            sequence_parallel_degree=sp_size,
            ring_degree=args.ring_degree,
            ulysses_degree=args.ulysses_degree,
        )

        if dist.is_initialized():
            sp_group_id = get_data_parallel_rank()
            if global_rank == 0:
                base_seed = torch.randint(0, 1000000, (1,)).item()
            else:
                base_seed = 0
            base_seed = torch.tensor([base_seed], device="cuda")
            dist.broadcast(base_seed, src=0)
            base_seed = base_seed.item()
            seed = base_seed + sp_group_id
            torch.manual_seed(seed)

    return global_rank, world_size, local_rank, sp_size, dp_size


def custom_collate_fn(batch):
    """自定义数据整理函数，用于处理不同长度的视频"""
    keys = batch[0].keys()
    data = {}
    batch_video_length = min([item["video"].shape[1] for item in batch])
    for key in keys:
        if key=='video': # 截断视频长度
            data[key] = [item[key][:,:batch_video_length] for item in batch]
        elif key == 'audio_emb':
            data[key] = [item[key][:batch_video_length] for item in batch]
        elif key == 'audio_embed_speaker1':
            data[key] = [item[key][:batch_video_length] for item in batch]
        elif key == 'audio_embed_speaker2':
            data[key] = [item[key][:batch_video_length] for item in batch]
        else:
            data[key] = [item[key] for item in batch]
    return data


def get_distributed_sampler(dataset, args, global_rank, world_size, sp_size, dp_size):
    """获取分布式采样器"""
    if dp_size == 1:
        sampler = None
    else:
        if args.use_usp:
            from xfuser.core.distributed import get_data_parallel_rank
            dp_rank = get_data_parallel_rank()
        else:
            dp_rank = global_rank  # 避免调用 _DP 相关内容

        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=False
        )

        if global_rank == 0:
            print(f"Using DistributedSampler: dp_size={dp_size}, dp_rank={dp_rank}")

    return sampler


def load_predefined_prompt_embeddings(base_dir="/nfs/datasets/multi_person/prompt_emb_concat", text_len=512):
    """
    加载三种预定义的prompt embeddings
    
    Args:
        base_dir: 预定义embedding的基础目录
        text_len: 文本长度，用于padding或截断
    
    Returns:
        dict: 包含三种类型embedding的字典
        {
            'talk_prompts': list of tensors,
            'silent_prompts_left': list of tensors, 
            'silent_prompts_right': list of tensors
        }
    """
    prompt_embeddings = {
        'talk_prompts': [],
        'silent_prompts_left': [],
        'silent_prompts_right': []
    }
    
    # 定义三个子目录
    subdirs = {
        'talk_prompts': 'talk_prompts',
        'silent_prompts_left': 'silent_prompts_left', 
        'silent_prompts_right': 'silent_prompts_right'
    }
    
    for key, subdir in subdirs.items():
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist")
            continue
            
        # 获取所有.safetensors文件
        pattern = os.path.join(dir_path, "*.safetensors")
        files = sorted(glob.glob(pattern))
        
        for file_path in files:
            try:
                # 加载embedding
                prompt_data = load_file(file_path)
                prompt_emb = prompt_data['context']
                
                # 处理长度
                if prompt_emb.shape[0] < text_len:
                    padding = torch.zeros(text_len - prompt_emb.shape[0], prompt_emb.shape[1])
                    prompt_emb = torch.cat([prompt_emb, padding], dim=0)
                else:
                    prompt_emb = prompt_emb[:text_len]
                
                prompt_embeddings[key].append(prompt_emb)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    print(f"Loaded {len(prompt_embeddings['talk_prompts'])} talk prompts")
    print(f"Loaded {len(prompt_embeddings['silent_prompts_left'])} silent left prompts") 
    print(f"Loaded {len(prompt_embeddings['silent_prompts_right'])} silent right prompts")
    
    return prompt_embeddings


def get_random_prompt_embedding(prompt_embeddings, prompt_type=None, device=None, dtype=None):
    """
    从预定义的prompt embeddings中随机选择一个
    
    Args:
        prompt_embeddings: 由load_predefined_prompt_embeddings返回的字典
        prompt_type: 指定类型 ('talk_prompts', 'silent_prompts_left', 'silent_prompts_right')
                    如果为None，则随机选择类型
        device: 目标设备
        dtype: 目标数据类型
    
    Returns:
        torch.Tensor: 随机选择的prompt embedding
    """
    if prompt_type is None:
        # 随机选择类型
        available_types = [k for k, v in prompt_embeddings.items() if len(v) > 0]
        if not available_types:
            raise ValueError("No prompt embeddings available")
        prompt_type = random.choice(available_types)
    
    if prompt_type not in prompt_embeddings:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    if len(prompt_embeddings[prompt_type]) == 0:
        raise ValueError(f"No embeddings available for type: {prompt_type}")
    
    # 随机选择一个embedding
    selected_embedding = random.choice(prompt_embeddings[prompt_type])
    
    # 移动到指定设备和类型
    if device is not None:
        selected_embedding = selected_embedding.to(device)
    if dtype is not None:
        selected_embedding = selected_embedding.to(dtype)
    
    return selected_embedding


def create_silence_video(video_tensor, cycle_frames=3):
    """
    创建沉默视频：使用前N帧循环播放，支持倒序播放模式
    
    Args:
        video_tensor: [b, 3, F, H, W] 原始视频tensor
        cycle_frames: int, 循环播放的帧数，默认为2
        
    Returns:
        silence_video: [b, 3, F, H, W] 沉默视频tensor
        
    播放模式：
    - cycle_frames=2: 12121212... (原逻辑)
    - cycle_frames=3: 123123123...
    - cycle_frames=4: 1234543212345... (倒序模式)
    - cycle_frames=5: 1234543212345...
    """
    batch_size, channels, num_frames, height, width = video_tensor.shape
    
    # 确保cycle_frames不超过视频总帧数
    cycle_frames = min(cycle_frames, num_frames)
    
    # 获取前cycle_frames帧
    cycle_video = video_tensor[:, :, :cycle_frames, :, :]  # [b, 3, cycle_frames, H, W]
    
    if cycle_frames <= 2:
        # 对于1-2帧，使用简单的交替模式
        if cycle_frames == 1:
            # 单帧重复
            silence_video = cycle_video.repeat(1, 1, num_frames, 1, 1)
        else:
            # 双帧交替：12121212...
            frame_1 = cycle_video[:, :, 0:1, :, :]  # [b, 3, 1, H, W]
            frame_2 = cycle_video[:, :, 1:2, :, :]  # [b, 3, 1, H, W]
            
            repeat_times = (num_frames + 1) // 2
            frame_1_repeated = frame_1.repeat(1, 1, repeat_times, 1, 1)
            frame_2_repeated = frame_2.repeat(1, 1, repeat_times, 1, 1)
            
            silence_frames = torch.stack([frame_1_repeated, frame_2_repeated], dim=3)
            silence_frames = silence_frames.view(batch_size, channels, repeat_times * 2, height, width)
            silence_video = silence_frames[:, :, :num_frames, :, :]
    else:
        # 对于3帧以上，使用倒序模式：1234543212345...
        # 创建一个完整的循环周期：123...cycle_frames...321
        forward_frames = cycle_video  # [b, 3, cycle_frames, H, W]
        reverse_frames = torch.flip(cycle_video[:, :, 1:-1, :, :], dims=[2])  # [b, 3, cycle_frames-2, H, W]
        
        # 拼接一个完整周期：123...cycle_frames...321
        one_cycle = torch.cat([forward_frames, reverse_frames], dim=2)  # [b, 3, 2*cycle_frames-2, H, W]
        cycle_length = 2 * cycle_frames - 2
        
        # 计算需要多少个完整周期
        num_cycles = (num_frames + cycle_length - 1) // cycle_length
        
        # 重复完整周期
        repeated_cycles = one_cycle.repeat(1, 1, num_cycles, 1, 1)  # [b, 3, num_cycles*cycle_length, H, W]
        
        # 截断到所需帧数
        silence_video = repeated_cycles[:, :, :num_frames, :, :]  # [b, 3, F, H, W]
    
    return silence_video


def extract_square_faces_from_ref_images(face_parser, ref_images, video_paths, first_frame_faces, crop_size=224, device=None, torch_dtype=None, global_rank=0, current_global_step=0):
    """
    使用人脸解析器从参考图像中提取方形人脸，如果检测不到人脸则回退到dataset提供的crop人脸
    
    Args:
        face_parser: FaceInference实例
        ref_images: list of tensors, 每个tensor shape: [3, H, W]
        video_paths: list of str, 对应的视频路径
        first_frame_faces: tensor, dataset提供的crop人脸 [b, 3, face_H, face_W]
        crop_size: int, 裁剪后的人脸尺寸
        device: torch设备
        torch_dtype: torch数据类型
        global_rank: 全局rank，用于控制日志输出
        current_global_step: 当前训练步数
        
    Returns:
        list of tensors: 每个tensor shape: [3, crop_size, crop_size]
    """
    square_faces = []
    
    for i, (ref_image, video_path) in enumerate(zip(ref_images, video_paths)):
        try:
            # 将tensor转换为numpy数组用于人脸检测
            # ref_image shape: [3, H, W], 需要转换为 [H, W, 3]
            # 确保转换为float32类型，避免BFloat16问题
            ref_image_np = ref_image.permute(1, 2, 0).cpu().float().numpy()
            
            # 处理值域：如果值域是[-1,1]，则转换到[0,1]；如果已经是[0,1]，则直接使用
            if ref_image_np.min() < 0:
                # 从[-1,1]转换到[0,1]
                ref_image_np = (ref_image_np + 1) / 2
            ref_image_np = np.clip(ref_image_np, 0, 1)
            
            # 转换为uint8格式
            ref_image_np = (ref_image_np * 255).astype(np.uint8)
            
            # # 添加调试信息
            # if global_rank == 0 and current_global_step % 100 == 0:
            #     print(f"[Face Parser] 原始ref_image值域: [{ref_image.min():.4f}, {ref_image.max():.4f}]")
            #     print(f"[Face Parser] 转换后numpy数组值域: [{ref_image_np.min()}, {ref_image_np.max()}]")
            #     print(f"[Face Parser] 转换后numpy数组形状: {ref_image_np.shape}")
            
            # 使用人脸解析器检测人脸
            face_result = face_parser.infer_from_array(ref_image_np, n=1)  # 只取最大的人脸
            
            if face_result and 'bboxes' in face_result and len(face_result['bboxes']) > 0:
                # 获取第一个（最大的）人脸的bbox
                bbox = face_result['bboxes'][0]  # [x, y, width, height]
                x, y, w, h = bbox
                
                # 从原图中裁剪人脸区域
                face_crop = ref_image_np[int(y):int(y+h), int(x):int(x+w)]
                
                # # 添加调试信息
                # if global_rank == 0 and current_global_step % 100 == 0:
                #     print(f"[Face Parser] 成功提取人脸 {i}: bbox=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
                #     print(f"[Face Parser] 裁剪后的人脸形状: {face_crop.shape}")
                #     print(f"[Face Parser] 裁剪后的人脸值域: [{face_crop.min()}, {face_crop.max()}]")
                
                # 将numpy数组转换回tensor并调整尺寸
                face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
                face_tensor = resize(face_tensor, size=(crop_size, crop_size), interpolation=InterpolationMode.BILINEAR)
                
                # # 添加更多调试信息
                # if global_rank == 0 and current_global_step % 100 == 0:
                #     print(f"[Face Parser] 调整尺寸后的tensor形状: {face_tensor.shape}")
                #     print(f"[Face Parser] 调整尺寸后的tensor值域: [{face_tensor.min():.4f}, {face_tensor.max():.4f}]")
                
                square_faces.append(face_tensor.to(device, dtype=torch_dtype))
                    
            else:
                # 如果没有检测到人脸，回退到使用dataset提供的crop人脸
                dataset_face = first_frame_faces[i]  # [3, face_H, face_W]
                
                # 添加调试信息
                if global_rank == 0 :
                    print(f"[Face Parser] 未检测到人脸 {i}，回退到dataset提供的crop人脸")
                    print(f"[Face Parser] Dataset人脸形状: {dataset_face.shape}")
                    print(f"[Face Parser] Dataset人脸值域: [{dataset_face.min():.4f}, {dataset_face.max():.4f}]")
                
                face_tensor = resize(dataset_face, size=(crop_size, crop_size), interpolation=InterpolationMode.BILINEAR)
                
                # 添加更多调试信息
                if global_rank == 0:
                    print(f"[Face Parser] 调整尺寸后的dataset人脸形状: {face_tensor.shape}")
                    print(f"[Face Parser] 调整尺寸后的dataset人脸值域: [{face_tensor.min():.4f}, {face_tensor.max():.4f}]")
                
                square_faces.append(face_tensor.to(device, dtype=torch_dtype))
                    
        except Exception as e:
            # 异常情况下回退到使用dataset提供的crop人脸
            dataset_face = first_frame_faces[i]  # [3, face_H, face_W]
            
            # 添加调试信息
            if global_rank == 0:
                print(f"[Face Parser] 处理图像 {i} 时出错: {str(e)}，回退到dataset提供的crop人脸")
                print(f"[Face Parser] 异常回退 - Dataset人脸形状: {dataset_face.shape}")
                print(f"[Face Parser] 异常回退 - Dataset人脸值域: [{dataset_face.min():.4f}, {dataset_face.max():.4f}]")
            
            face_tensor = resize(dataset_face, size=(crop_size, crop_size), interpolation=InterpolationMode.BILINEAR)
            
            # 添加更多调试信息
            if global_rank == 0:
                print(f"[Face Parser] 异常回退 - 调整尺寸后的tensor形状: {face_tensor.shape}")
                print(f"[Face Parser] 异常回退 - 调整尺寸后的tensor值域: [{face_tensor.min():.4f}, {face_tensor.max():.4f}]")
            
            square_faces.append(face_tensor.to(device, dtype=torch_dtype))
    
    return square_faces


def save_face_parser_debug_images(ref_cropped_list, ref_image, video_paths, first_frame_faces, current_step, crop_image_size, debug_dir="/nfs/zzzhong/codes/virtual_human/portrait_wan_14B/logs/debug/ref", global_rank=0):
    """
    保存人脸解析器的调试图像到debug目录
    
    Args:
        ref_cropped_list: list of tensors, 人脸解析器生成的方形人脸
        ref_image: tensor, 原始参考图像
        video_paths: list of str, 视频路径
        first_frame_faces: tensor, dataset提供的crop人脸
        current_step: int, 当前训练步数
        crop_image_size: int, 裁剪图像尺寸
        debug_dir: str, 调试目录路径
    """
    try:
        # 只在rank 0时创建debug目录
        if global_rank == 0:
            os.makedirs(debug_dir, exist_ok=True)
            os.chmod(debug_dir, 0o777)
        
        # 保存每个batch的调试图像
        for batch_idx, (ref_cropped, video_path) in enumerate(zip(ref_cropped_list, video_paths)):
            # 只在rank 0时创建子目录
            if global_rank == 0:
                batch_dir = os.path.join(debug_dir, f"step_{current_step}_batch_{batch_idx}")
                os.makedirs(batch_dir, exist_ok=True)
                os.chmod(batch_dir, 0o777)
            
            # 保存原始参考图像
            if ref_image is not None:
                ref_img_single = ref_image[batch_idx]  # [3, H, W]
                # 处理值域：如果值域是[-1,1]，则转换到[0,1]；如果已经是[0,1]，则直接使用
                if ref_img_single.min() < 0:
                    # 从[-1,1]转换到[0,1]
                    ref_img_single = (ref_img_single + 1) / 2
                ref_img_single = torch.clamp(ref_img_single, 0, 1)
                ref_img_path = os.path.join(batch_dir, "original_ref_image.png")
                save_image(ref_img_single, ref_img_path)
            
            # 保存dataset提供的原始crop人脸
            if first_frame_faces is not None:
                dataset_face_single = first_frame_faces[batch_idx]  # [3, face_H, face_W]
                # 处理值域：如果值域是[-1,1]，则转换到[0,1]；如果已经是[0,1]，则直接使用
                if dataset_face_single.min() < 0:
                    # 从[-1,1]转换到[0,1]
                    dataset_face_single = (dataset_face_single + 1) / 2
                dataset_face_single = torch.clamp(dataset_face_single, 0, 1)
                dataset_face_path = os.path.join(batch_dir, "dataset_crop_face.png")
                save_image(dataset_face_single, dataset_face_path)
            
            # 保存人脸解析器生成的方形人脸
            if ref_cropped is not None:
                ref_cropped_single = ref_cropped[batch_idx]  # [3, crop_size, crop_size]
                
                # 添加调试信息
                print(f"[Debug] Parsed face tensor shape: {ref_cropped_single.shape}")
                print(f"[Debug] Parsed face tensor dtype: {ref_cropped_single.dtype}")
                print(f"[Debug] Parsed face tensor device: {ref_cropped_single.device}")
                print(f"[Debug] Parsed face tensor stats: min={ref_cropped_single.min():.4f}, max={ref_cropped_single.max():.4f}, mean={ref_cropped_single.mean():.4f}, std={ref_cropped_single.std():.4f}")
                
                # 检查是否有异常值
                if torch.isnan(ref_cropped_single).any():
                    print(f"[Debug] Warning: Parsed face contains NaN values!")
                if torch.isinf(ref_cropped_single).any():
                    print(f"[Debug] Warning: Parsed face contains Inf values!")
                
                # 处理值域：如果值域是[-1,1]，则转换到[0,1]；如果已经是[0,1]，则直接使用
                if ref_cropped_single.min() < 0:
                    # 从[-1,1]转换到[0,1]
                    ref_cropped_single = (ref_cropped_single + 1) / 2
                ref_cropped_single = torch.clamp(ref_cropped_single, 0, 1)
                
                # 保存处理后的图像
                face_path = os.path.join(batch_dir, "parsed_square_face.png")
                save_image(ref_cropped_single, face_path)
                
                # 额外保存一个未处理的版本用于对比
                raw_face_path = os.path.join(batch_dir, "parsed_square_face_raw.png")
                save_image(ref_cropped[batch_idx], raw_face_path)
            
            # 保存视频路径信息
            info_path = os.path.join(batch_dir, "info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Video Path: {video_path}\n")
                f.write(f"Step: {current_step}\n")
                f.write(f"Batch Index: {batch_idx}\n")
                f.write(f"Face Size: {crop_image_size}x{crop_image_size}\n")
                if first_frame_faces is not None:
                    f.write(f"Dataset Face Size: {first_frame_faces[batch_idx].shape[1]}x{first_frame_faces[batch_idx].shape[2]}\n")
                
                # 添加tensor值域信息
                if ref_image is not None:
                    ref_img_single = ref_image[batch_idx]
                    f.write(f"Original Ref Image Range: [{ref_img_single.min():.4f}, {ref_img_single.max():.4f}]\n")
                
                if first_frame_faces is not None:
                    dataset_face_single = first_frame_faces[batch_idx]
                    f.write(f"Dataset Face Range: [{dataset_face_single.min():.4f}, {dataset_face_single.max():.4f}]\n")
                
                if ref_cropped is not None:
                    ref_cropped_single = ref_cropped[batch_idx]
                    f.write(f"Parsed Face Range: [{ref_cropped_single.min():.4f}, {ref_cropped_single.max():.4f}]\n")
            
            print(f"[Debug] 保存人脸解析器调试图像到: {batch_dir}")
            
    except Exception as e:
        print(f"[Debug] 保存人脸解析器调试图像时出错: {str(e)}")
 