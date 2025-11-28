import torch
import numpy as np
import cv2
import os
import librosa
import math

def calculate_frame_num_from_audio(audio_paths, fps=24, mode="pad"):
    """
    Calculate corresponding frame number based on audio file length
    
    Args:
        audio_paths (list): List of audio file paths
        fps (int): Video frame rate, default 24fps
        mode (str): Audio processing mode, "pad" or "concat". 
                   In "pad" mode, returns max duration.
                   In "concat" mode, returns sum of all durations.
        
    Returns:
        int: Calculated frame number, returns default 81 if audio file does not exist
    """
    if not audio_paths:
        raise ValueError("No audio files, cannot determine frame number")
    
    if mode == "concat":
        # Concat mode: sum all audio durations
        total_duration = 0
        for audio_path in audio_paths:
            if audio_path and os.path.exists(audio_path):
                try:
                    # Use librosa to get audio duration
                    duration = librosa.get_duration(filename=audio_path)
                    total_duration += duration
                    print(f"audio file {audio_path} duration: {duration:.2f} seconds")
                except Exception as e:
                    raise ValueError(f"Failed to read audio file {audio_path}: {e}")
        
        if total_duration > 0:
            # Calculate frame number, round up
            frame_num = int(math.ceil(total_duration * fps))
            # Ensure frame number is in 4n+1 format (model requirement)
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            print(f"Calculated frame number (concat mode): {frame_num} based on total audio duration {total_duration:.2f}s and frame rate {fps}fps")
            return frame_num
        else:
            raise ValueError("No audio files, cannot determine frame number")
    else:
        # Pad mode: use max duration (original behavior)
        max_duration = 0
        for audio_path in audio_paths:
            if audio_path and os.path.exists(audio_path):
                try:
                    # Use librosa to get audio duration
                    duration = librosa.get_duration(filename=audio_path)
                    max_duration = max(max_duration, duration)
                    print(f"audio file {audio_path} duration: {duration:.2f} seconds")
                except Exception as e:
                    raise ValueError(f"Failed to read audio file {audio_path}: {e}")
        
        if max_duration > 0:
            # Calculate frame number, round up
            frame_num = int(math.ceil(max_duration * fps))
            # Ensure frame number is in 4n+1 format (model requirement)
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            print(f"Calculated frame number (pad mode): {frame_num} based on max audio duration {max_duration:.2f}s and frame rate {fps}fps")
            return frame_num
        else:
            raise ValueError("No audio files, cannot determine frame number")

# 计算模型的参数量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_millions = total_params / 1e6  # Convert to millions
    return total_params_in_millions

# 构建空条件的audio_ref_features - 适配多人情况
def create_null_audio_ref_features(audio_ref_features):
    null_features = {}
    # 处理ref_face_list - 多人情况
    if 'ref_face_list' in audio_ref_features and audio_ref_features['ref_face_list']:
        null_ref_face_list = []
        for ref_face in audio_ref_features['ref_face_list']:
            if ref_face is not None:
                null_ref_face_list.append(ref_face.clone().detach())
            else:
                null_ref_face_list.append(None)
        null_features['ref_face_list'] = null_ref_face_list
    else:
        null_features['ref_face_list'] = []
    
    # 处理audio_list - 多人情况
    if 'audio_list' in audio_ref_features and audio_ref_features['audio_list']:
        null_audio_list = []
        for audio in audio_ref_features['audio_list']:
            if audio is not None:
                null_audio_list.append(torch.zeros_like(audio))
            else:
                null_audio_list.append(None)
        null_features['audio_list'] = null_audio_list
    else:
        null_features['audio_list'] = []
    
    return null_features

def process_audio_features(
    audio_paths=None,
    audio=None,
    mode="pad",
    F=None,
    frame_num=None,
    task_key=None,
    fps=None,
    wav2vec_model=None,
    vocal_separator_model=None,
    audio_output_dir=None,
    device=None,
    use_half=False,
    half_dtype=None,
    preprocess_audio=None,
    resample_audio=None,
):
    """
    Process audio files and extract audio features.
    
    Args:
        audio_paths (list): List of audio file paths (new format, supports multiple audio files)
        audio (str): Single audio file path (legacy format)
        mode (str): Audio processing mode, "pad" or "concat"
        F (int): Target frame number (already calculated outside)
        frame_num (int): Frame number for cache file naming (legacy)
        task_key (str): Task key for cache file naming
        fps (int): Frames per second
        wav2vec_model (str): Path to wav2vec model
        vocal_separator_model (str): Path to vocal separator model
        audio_output_dir (str): Directory for audio output
        device: Device to use for processing
        use_half (bool): Whether to use half precision
        half_dtype: Half precision dtype (torch.float16 or torch.float32)
        preprocess_audio: Function to preprocess audio
        resample_audio: Function to resample audio
        
    Returns:
        list: List of audio feature tensors
    """
    from .audio_utils import preprocess_audio as _preprocess_audio, resample_audio as _resample_audio
    
    # Use provided functions or import from audio_utils
    if preprocess_audio is None:
        preprocess_audio = _preprocess_audio
    if resample_audio is None:
        resample_audio = _resample_audio
    
    audio_feat_list = []
    
    if audio_paths and len(audio_paths) > 0:
        print(f"Processing {len(audio_paths)} audio files in {mode} mode: {audio_paths}")
        cache_dir = os.path.join(audio_output_dir, "audio_preprocess")
        os.makedirs(cache_dir, exist_ok=True)

        if mode == "concat":
            # Concat mode: record each audio's length, calculate total length, 
            # and pad each audio with zeros in non-speaker segments
            audio_lengths = []  # Store actual length of each audio in frames
            raw_audio_feat_list = []  # Store raw audio features before padding
            
            # First pass: process all audios and record their actual lengths
            for i, audio_path in enumerate(audio_paths):
                if audio_path and os.path.exists(audio_path):
                    print(f"Processing audio {i} (first pass): {audio_path}")
                    target_resampled_audio_path = os.path.join(cache_dir, f"{os.path.basename(audio_path).split('.')[0]}-{task_key}_16k_concat.wav")
                    if not os.path.exists(target_resampled_audio_path):
                        resample_audio(
                            audio_path,
                            target_resampled_audio_path,
                        )
                    with torch.no_grad():
                        # Process audio without padding to get actual length
                        audio_emb, audio_length = preprocess_audio(
                            wav_path=target_resampled_audio_path,
                            num_generated_frames_per_clip=-1,  # -1 means no padding
                            fps=fps,
                            wav2vec_model=wav2vec_model,
                            vocal_separator_model=vocal_separator_model,
                            cache_dir=cache_dir,
                            device=device,
                        )
                        # If half precision is enabled, use float16; otherwise use bfloat16
                        audio_dtype = half_dtype if use_half else torch.bfloat16
                        audio_emb = audio_emb.to(device, dtype=audio_dtype)
                    
                    # Get actual frame length (audio_length is in frames)
                    actual_frame_length = audio_emb.shape[0]
                    audio_lengths.append(actual_frame_length)
                    raw_audio_feat_list.append(audio_emb)
                    print(f"Audio {i} actual length: {actual_frame_length} frames, shape: {audio_emb.shape}")
                else:
                    print(f"Warning: Audio {i} path is empty or file not found: {audio_path}")
                    audio_lengths.append(0)
                    raw_audio_feat_list.append(None)
            
            # Calculate total length from actual processed frames
            total_length = sum(audio_lengths)
            print(f"Total audio length in concat mode (from processed frames): {total_length} frames")
            
            # Ensure total length is in 4n+1 format (model requirement)
            total_length = ((total_length - 1) // 4) * 4 + 1
            print(f"Adjusted total length to 4n+1 format: {total_length} frames")
            
            # Note: F was already calculated outside and passed as parameter
            # We should not update F here because it has been used to create other tensors (noise, mask, etc.)
            # If there's a mismatch, it means the calculation outside was inaccurate, but we'll use F as is
            if total_length > F:
                print(f"Warning: Actual processed frames ({total_length}) > pre-calculated F ({F}). Using F={F} to maintain consistency with other tensors.")
            elif total_length < F:
                print(f"Info: Actual processed frames ({total_length}) < pre-calculated F ({F}). Using F={F}.")
            else:
                print(f"Info: Actual processed frames ({total_length}) matches pre-calculated F={F}.")
            
            # Second pass: create padded audio features for each audio
            # Each audio is placed in its corresponding time segment, with zeros elsewhere
            cumulative_length = 0
            reference_feat_shape = None
            
            # First, find a reference feature shape from valid audio
            for raw_audio_feat in raw_audio_feat_list:
                if raw_audio_feat is not None:
                    reference_feat_shape = raw_audio_feat.shape[1:]  # Get shape without frame dimension
                    break
            
            if reference_feat_shape is None:
                raise ValueError("No valid audio files found in concat mode")
            
            for i, (raw_audio_feat, audio_len) in enumerate(zip(raw_audio_feat_list, audio_lengths)):
                if raw_audio_feat is not None and audio_len > 0:
                    # Create zero tensor with total length and same feature shape
                    padded_audio_feat = torch.zeros(
                        (F,) + reference_feat_shape,
                        dtype=raw_audio_feat.dtype,
                        device=raw_audio_feat.device
                    )
                    
                    # Place audio data in its corresponding time segment
                    end_pos = min(cumulative_length + audio_len, F)
                    actual_audio_len = end_pos - cumulative_length
                    padded_audio_feat[cumulative_length:end_pos] = raw_audio_feat[:actual_audio_len]
                    
                    audio_feat_list.append(padded_audio_feat)
                    print(f"Audio {i} padded: placed at frames [{cumulative_length}:{end_pos}], shape: {padded_audio_feat.shape}")
                    cumulative_length += audio_len
                else:
                    # Create zero features for missing audio with total length
                    zero_audio_feat = torch.zeros(
                        (F,) + reference_feat_shape,
                        dtype=torch.bfloat16 if not use_half else half_dtype,
                        device=device
                    )
                    audio_feat_list.append(zero_audio_feat)
                    print(f"Audio {i} is missing, created zero features with shape: {zero_audio_feat.shape}")
        else:
            # Pad mode: keep existing logic, no changes needed
            for i, audio_path in enumerate(audio_paths):
                if audio_path and os.path.exists(audio_path):
                    print(f"Processing audio {i}: {audio_path}")
                    target_resampled_audio_path = os.path.join(cache_dir, f"{os.path.basename(audio_path).split('.')[0]}-{task_key}_16k_{F}.wav")
                    if not os.path.exists(target_resampled_audio_path):
                        resample_audio(
                            audio_path,
                            target_resampled_audio_path,
                        )
                    with torch.no_grad():
                        print(f"wav2vec_model: {wav2vec_model}")
                        print(f"cache_dir:{cache_dir}")
                        # Use dynamically determined frame number F
                        audio_emb, audio_length = preprocess_audio(
                            wav_path=target_resampled_audio_path,
                            num_generated_frames_per_clip=F,  # Use dynamically determined frame number
                            fps=fps,
                            wav2vec_model=wav2vec_model,
                            vocal_separator_model=vocal_separator_model,
                            cache_dir=cache_dir,
                            device=device,
                        )
                        # If half precision is enabled, use float16; otherwise use bfloat16
                        audio_dtype = half_dtype if use_half else torch.bfloat16
                        audio_emb = audio_emb.to(device, dtype=audio_dtype)
                    
                    audio_feat = audio_emb[:F]  # Use dynamically determined frame number
                    audio_feat_list.append(audio_feat)
                    print(f"Audio {i} processed, shape: {audio_feat.shape}")
                else:
                    print(f"Warning: Audio {i} path is empty or file not found: {audio_path}")
                    # Create zero features for missing audio
                    if len(audio_feat_list) > 0:
                        # Use first audio's shape to create zero features
                        zero_audio_feat = torch.zeros_like(audio_feat_list[0])
                        audio_feat_list.append(zero_audio_feat)
                    else:
                        print(f"Error: No valid audio files found, cannot create zero features")
    else:
        # Compatible with old format: use single audio parameter
        if audio is not None:
            print(f"Processing single audio (legacy format): {audio}")
            cache_dir = os.path.join(audio_output_dir, "audio_preprocess")
            os.makedirs(cache_dir, exist_ok=True)

            target_resampled_audio_path = os.path.join(cache_dir, f"{os.path.basename(audio).split('.')[0]}-16k.wav")
            if not os.path.exists(target_resampled_audio_path):
                audio = resample_audio(
                    audio,
                    target_resampled_audio_path,
                )
            with torch.no_grad():
                # Use dynamically determined frame number F
                audio_emb, audio_length = preprocess_audio(
                    wav_path=audio,
                    num_generated_frames_per_clip=F,  # Use dynamically determined frame number
                    fps=fps,
                    wav2vec_model=wav2vec_model,
                    vocal_separator_model=vocal_separator_model,
                    cache_dir=cache_dir,
                    device=device,
                )
                # If half precision is enabled, use float16; otherwise use bfloat16
                audio_dtype = half_dtype if use_half else torch.bfloat16
                audio_emb = audio_emb.to(device, dtype=audio_dtype)
            
            audio_feat = audio_emb[:F]  # Use dynamically determined frame number
            audio_feat_list.append(audio_feat)
            print(f"Single audio processed, shape: {audio_feat.shape}")
        else:
            print("No audio files provided")
    
    return audio_feat_list

@torch.cuda.amp.autocast(dtype=torch.float32)
def optimized_scale(positive_flat, negative_flat):
    # Calculate dot production
    positive_norm = torch.norm(positive_flat, dim=-1, keepdim=True)
    negative_norm = torch.norm(negative_flat, dim=-1, keepdim=True)
    
    # Calculate cosine similarity
    cosine_sim = torch.sum(positive_flat * negative_flat, dim=-1, keepdim=True) / (positive_norm * negative_norm + 1e-8)
    
    # Calculate scale factor
    scale = (positive_norm / (negative_norm + 1e-8)) * cosine_sim
    
    return scale


def expand_face_mask_flexible(face_mask, width_scale_factor, height_scale_factor):
    """
    将face_mask中值为1的区域按指定的宽度和高度倍数独立扩大
    
    Args:
        face_mask: tensor, shape: [H, W]，原始的face mask
        width_scale_factor: float, 宽度扩大倍数
        height_scale_factor: float, 高度扩大倍数
        
    Returns:
        tensor: shape: [H, W]，扩大后的face mask
    """
    if width_scale_factor == 1.0 and height_scale_factor == 1.0:
        return face_mask
        
    # 找到mask中非零区域的边界框
    mask_indices = torch.nonzero(face_mask > 0.5)
    if mask_indices.numel() == 0:
        return face_mask
        
    # 计算当前mask的边界框
    min_h, min_w = mask_indices.min(dim=0)[0]
    max_h, max_w = mask_indices.max(dim=0)[0]
    
    # 计算中心点
    center_h = (min_h + max_h) / 2.0
    center_w = (min_w + max_w) / 2.0
    
    # 计算当前bbox的尺寸
    current_h = max_h - min_h + 1
    current_w = max_w - min_w + 1
    
    # 计算扩大后的尺寸，宽度和高度独立缩放
    new_h = int(current_h * height_scale_factor)
    new_w = int(current_w * width_scale_factor)
    
    # 计算新的边界框（居中扩大）
    new_min_h = int(center_h - new_h / 2.0)
    new_max_h = int(center_h + new_h / 2.0)
    new_min_w = int(center_w - new_w / 2.0)
    new_max_w = int(center_w + new_w / 2.0)
    
    # 确保新边界框不超出原图像范围
    H, W = face_mask.shape
    new_min_h = max(0, new_min_h)
    new_max_h = min(H - 1, new_max_h)
    new_min_w = max(0, new_min_w)
    new_max_w = min(W - 1, new_max_w)
    
    # 创建新的mask
    expanded_mask = torch.zeros_like(face_mask)
    
    # 将原始mask区域调整到新的边界框
    if new_max_h > new_min_h and new_max_w > new_min_w:
        # 提取原始mask的内容
        original_content = face_mask[min_h:max_h+1, min_w:max_w+1]
        
        # 将原始内容缩放到新的尺寸
        target_h = new_max_h - new_min_h + 1
        target_w = new_max_w - new_min_w + 1
        
        if target_h > 0 and target_w > 0:
            scaled_content = torch.nn.functional.interpolate(
                original_content.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            # 将缩放后的内容放置到新位置
            expanded_mask[new_min_h:new_max_h+1, new_min_w:new_max_w+1] = scaled_content
    
    return expanded_mask


def gen_inference_masks(masks, img_shape, num_frames=None):
    """
    为推理生成与训练时相同格式的mask
    注意：推理时的mask是按整个图片标记的，不需要切割50%的逻辑
    为了适配训练格式，需要添加batch维度和帧维度 [H, W] -> [1, F, H, W]
    
    Args:
        masks: list of tensors, 人脸检测模型生成的mask列表，每个mask都是[H, W]格式
        img_shape: tuple, 图像形状 (H, W)
        num_frames: int, 视频帧数
        
    Returns:
        dict: 包含face_mask_list的字典，human_mask_list设为None
    """
    H, W = img_shape
    F = num_frames if num_frames is not None else 1
    num_faces = len(masks)
    
    print(f"gen_inference_masks: 处理{num_faces}个人脸，图像尺寸{H}x{W}，帧数{F}")
    
    with torch.no_grad():
        face_mask_list = []
        
        # 为每个人脸生成多帧mask
        for i, mask in enumerate(masks):
            # 创建多帧mask：所有帧都使用face_mask
            face_mask_multi = mask.unsqueeze(0).unsqueeze(0).repeat(1, 1, F, 1, 1)  # [B, C, F, H, W]
            face_mask_list.append(face_mask_multi)
            print(f"人脸{i}: mask shape {mask.shape} -> 多帧mask shape {face_mask_multi.shape}")
        
        # 构建concat mask - 将所有mask在宽度方向拼接
        if num_faces > 1:
            face_mask_concat = torch.cat(face_mask_list, dim=4)  # [B, C, F, H, num_faces*W]
            print(f"拼接后的concat mask shape: {face_mask_concat.shape}")
        else:
            face_mask_concat = face_mask_list[0]
            print(f"单人模式，concat mask shape: {face_mask_concat.shape}")
        
        return {
            "face_mask_list": face_mask_list,
            "human_mask_list": None,  # 不再使用human mask
            "face_mask_concat": face_mask_concat,
            "num_faces": num_faces
        }


def expand_bbox_and_crop_image(img, bbox, width_scale_factor, height_scale_factor):
    """
    将bbox按scale_factor放大并从图像中安全切割对应区域
    
    Args:
        img: tensor, shape: [C, H, W], 输入图像 (值域为-1到1)
        bbox: list or tuple, [x1, y1, x2, y2], bbox坐标
        width_scale_factor: float, 宽度放大倍数
        height_scale_factor: float, 高度放大倍数
        
    Returns:
        tuple: (cropped_image, new_bbox)
        - cropped_image: tensor, shape: [C, new_h, new_w], 切割后的图像
        - new_bbox: list, [new_x1, new_y1, new_x2, new_y2], 调整后的bbox坐标
    """
    # 获取原始bbox坐标
    x1, y1, x2, y2 = bbox
    
    # 获取图像尺寸
    _, img_h, img_w = img.shape
    
    # 计算bbox的中心点和原始尺寸
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    original_w = x2 - x1
    original_h = y2 - y1
    
    # 计算放大后的尺寸
    new_w = original_w * width_scale_factor
    new_h = original_h * height_scale_factor
    
    # 计算放大后的bbox坐标（以中心点为准）
    new_x1 = center_x - new_w / 2.0
    new_y1 = center_y - new_h / 2.0
    new_x2 = center_x + new_w / 2.0
    new_y2 = center_y + new_h / 2.0
    
    # 确保bbox不超出图像边界，同时保持最小尺寸
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)
    
    # 确保切割后的尺寸至少为1像素
    if new_x2 <= new_x1:
        # 如果宽度为0或负数，调整为最小可用宽度
        if center_x < img_w / 2:
            new_x1 = max(0, int(center_x) - 1)
            new_x2 = min(img_w, new_x1 + max(1, int(original_w)))
        else:
            new_x2 = min(img_w, int(center_x) + 1)
            new_x1 = max(0, new_x2 - max(1, int(original_w)))
    
    if new_y2 <= new_y1:
        # 如果高度为0或负数，调整为最小可用高度
        if center_y < img_h / 2:
            new_y1 = max(0, int(center_y) - 1)
            new_y2 = min(img_h, new_y1 + max(1, int(original_h)))
        else:
            new_y2 = min(img_h, int(center_y) + 1)
            new_y1 = max(0, new_y2 - max(1, int(original_h)))
    
    # 转换为整数坐标
    new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
    
    # 最终检查，确保坐标有效
    assert new_x2 > new_x1 and new_y2 > new_y1, f"Invalid bbox after adjustment: [{new_x1}, {new_y1}, {new_x2}, {new_y2}]"
    
    # 从原图切割放大后的区域
    cropped_image = img[:, new_y1:new_y2, new_x1:new_x2]
    
    return cropped_image, [new_x1, new_y1, new_x2, new_y2]


def gen_smooth_transition_mask_for_dit(face_mask, lat_h, lat_w, F, device, mask_dtype, target_translate=(0, 0), target_scale=1.0):
    """
    Generate smooth transition mask based on face_mask and latent shape for DIT mask
    First frame is all white (all 1s), subsequent frames gradually transition from original position to target position and scale
    
    Args:
        face_mask: tensor, shape: [H, W]
        lat_h: int, latent height
        lat_w: int, latent width
        F: int, number of frames in original video
        device: torch.device, device to create tensors on
        mask_dtype: torch.dtype, dtype for mask tensors
        target_translate: tuple, (x, y) target translation amount
        target_scale: float, target scale ratio
        
    Returns:
        tensor: shape: [4, F, lat_h, lat_w], mask for DIT
    """
    
    # Resize face_mask to latent size
    face_mask_resized = torch.nn.functional.interpolate(
        face_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        size=(lat_h, lat_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)  # [lat_h, lat_w]
    
    # Create mask, first frame all white (all 1s), remaining frames gradually transition
    msk = torch.zeros(1, F, lat_h, lat_w, device=device, dtype=mask_dtype)
    msk[:, 0:1] = 1.0  # First frame all white
    
    if F > 1:
        # Generate different transformation parameters for each frame to achieve smooth transition
        for frame_idx in range(1, F):
            # Calculate transition progress for current frame (0 to 1)
            progress = (frame_idx - 1) / (F - 2) if F > 2 else 1.0
            
            # Use linear transition for more uniform changes
            # progress is already linear, use directly
            
            # Translation and scale for current frame (only horizontal translation allowed)
            current_translate = (
                0,  # Vertical direction always 0, no vertical movement allowed
                int(target_translate[1] * progress)  # Only use horizontal translation
            )
            current_scale = 1.0 + (target_scale - 1.0) * progress
            
            # Generate mask for current frame
            if current_scale != 1.0:
                # Calculate scaled size
                scaled_h = int(lat_h * current_scale)
                scaled_w = int(lat_w * current_scale)
                
                # Scale mask
                scaled_mask = torch.nn.functional.interpolate(
                    face_mask_resized.unsqueeze(0).unsqueeze(0),  # [1, 1, lat_h, lat_w]
                    size=(scaled_h, scaled_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)  # [scaled_h, scaled_w]
                
                # Create zero mask of target size
                transformed_mask = torch.zeros(lat_h, lat_w, device=device, dtype=mask_dtype)
                
                # Calculate placement position (centered)
                start_h = max(0, (lat_h - scaled_h) // 2)
                start_w = max(0, (lat_w - scaled_w) // 2)
                end_h = min(lat_h, start_h + scaled_h)
                end_w = min(lat_w, start_w + scaled_w)
                
                # Calculate crop range in scaled_mask
                src_start_h = max(0, (scaled_h - lat_h) // 2)
                src_start_w = max(0, (scaled_w - lat_w) // 2)
                src_end_h = src_start_h + (end_h - start_h)
                src_end_w = src_start_w + (end_w - start_w)
                
                # Place scaled mask to target position
                transformed_mask[start_h:end_h, start_w:end_w] = scaled_mask[src_start_h:src_end_h, src_start_w:src_end_w]
            else:
                transformed_mask = face_mask_resized.clone().to(dtype=mask_dtype)
            
            # Apply horizontal translation, stop when touching boundary
            translate_w = current_translate[1]  # Only take horizontal translation
            if translate_w != 0:
                # Find horizontal boundaries of mask
                mask_indices = torch.nonzero(transformed_mask > 0.5)
                if mask_indices.numel() > 0:
                    mask_min_w = mask_indices[:, 1].min().item()
                    mask_max_w = mask_indices[:, 1].max().item()
                    
                    # Calculate actual available horizontal translation amount
                    if translate_w < 0:
                        # When moving left, check left boundary
                        max_translate_w = -mask_min_w
                        actual_translate_w = max(translate_w, max_translate_w)
                    else:
                        # When moving right, check right boundary
                        max_translate_w = lat_w - 1 - mask_max_w
                        actual_translate_w = min(translate_w, max_translate_w)
                    
                    # If there is valid translation amount, execute translation
                    if actual_translate_w != 0:
                        # Use torch.roll for horizontal translation, but ensure not exceeding boundary
                        if abs(actual_translate_w) <= min(mask_min_w, lat_w - 1 - mask_max_w):
                            # Only use roll within safe range
                            transformed_mask = torch.roll(transformed_mask, shifts=actual_translate_w, dims=1)
                        else:
                            # Manually copy to avoid wrapping
                            new_mask = torch.zeros_like(transformed_mask, dtype=mask_dtype)
                            if actual_translate_w > 0:
                                # Move right
                                new_mask[:, actual_translate_w:] = transformed_mask[:, :-actual_translate_w]
                            else:
                                # Move left
                                new_mask[:, :actual_translate_w] = transformed_mask[:, -actual_translate_w:]
                            transformed_mask = new_mask
            
            # Assign mask for current frame
            msk[:, frame_idx:frame_idx+1] = transformed_mask.unsqueeze(0).unsqueeze(0)
    
    # Reference encode_image_vae processing method, convert mask to format required by DIT
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ], dim=1)
    
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]  # shape: [4, F, lat_h, lat_w]
    return msk