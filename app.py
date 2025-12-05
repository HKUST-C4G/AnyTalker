# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
import sys
import json
import warnings
from datetime import datetime
import gradio as gr
warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

# 导入 AnyTalker 相关的模块
import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import cache_video, str2bool
from wan.utils.infer_utils import calculate_frame_num_from_audio
from utils.get_face_bbox import FaceInference


def str2bool(v):
    """字符串转布尔值工具函数"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    
    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        if any(key in args.task for key in ["i2v", "a2v"]):
            args.sample_steps = 40
        else:
            args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if any(key in args.task for key in ["i2v", "a2v"]) and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # For a2v tasks, frame_num will be determined by audio length if not specified
    if args.frame_num is None:
        args.frame_num = None

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    
    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="a2v-1.3B",
        # choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        # choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1. For a2v tasks, if not specified, frame number will be automatically determined based on audio length."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints/Wan2.1-Fun-1.3B-Inp",
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--post_trained_checkpoint_path",
        type=str,
        default="./checkpoints/AnyTalker/1_3B-single-v1.pth",
        help="The path to the posted-trained checkpoint file.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=True,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--use_half",
        type=str2bool,
        default=True,
        help="Whether to use half precision for model inference, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The directory to save the generated image or video to.")  
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=44,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="The audio to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=4.5,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--cfg_zero",
        action="store_true",
        default=False,
        help="Whether to use adaptive CFG-Zero guidance instead of fixed guidance scale.")
    parser.add_argument(
        "--zero_init_steps",
        type=int,
        default=0,
        help="Number of initial steps to use zero guidance when using cfg_zero.")
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=24,
        help="The frames per second (FPS) of the generated video. Overrides the default value from the config.")
    parser.add_argument(
        "--batch_gen_json",
        type=str,
        default=None,
        help="Path to prompts.json file for batch processing. Images and outputs are in the same directory.")
    parser.add_argument(
        "--batch_output",
        type=str,
        default=None,
        help="Directory to save generated videos when using batch processing. If not specified, defaults to the json filename (without extension) in the same directory.")
    parser.add_argument(
        "--dit_config",
        type=str,
        default="./checkpoints/AnyTalker/config_af2v_1_3B.json",
        help="The path to the dit config file.")
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.15,
        help="Threshold for InsightFace face detection.")
    parser.add_argument(
        "--mode",
        type=str,
        default="pad",
        choices=["pad", "concat"],
        help="The mode to use for audio processing.")
    parser.add_argument(
        "--audio_save_dir",
        type=str,
        default='save_audio/gradio',
        help="The path to save the audio embedding.")
    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def run_graio_demo(args):
    # 设置 Gradio 临时文件目录
    gradio_temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gradio_temp')
    os.makedirs(gradio_temp_dir, exist_ok=True)
    os.environ['GRADIO_TEMP_DIR'] = gradio_temp_dir
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())
        
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    # 加载配置
    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
    
    # 设置 fps
    cfg.fps = args.sample_fps if args.sample_fps is not None else cfg.fps
    
    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    os.makedirs(args.audio_save_dir, exist_ok=True)

    logging.info("Creating AnyTalker pipeline.")
    # 加载模型
    wan_a2v = wan.WanAF2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        use_half=args.use_half,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        t5_cpu=args.t5_cpu,
        post_trained_checkpoint_path=args.post_trained_checkpoint_path,
        dit_config=args.dit_config,
    )
    
    # 创建 InsightFace 人脸检测器
    face_processor = FaceInference(det_thresh=args.det_thresh, ctx_id=local_rank)
    logging.info("Model and face processor loaded successfully.")

    def generate_video(img2vid_image, img2vid_prompt, n_prompt, img2vid_audio_1, img2vid_audio_2, img2vid_audio_3,
                    sd_steps, seed, guide_scale, person_num_selector, audio_mode_selector):
        input_data = {}
        input_data["prompt"] = img2vid_prompt
        input_data["cond_image"] = img2vid_image
        input_data["audio_mode"] = audio_mode_selector  # "pad" or "concat"
        
        # 根据人数收集音频路径
        audio_paths = []
        if person_num_selector == "1 Person":
            if img2vid_audio_1:
                audio_paths.append(img2vid_audio_1)
        elif person_num_selector == "2 Persons":
            if img2vid_audio_1:
                audio_paths.append(img2vid_audio_1)
            if img2vid_audio_2:
                audio_paths.append(img2vid_audio_2)
        elif person_num_selector == "3 Persons":
            if img2vid_audio_1:
                audio_paths.append(img2vid_audio_1)
            if img2vid_audio_2:
                audio_paths.append(img2vid_audio_2)
            if img2vid_audio_3:
                audio_paths.append(img2vid_audio_3)
        
        input_data["audio_paths"] = audio_paths

        logging.info(f"Generating video with {len(audio_paths)} audio(s), mode: {audio_mode_selector}")
        
        # 根据音频长度计算帧数
        current_frame_num = args.frame_num
        if current_frame_num is None:
            if audio_paths and len(audio_paths) > 0:
                # 使用 cfg 中的 fps，如果不可用则使用默认值 24
                fps = getattr(cfg, 'fps', 24)
                current_frame_num = calculate_frame_num_from_audio(audio_paths, fps, mode=audio_mode_selector)
                logging.info(f"Dynamically determined frame number: {current_frame_num} (mode: {audio_mode_selector})")
            else:
                # 没有音频时使用默认帧数
                current_frame_num = 81  # 默认帧数
                logging.info(f"No audio provided, using default frame number: {current_frame_num}")
        else:
            logging.info(f"Using specified frame number: {current_frame_num}")
        
        # 读取图片
        img = Image.open(input_data["cond_image"]).convert("RGB")
        
        # 生成视频
        video = wan_a2v.generate(
            input_data["prompt"],
            img,
            audio=audio_paths[0] if audio_paths and len(audio_paths) > 0 else None,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=current_frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=sd_steps,
            guide_scale=guide_scale,
            seed=seed if seed >= 0 else args.base_seed,
            offload_model=args.offload_model,
            cfg_zero=args.cfg_zero,
            zero_init_steps=args.zero_init_steps,
            face_processor=face_processor,
            img_path=input_data["cond_image"],
            audio_paths=audio_paths,
            task_key="gradio_output",
            mode=audio_mode_selector,
        )
        
        if isinstance(video, dict):
            video = video['original']
        
        # 生成输出文件名（替换特殊字符避免 shell 解析问题）
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_").replace(",", "").replace("*", "x")[:50]
        formatted_size = args.size.replace('*', 'x')
        save_file = f"outputs/{args.task}_{formatted_size}_{formatted_prompt}_{formatted_time}"
        
        # 确保输出目录存在
        os.makedirs("outputs", exist_ok=True)
        
        # 注意：cache_video 不会自动添加后缀，需要传入完整文件名
        output_file = save_file + '.mp4'
        
        logging.info(f"Saving generated video to {output_file}")
        cache_video(
            tensor=video[None],
            save_file=output_file,
            fps=args.sample_fps if args.sample_fps is not None else cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        
        # 如果有音频文件，进行音频合成
        if audio_paths:
            existing_audio_paths = [path for path in audio_paths if path and os.path.exists(path)]
            if existing_audio_paths:
                # 构建输出文件名
                audio_names = [os.path.basename(path).split('.')[0] for path in existing_audio_paths]
                audio_suffix = "_".join([f"audio{i}_{name}" for i, name in enumerate(audio_names)])
                audio_video_path = save_file + f'_{audio_suffix}_cfg_{guide_scale}.mp4'
                
                # 构建 ffmpeg 命令
                if len(existing_audio_paths) == 1:
                    # 只有一个音频
                    ffmpeg_command = f'ffmpeg -i "{output_file}" -i "{existing_audio_paths[0]}" -vcodec libx264 -acodec aac -crf 18 -shortest -y "{audio_video_path}"'
                else:
                    input_args = f'-i "{output_file}"'
                    if audio_mode_selector == "concat":
                        # concat 模式：串联音频
                        for audio_path in existing_audio_paths:
                            input_args += f' -i "{audio_path}"'
                        
                        num_audios = len(existing_audio_paths)
                        concat_inputs = ''.join([f'[{i+1}:a]' for i in range(num_audios)])
                        filter_complex = f'"{concat_inputs}concat=n={num_audios}:v=0:a=1[aout]"'
                        
                        ffmpeg_command = (
                            f'ffmpeg {input_args} -filter_complex {filter_complex} '
                            f'-map 0:v -map "[aout]" -vcodec libx264 -acodec aac -crf 18 -y "{audio_video_path}"'
                        )
                    else:
                        # pad 模式：混合所有音频
                        filter_inputs = []
                        for i, audio_path in enumerate(existing_audio_paths):
                            input_args += f' -i "{audio_path}"'
                            filter_inputs.append(f'[{i+1}:a]')
                        
                        filter_complex = f'{"".join(filter_inputs)}amix=inputs={len(existing_audio_paths)}:duration=shortest[aout]'
                        ffmpeg_command = f'ffmpeg {input_args} -filter_complex "{filter_complex}" -map 0:v -map "[aout]" -vcodec libx264 -acodec aac -crf 18 -y "{audio_video_path}"'
                
                logging.info(f"Adding audio: {ffmpeg_command}")
                os.system(ffmpeg_command)
                
                # 删除没有音频的原始视频文件
                if os.path.exists(audio_video_path):
                    os.remove(output_file)
                    output_file = audio_video_path
                    logging.info(f"Final video saved to: {output_file}")
                else:
                    logging.warning(f"Audio synthesis failed, keeping original video: {output_file}")
            else:
                logging.info(f"No valid audio files found, video saved to: {output_file}")
        else:
            logging.info(f"No audio files provided, video saved to: {output_file}")
        
        logging.info("Finished.")
        return output_file

            
        
    with gr.Blocks() as demo:

        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                       AnyTalker
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        Let your characters interact naturally.
                    </div>
                    <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                        <a href='https://hkust-c4g.github.io/AnyTalker-homepage/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
                        <a href='https://huggingface.co/zzz66/AnyTalker-1.3B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
                        <a href='https://arxiv.org/abs/2511.23475/'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
                    </div>


                    """)

        with gr.Row():
            with gr.Column(scale=1):
                img2vid_image = gr.Image(
                    type="filepath",
                    label="Upload Input Image",
                    elem_id="image_upload",
                )
                img2vid_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate",
                )
                
                
                with gr.Accordion("Audio Options", open=True):
                    person_num_selector = gr.Radio(
                        choices=["1 Person", "2 Persons", "3 Persons"],
                        label="Number of Persons (determined by audio inputs)",
                        value="1 Person"
                    )
                    audio_mode_selector = gr.Radio(
                        choices=["pad", "concat"],
                        label="Audio Processing Mode",
                        value="pad"
                    )
                    gr.Markdown("""
                    **Audio Mode Description:**
                    - **pad**: Select this if every audio input track has already been zero-padded to a common length.
                    - **concat**: Select this if you want the script to chain each speaker's clips together and then zero-pad the non-speaker segments to reach a uniform length.
                    """)
                    gr.Markdown("""
                    **Audio Binding Order:**
                    - Audio inputs are bound to persons based on their positions in the input image, from **left to right**.
                    - Person 1 corresponds to the leftmost person, Person 2 to the middle person (if any), and Person 3 to the rightmost person (if any).
                    """)
                    # 三个音频输入框始终可见，读取时根据 person_num_selector 只读取前 n 个
                    img2vid_audio_1 = gr.Audio(label="Audio for Person 1 (Leftmost)", type="filepath", visible=True)
                    img2vid_audio_2 = gr.Audio(label="Audio for Person 2 (Middle)", type="filepath", visible=True)
                    img2vid_audio_3 = gr.Audio(label="Audio for Person 3 (Rightmost)", type="filepath", visible=True)

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        sd_steps = gr.Slider(
                            label="Diffusion steps",
                            minimum=1,
                            maximum=1000,
                            value=40,
                            step=1)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=41)
                    with gr.Row():
                        guide_scale = gr.Slider(
                            label="Guide Scale",
                            minimum=0,
                            maximum=20,
                            value=4.5,
                            step=0.1)
                    # with gr.Row():
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe the negative prompt you want to add",
                        value="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
                    )

                run_i2v_button = gr.Button("Generate Video")

            with gr.Column(scale=2):
                result_gallery = gr.Video(
                    label='Generated Video', interactive=False, height=600, )
                
                gr.Markdown("""
                ### Example Cases
                
                *Note: Generation time (tested on NVIDIA H200 GPU) may vary depending on GPU specifications and system load.*
                """)
                
                # 隐藏的文本组件用于在 Examples 表格中显示生成耗时
                generation_time_display = gr.Textbox(label="Generation Time (H200 GPU)", visible=True, interactive=False)
                
                # 创建一个函数来处理 examples 选择，同时更新音频输入框的可见性
                def handle_example_select(image, prompt, person_num, audio_mode, audio1, audio2, audio3, gen_time):
                    # 三个音频输入框始终可见，只返回值，不改变可见性
                    # 读取时根据 person_num_selector 只读取前 n 个音频
                    return (
                        image, prompt, person_num, audio_mode,
                        audio1, audio2, audio3, gen_time
                    )
                
                examples_component = gr.Examples(
                    examples = [
                        ["./input_example/images/1p-0.png", "The man stands in the dusty western street, backlit by the setting sun, and his determined gaze speaks of a rugged spirit.", "1 Person", "pad", "./input_example/audios/1p-0.wav", None, None, "~4 minutes"],
                        ["./input_example/images/2p-0.png", "The two people are talking to each other.", "2 Persons", "pad", "./input_example/audios/2p-0-left.wav", "./input_example/audios/2p-0-right.wav", None, "~10 minutes"],
                        ["./input_example/images/2p-1.png", "In a casual, intimate setting, a man and a woman are engaged in a heartfelt conversation inside a car. The man, sporting a denim jacket over a blue shirt, sits attentively with a seatbelt fastened, his gaze fixed on the woman beside him. The woman, wearing a black tank top and a denim jacket draped over her shoulders, smiles warmly, her eyes reflecting genuine interest and connection. The car's interior, with its beige seats and simple design, provides a backdrop that emphasizes their interaction. The scene captures a moment of shared understanding and connection, set against the soft, diffused light of an overcast day. A medium shot from a slightly angled perspective, focusing on their expressions and body language.", "2 Persons", "pad", "./input_example/audios/2p-1-left.wav", "./input_example/audios/2p-1-right.wav", None, "~6 minutes"],
                        ["./input_example/images/2p-2.png", "In a cozy recording studio, a man and a woman are singing together. The man, with tousled brown hair, stands to the left, wearing a light green button-down shirt. His gaze is directed towards the woman, who is smiling warmly. She, with wavy dark hair, is dressed in a black floral dress and stands to the right, her eyes closed in enjoyment. Between them is a professional microphone, capturing their harmonious voices. The background features wooden panels and various audio equipment, creating an intimate and focused atmosphere. The lighting is soft and warm, highlighting their expressions and the intimate setting. A medium shot captures their interaction closely.", "2 Persons", "pad", "./input_example/audios/2p-2-left.wav", "./input_example/audios/2p-2-right.wav", None, "~8 minutes"],
                    ],
                    inputs = [img2vid_image, img2vid_prompt, person_num_selector, audio_mode_selector, img2vid_audio_1, img2vid_audio_2, img2vid_audio_3, generation_time_display],
                    outputs = [img2vid_image, img2vid_prompt, person_num_selector, audio_mode_selector, img2vid_audio_1, img2vid_audio_2, img2vid_audio_3, generation_time_display],
                    fn=handle_example_select,
                )


        run_i2v_button.click(
            fn=generate_video,
            inputs=[img2vid_image, img2vid_prompt, n_prompt, img2vid_audio_1, img2vid_audio_2, img2vid_audio_3, sd_steps, seed, guide_scale, person_num_selector, audio_mode_selector],
            outputs=[result_gallery],
        )
    demo.launch(server_name="0.0.0.0", debug=True, server_port=8418)

        


if __name__ == "__main__":
    args = _parse_args()
    run_graio_demo(args)
    

