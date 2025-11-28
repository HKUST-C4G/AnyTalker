import argparse
import logging
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import json
import wan
import random
from PIL import Image

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, str2bool
from wan.utils.infer_utils import calculate_frame_num_from_audio
from utils.get_face_bbox import FaceInference



EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "A simple and dignified beauty",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "a2v-1.3B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "/data/jsfeng/114/jsfeng/portrait_wan/data/test_data/images/female_asian_01.jpg",
        "audio":
            "",  
    },
    "a2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "/data/jsfeng/114/jsfeng/portrait_wan/data/test_data/images/female_asian_01.jpg",
        "audio":
            "",  
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"
    

    # Check that either (image and prompt) or batch_gen_dir is provided
    # batch_gen_dir related logic is deprecated, no need to validate
    
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

    # The default number of frames are 1 for text-to-image tasks and None for other tasks (will be determined by audio length).
    if args.frame_num is None:
        if "t2i" in args.task:
            args.frame_num = 1
        else:
            # For a2v tasks, frame_num will be determined by audio length if not specified
            args.frame_num = None

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"
    
    # A2V frame_num check - allow None (will be dynamically determined based on audio length)
    if any(key in args.task for key in ["a2v", "i2v"]):
        if args.frame_num is not None:
            # If frame_num is specified, ensure it's in 4n+1 format
            assert (args.frame_num - 1) % 4 == 0, f"frame_num must be in format 4n+1 for task {args.task}, got {args.frame_num}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
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
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--post_trained_checkpoint_path",
        type=str,
        default=None,
        help="The path to the posted-trained checkpoint file.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--use_half",
        type=str2bool,
        default=None,
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
        default=-1,
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
        default=3.0,
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
        default=None,
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
        default=None,
        help="The path to the dit config file.")
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.3,
        help="Threshold for InsightFace face detection.")
    parser.add_argument(
        "--mode",
        type=str,
        default="pad",
        choices=["pad", "concat"],
        help="The mode to use for audio processing.")
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


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")

    cfg = WAN_CONFIGS[args.task]
    # If not changed here, will use default fixed value
    cfg.fps = args.sample_fps if args.sample_fps is not None else cfg.fps
    # For a2v tasks, if frame_num is None, will be dynamically determined during audio processing
    if args.frame_num is not None:
        cfg.num_frames = args.frame_num
    print(f"############ fps is set to {cfg.fps} ############")
    print(f"############ num_frames is set to {cfg.num_frames} ############")

    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")


    if rank == 0:
        print(f"Batch generation mode from json: {args.batch_gen_json}")
    batch_dir = os.path.dirname(args.batch_gen_json)
    prompts_json_path = args.batch_gen_json
    # Read prompts.json file - all processes need to read
    with open(prompts_json_path, 'r') as f:
        prompts_dict = json.load(f)
    print(f"Rank {rank}: Loaded {len(prompts_dict)} prompts from {prompts_json_path}")
    
    # Multi-GPU parallel processing: assign tasks to different GPUs
    prompts_items = list(prompts_dict.items())
    # Single GPU mode
    rank_prompts = prompts_items
    print(f"Single GPU processing: {len(rank_prompts)} items")
    
    # Create A2V model
    logging.info("Creating WanA2V pipeline for batch processing.")
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

    # Create face_processor
    # Use current process's local_rank as GPU device ID
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    face_processor = FaceInference(det_thresh=args.det_thresh, ctx_id=local_rank)
    
    if args.batch_output is not None:
        output_dir = args.batch_output
    else:
        json_base = os.path.splitext(os.path.basename(args.batch_gen_json))[0]
        output_dir = os.path.join(batch_dir, json_base)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Rank {rank}: Using output directory: {output_dir}")
    
    # Check if there are tasks to process
    if len(rank_prompts) == 0:
        print(f"Rank {rank}: No tasks assigned, skipping processing")
        completed = 0
        skipped = 0
    else:
        completed = 0
        skipped = 0
        for task_key, data in rank_prompts:
            # Check JSON data structure
            if isinstance(data, dict):
                # New format: supports audio_list field, can input multiple audio files
                enhance_caption = data.get("enhance_caption", "")
                caption = data.get("caption", "")
                image_path = data.get("image_path", "")
                
                # Check if enhance_caption and caption are consistent
                captions_are_different = enhance_caption != caption and enhance_caption and caption
                
                # Prefer audio_list, if not available use audio_left and audio_right
                audio_list = data.get("audio_list", [])
                if audio_list:
                    # Use audio_list, supports multiple audio files
                    audio_paths = audio_list
                    print(f"Rank {rank}: Using audio_list, total {len(audio_paths)} audio files")
                else:
                    # Compatible with old format: audio_left and audio_right
                    audio_left_path = data.get("audio_left", "")
                    audio_right_path = data.get("audio_right", "")
                    audio_paths = []
                    if audio_left_path:
                        audio_paths.append(audio_left_path)
                    if audio_right_path:
                        audio_paths.append(audio_right_path)
                    print(f"Rank {rank}: Using audio_left/audio_right format, total {len(audio_paths)} audio files")
            else:
                # Old format: directly a prompt string
                enhance_caption = data
                caption = data
                captions_are_different = False
                audio_paths = []
                image_path = ""
            
            # Determine list of prompts to generate
            prompts_to_generate = []
            if captions_are_different:
                # If caption and enhance_caption are inconsistent, generate twice
                prompts_to_generate = [
                    ("caption", caption),
                    ("enhance_caption", enhance_caption)
                ]
                print(f"Rank {rank}: Caption and enhance_caption are inconsistent, will generate twice")
            else:
                # Otherwise generate only once, prefer enhance_caption
                prompt_to_use = enhance_caption if enhance_caption else caption
                prompts_to_generate = [("default", prompt_to_use)]
                print(f"Rank {rank}: Caption and enhance_caption are consistent, generate only once")
            
            # Generate video for each prompt
            for prompt_type, current_prompt in prompts_to_generate:
                # Build output file path, create subfolder if generating multiple times
                if len(prompts_to_generate) > 1:
                    # Create subfolder
                    subfolder = os.path.join(output_dir, prompt_type)
                    os.makedirs(subfolder, exist_ok=True)
                    output_file = os.path.join(subfolder, f"{task_key}.mp4")
                    print(f"Rank {rank}: Creating subfolder: {subfolder}")
                else:
                    output_file = os.path.join(output_dir, f"{task_key}.mp4")
                
                # Check if final output file already exists, skip if exists
                # Consider filename after audio synthesis
                final_output_file = output_file
                if audio_paths:
                    # Build output filename containing all audio filenames
                    audio_names = []
                    for audio_path in audio_paths:
                        if os.path.exists(audio_path):
                            audio_name = os.path.basename(audio_path).split('.')[0]
                            audio_names.append(audio_name)
                    if audio_names:
                        audio_suffix = "_".join([f"audio{i}_{name}" for i, name in enumerate(audio_names)])
                        final_output_file = output_file[:-4] + f'_{audio_suffix}_cfg_{args.sample_guide_scale}.mp4'
                
                # Check if final output file already exists
                if os.path.exists(final_output_file):
                    print(f"Rank {rank}: Final output file {final_output_file} already exists, skipping.")
                    skipped += 1
                    continue
                
                # Also check if intermediate file exists (no audio version)
                if os.path.exists(output_file):
                    print(f"Rank {rank}: Intermediate output file {output_file} already exists, skipping.")
                    skipped += 1
                    continue
            
                print(f"Rank {rank}: Processing {task_key} ({prompt_type}): {current_prompt[:50]}...")
                for i, audio_path in enumerate(audio_paths):
                    if audio_path and os.path.exists(audio_path):
                        print(f"Rank {rank}: Using audio {i}: {audio_path}")
                    else:
                        print(f"Rank {rank}: Warning: audio {i} path is empty or file not found: {audio_path}")
                
                # If frame_num is None, dynamically determine based on audio length
                current_frame_num = args.frame_num
                if current_frame_num is None:
                    if audio_paths and len(audio_paths) > 0:
                        # Use fps from config, if not available use default 24
                        fps = getattr(cfg, 'fps', 24)
                        current_frame_num = calculate_frame_num_from_audio(audio_paths, fps, mode=args.mode)
                        print(f"Rank {rank}: Dynamically determined frame number: {current_frame_num} (mode: {args.mode})")
                    else:
                        # Use default frame number when no audio files
                        for audio_path in audio_paths:
                            if not os.path.exists(audio_path):
                                raise ValueError(f"{task_key} has no audio files, {audio_path} error, cannot determine frame number")
                else:
                    print(f"Rank {rank}: Using specified frame number: {current_frame_num}")   
                # Read image
                img = Image.open(image_path).convert("RGB")
                # Load bbox  
                # Process extended prompt if needed
                input_prompt = current_prompt
                # Generate video - pass audio path list
                video = wan_a2v.generate(
                    input_prompt,
                    img,
                    audio=audio_paths[0] if audio_paths and len(audio_paths) > 0 else None,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=current_frame_num,  # Use calculated frame number
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed if args.base_seed is not None else 42,
                    offload_model=args.offload_model,
                    cfg_zero=args.cfg_zero,
                    zero_init_steps=args.zero_init_steps,
                    face_processor=face_processor,
                    img_path=image_path,
                    audio_paths=audio_paths,  # Pass complete audio path list
                    task_key=task_key,
                    mode=args.mode,  # Pass audio processing mode
                )
            
                # Directly use original video, do not apply mask overlay
                if isinstance(video, dict):
                    video = video['original']

                # Save video - each process saves its own processed files
                if video is not None:
                    print(f"Rank {rank}: Saving generated video to {output_file}")
                    cache_video(
                        tensor=video[None],
                        save_file=output_file,
                        fps=args.sample_fps if args.sample_fps is not None else cfg.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))
                    
                    # If there are audio files, perform audio synthesis
                    if audio_paths:
                        # Filter out existing audio files
                        existing_audio_paths = [path for path in audio_paths if path and os.path.exists(path)]
                        if existing_audio_paths:
                            # Build output filename containing all audio filenames
                            audio_names = [os.path.basename(path).split('.')[0] for path in existing_audio_paths]
                            audio_suffix = "_".join([f"audio{i}_{name}" for i, name in enumerate(audio_names)])
                            audio_video_path = output_file[:-4] + f'_{audio_suffix}_cfg_{args.sample_guide_scale}.mp4'
                            
                            # Build ffmpeg command, supports multiple audio files
                            if len(existing_audio_paths) == 1:
                                # Only one audio
                                ffmpeg_command = f'ffmpeg -i "{output_file}" -i "{existing_audio_paths[0]}" -vcodec libx264 -acodec aac -crf 18 -shortest -y "{audio_video_path}"'
                            else:
                                input_args = f'-i "{output_file}"'
                                if args.mode == "concat":
                                     # Base input: stream 0 is the video file
                                    for audio_path in existing_audio_paths:
                                        input_args += f' -i "{audio_path}"'
                                   
                                    num_audios = len(existing_audio_paths)
                                    # Build concat filter audio inputs, e.g. [1:a][2:a]...[N:a]
                                    concat_inputs = ''.join([f'[{i+1}:a]' for i in range(num_audios)])
                                    # Use concat filter: join audio streams in time, keep only audio (v=0, a=1)
                                    filter_complex = f'"{concat_inputs}concat=n={num_audios}:v=0:a=1[aout]"'

                                    # Map original video stream and the concatenated audio stream to the output file
                                    ffmpeg_command = (
                                        f'ffmpeg {input_args} -filter_complex {filter_complex} '
                                        f'-map 0:v -map "[aout]" -vcodec libx264 -acodec aac -crf 18 -y "{audio_video_path}"'
            )
                                else:
                                    # Multiple audio: mix all audio files
                                    filter_inputs = []
                                    for i, audio_path in enumerate(existing_audio_paths):
                                        input_args += f' -i "{audio_path}"'
                                        filter_inputs.append(f'[{i+1}:a]')
                                    
                                    filter_complex = f'{"".join(filter_inputs)}amix=inputs={len(existing_audio_paths)}:duration=shortest[aout]'
                                    ffmpeg_command = f'ffmpeg {input_args} -filter_complex "{filter_complex}" -map 0:v -map "[aout]" -vcodec libx264 -acodec aac -crf 18 -y "{audio_video_path}"'
                            
                            print(f"Rank {rank}: Adding audio: {ffmpeg_command}")
                            os.system(ffmpeg_command)
                            # Delete original video file without audio
                            os.remove(output_file)
                            print(f"Rank {rank}: Final video saved to: {audio_video_path}")
                        else:
                            print(f"Rank {rank}: No valid audio files found, video saved to: {output_file}")
                    else:
                        print(f"Rank {rank}: No audio files provided, video saved to: {output_file}")
                    
                    completed += 1
    
   
    total_completed = completed
    total_skipped = skipped

    print(f"Rank {rank}: Batch processing completed. Generated {total_completed} videos, skipped {total_skipped} entries.")
    

if __name__ == "__main__":
    args = _parse_args()
    generate(args)
