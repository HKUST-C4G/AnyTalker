import copy
import gc
import logging
import math
import os
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.cuda.amp as amp
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .modules.clip import CLIPModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .modules.model import WanModel
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.audio_utils import preprocess_audio, resample_audio
from .utils.infer_utils import create_null_audio_ref_features, \
expand_face_mask_flexible, gen_inference_masks, expand_bbox_and_crop_image, count_parameters, \
gen_smooth_transition_mask_for_dit, process_audio_features, process_audio_features


class WanAF2V:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        t5_cpu=False,
        init_on_cpu=True,
        use_gradient_checkpointing=False,
        post_trained_checkpoint_path=None,
        dit_config=None,
        crop_image_size=224,
        use_half=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            post_trained_checkpoint_path (`str`, *optional*, defaults to None):
                Path to the post-trained checkpoint file. If provided, model will be loaded from this checkpoint.
            use_half (`bool`, *optional*, defaults to False):
                Whether to use half precision (float16/bfloat16) for model inference. Reduces memory usage.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.use_half = use_half

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        # If half precision is enabled, force use float16 for faster speed
        if use_half:
            self.half_dtype = torch.float16
            logging.info(f"Half precision enabled, using dtype: {self.half_dtype} (forced float16 for faster inference)")
        else:
            self.half_dtype = torch.float32

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=None,
        )

        # Output model parameter count
        model_param_count = count_parameters(self.text_encoder.model)
        logging.info(f"Text Model parameters: {model_param_count}M")

        
        self.crop_image_size = crop_image_size
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        
        # Output model parameter count
        model_param_count = count_parameters(self.vae.model)
        logging.info(f"VAE Model parameters: {model_param_count}M")

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))
        
        # Output model parameter count
        model_param_count = count_parameters(self.clip.model)
        logging.info(f"CLIP Model parameters: {model_param_count}M")

        logging.info(f"Creating WanModel from {checkpoint_dir}")

      
        if post_trained_checkpoint_path:
            try:
                if rank == 0:
                    print(f"Loading post-trained model from {post_trained_checkpoint_path}")
                # Load model config from original directory
                # TODO: config can also be specified in current directory
                # config_dict = json.load(open(os.path.join(checkpoint_dir, 'config.json')))
                config_dict = dit_config
                
                self.model = WanModel.from_config(config_dict)
                # All cards directly load safetensors
                #model_state = load_file(post_trained_checkpoint_path)
                checkpoint = torch.load(post_trained_checkpoint_path, map_location='cpu', weights_only=True)
                model_state = checkpoint['model']
                self.model.load_state_dict(model_state)
                if rank == 0:
                    print(f"safertensors have been loaded: {post_trained_checkpoint_path}")
            except Exception as e:
                if rank == 0:
                    print(f"Error loading post-trained model: {e}")
                raise e
        else:   
            self.model = WanModel.from_pretrained(checkpoint_dir)

        self.model.eval().requires_grad_(False)

        # Output model parameter count
        model_param_count = count_parameters(self.model)
        logging.info(f"DiT Model parameters: {model_param_count}M")
        

        # Enable gradient checkpointing (only effective during training)
        if use_gradient_checkpointing:
            self.model.enable_gradient_checkpointing()  # Assume WanModel supports this method
            logging.info("Gradient checkpointing enabled for WanModel")
       
        self.sp_size = 1


        if not init_on_cpu:
            self.model.to(self.device)
            # If half precision is enabled, convert model to half precision
            if use_half:
                try:
                    self.model = self.model.to(dtype=self.half_dtype)
                    logging.info(f"Model converted to {self.half_dtype} precision")
                except Exception as e:
                    logging.warning(f"Failed to convert model to half precision: {e}. Continuing with float32.")
                    self.use_half = False
                    self.half_dtype = torch.float32

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(
        self,
        input_prompt,
        img,
        audio=None,  # New audio input
        max_area=720 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        cfg_zero=False,
        zero_init_steps=0,
        face_processor=None, # InsightFace model
        img_path=None, # For InsightFace use
        audio_paths=None, # New: audio path list, supports multiple audio files
        task_key=None,
        mode="pad",  # Audio processing mode: "pad" or "concat"
    ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            cfg_zero (`bool`, *optional*, defaults to False):
                Whether to use adaptive CFG-Zero guidance instead of fixed guidance scale
            zero_init_steps (`int`, *optional*, defaults to 0):
                Number of initial steps to use zero guidance when using cfg_zero

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        # If half precision is enabled, convert input image to half precision
        if self.use_half:
            img = img.to(dtype=self.half_dtype)

        # Save audio_paths parameter for later use
        self.audio_paths = audio_paths

        # Use passed frame_num directly (it's already calculated outside in concat mode)
        F = frame_num
        print(f"Using frame number: {F} (mode: {mode})")
        h, w = img.shape[1:]
        print(f"Input image size: {h}, {w}, {max_area}")
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        
        # Save original aspect ratio
        original_aspect_ratio = h / w
        
        # Find nearest size combination divisible by 32
        # First try adjusting based on width
        w_candidate1 = (w // 32) * 32
        if w_candidate1 == 0:
            w_candidate1 = 32
        h_candidate1 = (int(w_candidate1 * original_aspect_ratio) // 32) * 32
        if h_candidate1 == 0:
            h_candidate1 = 32
        
        # Then try adjusting based on height
        h_candidate2 = (h // 32) * 32
        if h_candidate2 == 0:
            h_candidate2 = 32
        w_candidate2 = (int(h_candidate2 / original_aspect_ratio) // 32) * 32
        if w_candidate2 == 0:
            w_candidate2 = 32
        
        # Select combination with smallest difference from original size
        diff1 = abs(w_candidate1 - w) + abs(h_candidate1 - h)
        diff2 = abs(w_candidate2 - w) + abs(h_candidate2 - h)
        
        if diff1 <= diff2:
            w, h = w_candidate1, h_candidate1
        else:
            w, h = w_candidate2, h_candidate2
        
        # Recalculate lat_h and lat_w to maintain consistency
        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]
        
        print(f"Processed image size: {h}, {w}, {lat_h}, {lat_w}")
        
        # Create noise after adjusting lat_h and lat_w
        latent_frame_num = (F - 1) // self.vae_stride[0] + 1
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        # Select dtype based on whether half precision is enabled
        noise_dtype = self.half_dtype if self.use_half else torch.float32
        noise = torch.randn(16, latent_frame_num, lat_h, lat_w, dtype=noise_dtype, generator=seed_g, device=self.device)
        print(f"noise shape: {noise.shape}, dtype: {noise.dtype}")
        
        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        print(f"Max seq_len: {max_seq_len}")

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        with torch.no_grad():
            # Ensure input image uses correct precision
            img_input = img[:, None, :, :]
            if self.use_half:
                img_input = img_input.to(dtype=self.half_dtype)
            clip_context = self.clip.visual([img_input])
            # If half precision is enabled, convert clip_context to half precision
            if self.use_half:
                clip_context = clip_context.to(dtype=self.half_dtype)
       

        """
        Start of i2v mask and ref latent construction logic
        """
        # Select mask dtype based on whether half precision is enabled
        mask_dtype = self.half_dtype if self.use_half else torch.float32
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device, dtype=mask_dtype) # Use F corresponding to original frame number
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                        dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        # Latent containing first frame ref
        with torch.no_grad():
            # VAE encoding requires float32, but we can convert result to half precision
            img_for_vae = img[None].cpu()
            if img_for_vae.dtype != torch.float32:
                img_for_vae = img_for_vae.float()
            y = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img_for_vae, size=(h, w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, F-1, h, w, dtype=torch.float32)
                ],
                            dim=1).to(self.device)
            ])[0]
            # If half precision is enabled, convert VAE encoding result to half precision
            if self.use_half:
                y = y.to(dtype=self.half_dtype)
            print(f"y shape after VAE encode: {y.shape}, dtype: {y.dtype}")

        # Initialize face_clip_context variable
        face_clip_context = None
        if face_processor is not None:
            w_scale_factor = 1.2
            h_scale_factor = 1.1
            
            # Determine number of faces needed based on number of audio files
            if hasattr(self, 'audio_paths') and self.audio_paths:
                # Use new audio_paths parameter
                n_faces_needed = len(self.audio_paths)
                print(f"number of faces needed: {n_faces_needed}")
            else:
                # Only one audio or no audio
                n_faces_needed = 1
                print(f"only one audio or no audio, number of faces needed: {n_faces_needed}")
            
            # Use modified infer method, pass n parameter
            face_info = face_processor.infer(img_path, n=n_faces_needed)
            # Get actual number of faces detected
            n = len(face_info['masks'])  # Get sample count
            print(f"number of faces detected: {n}")
            # Strictly arrange left-to-right, no random sorting
            # Assume face detection results are already arranged left-to-right
            print(f"the face order is set to left-to-right: {list(range(n))}")
            masks = face_info['masks']   
            
            print(f"video with face processor, scale up w={w_scale_factor}, h={h_scale_factor} ###")
            # First expand mask, then perform smooth transition processing
            expanded_masks = []
            for mask in masks:
                expanded_mask = expand_face_mask_flexible(
                    torch.from_numpy(mask).to(self.device, dtype=img.dtype).clone(), 
                    width_scale_factor=w_scale_factor, 
                    height_scale_factor=h_scale_factor
                )
                expanded_masks.append(expanded_mask)
            global_mask = torch.zeros_like(expanded_masks[0])
            for mask in expanded_masks:
                global_mask += mask

            # Select mask dtype based on whether half precision is enabled
            mask_dtype = self.half_dtype if self.use_half else torch.float32
            dit_mask = gen_smooth_transition_mask_for_dit(
                global_mask, 
                lat_h, 
                lat_w, 
                F, 
                device=self.device,
                mask_dtype=mask_dtype,
                target_translate=(0, 0),  # Random left or right
                target_scale=1  # No longer scaling in gen_smooth_transition_mask_for_dit
            )
            y = torch.cat([dit_mask, y], dim=0)
            
            # Resize expanded_masks to VAE encoded latent size
            resized_masks = []
            with torch.no_grad():
                for mask in expanded_masks:
                    # Resize mask from original size directly to VAE encoded latent size
                    latent_mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                        size=(lat_h // 2, lat_w // 2),  # Directly resize to latent size
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)  
                    
                    resized_masks.append(latent_mask)
            
            # Calculate number of frames after VAE encoding
            latent_frame_num = (F - 1) // self.vae_stride[0] + 1
            
            inference_masks = gen_inference_masks(
                resized_masks, 
                (lat_h // 2, lat_w // 2),  # Use VAE encoded latent size
                num_frames=latent_frame_num  # Use number of frames after VAE encoding
            )
            bboxes = face_info['bboxes']
            
            # Loop through all faces, extract CLIP features
            face_clip_context_list = []
            for i, bbox in enumerate(bboxes):
                try:
                    # Directly process bbox, assume format is [x, y, w, h]
                    bbox_x, bbox_y, bbox_w, bbox_h = bbox
                    bbox_converted = [bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h]  # Convert to [x1, y1, x2, y2]
                    
                    # Use encapsulated function for bbox expansion and image cropping, but cropping uses original bbox
                    cropped_face, adjusted_bbox = expand_bbox_and_crop_image(
                        img, 
                        bbox_converted,  
                        width_scale_factor=1, 
                        height_scale_factor=1
                    )
               
                    cropped_face = cropped_face.to(self.device)
                    # If half precision is enabled, convert cropped face to half precision
                    if self.use_half:
                        cropped_face = cropped_face.to(dtype=self.half_dtype)
                    
                    with torch.no_grad():
                        # CLIP encoding method
                        face_clip_context = self.clip.visual([cropped_face[:, None, :, :]])
                        # If half precision is enabled, convert face_clip_context to half precision
                        if self.use_half:
                            face_clip_context = face_clip_context.to(dtype=self.half_dtype)
         
                    # Add to list
                    face_clip_context_list.append(face_clip_context)

                    
                except Exception as e:
                    print(f"error on face {i+1}: {e}")
                    continue
            
            print(f"face feature extraction loop completed, successfully processed {len(face_clip_context_list)} faces")
            
            # For backward compatibility, keep original variable name
            face_clip_context = face_clip_context_list[0] if face_clip_context_list else None

        else:
            y = torch.concat([msk, y])

        # Process multiple audio files using utility function
        audio_feat_list = process_audio_features(
            audio_paths=audio_paths,
            audio=audio,
            mode=mode,
            F=F,
            frame_num=frame_num,
            task_key=task_key,
            fps=self.config.fps,
            wav2vec_model=self.config.wav2vec,
            vocal_separator_model=self.config.vocal_separator_path,
            audio_output_dir=self.config.audio_output_dir,
            device=self.device,
            use_half=self.use_half,
            half_dtype=self.half_dtype,
            preprocess_audio=preprocess_audio,
            resample_audio=resample_audio,
        )

        # Prepare audio_ref_features - new list mode
        audio_ref_features = {
            "ref_face": face_clip_context, # face clip features (backward compatible)
        }
        
        # Build face feature list and audio feature list
        ref_face_list = face_clip_context_list.copy() if face_clip_context_list else []
        audio_list = audio_feat_list.copy() if audio_feat_list else []
             
        # Add lists to audio_ref_features
        audio_ref_features["ref_face_list"] = ref_face_list
        audio_ref_features["audio_list"] = audio_list
        
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        if offload_model:
            self.clip.model.cpu()
            torch.cuda.empty_cache()
        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=5,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            # Ensure latent uses correct precision
            if self.use_half and latent.dtype != self.half_dtype:
                latent = latent.to(dtype=self.half_dtype)

            # If half precision is enabled, convert context to half precision
            if self.use_half:
                context = [c.to(dtype=self.half_dtype) for c in context]
                context_null = [c.to(dtype=self.half_dtype) for c in context_null]

            # for cfg - use complete audio_ref_features
            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                'audio_ref_features': audio_ref_features 
            }

            # Create different types of null condition parameters
            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                'audio_ref_features': create_null_audio_ref_features(audio_ref_features)
            }
            
            # Add inference-generated masks to model parameters
            if face_processor is not None and 'inference_masks' in locals():
                # Use deep copy to ensure each call uses independent mask data
                arg_c['face_mask_list'] = copy.deepcopy(inference_masks['face_mask_list'])   
                arg_null['face_mask_list'] = copy.deepcopy(inference_masks['face_mask_list'])    

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            masks_flattened = False # Set to False for first time
           
            for i, t in enumerate(tqdm(timesteps)):
                # Ensure latent is on correct device and precision
                latent = latent.to(self.device)
                if self.use_half and latent.dtype != self.half_dtype:
                    latent = latent.to(dtype=self.half_dtype)
                latent_model_input = [latent]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model_output_cond = self.model(
                    latent_model_input, 
                    t=timestep, 
                    masks_flattened=masks_flattened, 
                    **arg_c
                )
                noise_pred_cond = model_output_cond[0]
                # Ensure noise_pred_cond uses correct precision
                if self.use_half and noise_pred_cond.dtype != self.half_dtype:
                    noise_pred_cond = noise_pred_cond.to(dtype=self.half_dtype)
                noise_pred_cond = noise_pred_cond.to(
                        torch.device('cpu') if offload_model else self.device)
   
                if offload_model:
                    torch.cuda.empty_cache()

                if not cfg_zero:
                    model_output_uncond = self.model(
                        latent_model_input, 
                        t=timestep, 
                        masks_flattened=masks_flattened, 
                        **arg_null
                    )
                    noise_pred_uncond = model_output_uncond[0]
                    # Ensure noise_pred_uncond uses correct precision
                    if self.use_half and noise_pred_uncond.dtype != self.half_dtype:
                        noise_pred_uncond = noise_pred_uncond.to(dtype=self.half_dtype)
                    noise_pred_uncond = noise_pred_uncond.to(
                            torch.device('cpu') if offload_model else self.device)
                else:
                    noise_pred_uncond = None

                masks_flattened = True  # No need to execute later
                
                if offload_model:
                    torch.cuda.empty_cache()
                    
                if cfg_zero:
                    noise_pred = noise_pred_cond
                else:
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                # Ensure noise_pred uses correct precision
                if self.use_half and noise_pred.dtype != self.half_dtype:
                    noise_pred = noise_pred.to(dtype=self.half_dtype)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device
                )

                # scheduler.step may require float32, so temporarily convert
                noise_pred_for_scheduler = noise_pred.float() if self.use_half else noise_pred
                latent_for_scheduler = latent.float() if self.use_half else latent
                
                temp_x0 = sample_scheduler.step(
                    noise_pred_for_scheduler.unsqueeze(0),
                    t,
                    latent_for_scheduler.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                
                # Convert back to half precision (if enabled)
                if self.use_half:
                    temp_x0 = temp_x0.to(dtype=self.half_dtype)
                latent = temp_x0.squeeze(0)
                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            # VAE decoding requires float32, so temporarily convert
            x0_for_decode = [x.float() if self.use_half else x for x in x0]
            videos = self.vae.decode(x0_for_decode)
            result_videos = videos[0]

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        return result_videos if self.rank == 0 else None
