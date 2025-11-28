# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = torch.bfloat16

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = "This shot begins with a scene, then becomes a distorted scene with subtitles, captions, and words on the bottom, featuring an ugly, incomplete or unnatural-looking character or object with extra fingers, poorly drawn hands and face, deformed and disfigured limbs, fused fingers, three legs, deformed arms or legs, distorted body passing through objects, swapped left and right arms, swapped left and right legs, or a deformed face on the back. The mouth movements are unnatural and not synchronized with the speech; lip motion is delayed or mismatched with the voice, showing poor lip-sync accuracy, with visible imperfections or distortions in the teeth. The facial expressions are stiff, frozen, or flickering, with jittering eyes and inconsistent gaze direction. The characters' necks or heads appear twisted. The objects and background in the scene are crashed, broken, or deformed, unnaturally distorted, and changing their front and back orientation."
# wan_shared_cfg.sample_neg_prompt = "Vivid tones, overexposed, static, blurry details, subtitles, style, work, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, motionless frame, cluttered background, three legs, crowded background people, walking backwards, shot switch, transition, shot cut"
# wan_shared_cfg.sample_neg_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, Lens changes"