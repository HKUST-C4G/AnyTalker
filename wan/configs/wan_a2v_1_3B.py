# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg
import torch
#------------------------ Wan A2V 1.3B ------------------------#

a2v_1_3B = EasyDict(__name__='Config: Wan A2V 1_3B')
a2v_1_3B.update(wan_shared_cfg)

# t5
a2v_1_3B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
a2v_1_3B.t5_tokenizer = 'google/umt5-xxl'

# vae
a2v_1_3B.vae_checkpoint = 'Wan2.1_VAE.pth'
a2v_1_3B.vae_stride = (4, 8, 8)

# clip
a2v_1_3B.clip_model = 'clip_xlm_roberta_vit_h_14'
a2v_1_3B.clip_dtype = torch.float16
a2v_1_3B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
a2v_1_3B.clip_tokenizer = 'xlm-roberta-large'

a2v_1_3B.vocal_separator_path =  "checkpoints/vocal_separator/Kim_Vocal_2.onnx"
a2v_1_3B.wav2vec = "checkpoints/wav2vec2-base-960h"

a2v_1_3B.audio_output_dir = "outputs"
a2v_1_3B.num_frames = 81
a2v_1_3B.fps = 24
#a2v_14B.model_type = "a2v" # 现在不起效

# transformer
a2v_1_3B.patch_size = (1, 2, 2)
a2v_1_3B.dim = 1536
a2v_1_3B.ffn_dim = 8960
a2v_1_3B.freq_dim = 256
a2v_1_3B.num_heads = 12
a2v_1_3B.num_layers = 30
a2v_1_3B.window_size = (-1, -1)
a2v_1_3B.qk_norm = True
a2v_1_3B.cross_attn_norm = True
a2v_1_3B.eps = 1e-6
