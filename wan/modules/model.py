# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import random
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention, attention
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from .audio_proj import AudioProjModel
import warnings
warnings.filterwarnings('ignore')

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)
    # Changed float64 to float32 here

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    # Changed float64 to float32 here
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        # Changed float64 to float32 here
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanA2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.k_audio = nn.Linear(dim, dim)
        self.v_audio = nn.Linear(dim, dim)
        self.norm_k_audio = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, temporal_mask=None, face_mask_list=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            temporal_mask(Tensor): Shape [B, L2]
            face_mask_list(list): Shape [n, B, L1]
        """

        context_img = context[1]
        context_audio = context[2]
        context = context[0]

        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        k_audio = self.norm_k_audio(self.k_audio(context_audio)).view(b, -1, n, d)
        v_audio = self.v_audio(context_audio).view(b, -1, n, d)


        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        
        if temporal_mask is not None:
            audio_x = attention(q, k_audio, v_audio, k_lens=None, attn_mask=temporal_mask)
        else:
            audio_x = flash_attention(q, k_audio, v_audio, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        audio_x = audio_x.flatten(2)
        x = x + img_x + audio_x
        x = self.o(x)
        return x

class WanAF2VCrossAttention(WanSelfAttention):
    """ For audio CA output, apply additional Ref attention
        Ref cond input may come from face recognition embedding / clip embedding / 3d vae token
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 use_concat_attention=True):  # New parameter to control whether to use concat mode
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.k_audio = nn.Linear(dim, dim)
        self.v_audio = nn.Linear(dim, dim)
        self.norm_k_audio = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.k_face = nn.Linear(dim, dim)
        self.v_face = nn.Linear(dim, dim)
        self.norm_k_face = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
        # New parameter to control attention mode
        self.use_concat_attention = use_concat_attention

    def forward(
        self, 
        x, 
        context, 
        context_lens, 
        temporal_mask=None,
        face_mask_list=None,
        use_token_mask=True,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            temporal_mask(Tensor): Shape [B, L2]
            
        Usage example:
            # Original mode (separated attention)
            model = WanModel(model_type='a2v_af', use_concat_attention=False)
            
            # New mode (concat attention)
            model = WanModel(model_type='a2v_af', use_concat_attention=True)
            # In new mode, face token is always visible, audio part follows temporal_mask logic
        """
        # [text, image, audio list, audio ref list]
        context_img = context[1]
        context_audios = context[2]
        face_context_list = context[3]
        context = context[0]

        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        """New face kv for audio focus
            n people means n face attn operations
        """
        k_face_list = []
        v_face_list = []
        k_audio_list = []
        v_audio_list = []
        
        # Ensure audio and face lists have consistent length
        min_length = min(len(context_audios), len(face_context_list))
        # print(f"WanAF2VCrossAttention: Processing {min_length} audio-face pairs")
        
        for i in range(min_length):
            context_audio = context_audios[i]
            face_context = face_context_list[i]
            
            # Extract audio features
            k_audio = self.norm_k_audio(self.k_audio(context_audio)).view(b, -1, n, d)
            v_audio = self.v_audio(context_audio).view(b, -1, n, d)
            k_audio_list.append(k_audio)
            v_audio_list.append(v_audio)
    
            # Extract face features
            k_face = self.norm_k_face(self.k_face(face_context)).view(b, -1, n, d)
            v_face = self.v_face(face_context).view(b, -1, n, d)
            k_face_list.append(k_face)
            v_face_list.append(v_face)

        # text attn 
        x = flash_attention(q, k, v, k_lens=context_lens)
        # ref image attn 
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        
        """ For each id, execute identity-aware audio ca
            Method 1: Add residual connection between each person, make it causal
            Method 2: No residual connection
            Method 3: Add residual connection only at the end, preserve original driving information
            Method 4: Don't split into two steps, directly do three-modal CA (video, audio, face)
        """

        af_output_list = []
        # Ensure all lists have consistent length
        min_length = min(len(k_face_list), len(v_face_list), len(k_audio_list), len(v_audio_list), len(face_mask_list))
        # print(f"Processing {min_length} audio-face pairs")
        
        for i in range(min_length):
            k_face = k_face_list[i]
            v_face = v_face_list[i]
            k_audio = k_audio_list[i]
            v_audio = v_audio_list[i]
            face_mask = face_mask_list[i]
            # concat face and audio features
            k_concat = torch.cat([k_face, k_audio], dim=1)  # [B, L_face+L_audio, n, d]
            v_concat = torch.cat([v_face, v_audio], dim=1)  # [B, L_face+L_audio, n, d]
            
            # Construct attention mask
            if temporal_mask is not None:
                # Get face token count
                face_len = k_face.shape[1]
                audio_len = k_audio.shape[1]
                
                # Create new mask: face part all True, audio part follows original mask
                # Fix dimensions: [B, 1, seq_len_q, seq_len_kv]
                new_mask = torch.ones((b, 1, q.shape[1], face_len + audio_len), 
                                    dtype=torch.bool, device=temporal_mask.device)
                
                # face part is always visible
                new_mask[..., :face_len] = True
                
                # audio part follows original mask logic - need to adjust temporal_mask shape
                # temporal_mask shape is [B, 1, seq_len_q, audio_len]
                if temporal_mask.shape[-1] == audio_len:
                    # Ensure dimension match
                    new_mask[..., face_len:] = temporal_mask  # [B, 1, seq_len_q, audio_len]
            
                audio_x = attention(q, k_concat, v_concat, k_lens=None, attn_mask=new_mask)
            else:
                # When no mask, all tokens are visible
                audio_x = flash_attention(q, k_concat, v_concat, k_lens=None)
            
            if use_token_mask:
                # Multiply output by face_mask
                af_output_list.append(audio_x.flatten(2) * face_mask)
            else:
                af_output_list.append(audio_x.flatten(2))
                  
        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        for af_output in af_output_list:
            x = x + af_output
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'a2v_cross_attn': WanA2VCrossAttention,
    'a2v_cross_attn_af': WanAF2VCrossAttention
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 use_concat_attention=False):  # New parameter
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        
        # Create corresponding cross attention based on cross_attn_type
        if cross_attn_type == 'a2v_cross_attn_af':
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                          num_heads,
                                                                          (-1, -1),
                                                                          qk_norm,
                                                                          eps,
                                                                          use_concat_attention)  # Pass new parameter
        else:
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                          num_heads,
                                                                          (-1, -1),
                                                                          qk_norm,
                                                                          eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        temporal_mask=None,  # For audio alignment
        face_mask_list=None, # Multi-person binding
        human_mask_list=None, # Multi-person binding (deprecated, set to None)
        use_token_mask=True,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, temporal_mask=None):
            if isinstance(self.cross_attn, WanAF2VCrossAttention):
                # human_mask_list is now None, no longer used
                x = x + self.cross_attn(
                    self.norm3(x), 
                    context, 
                    context_lens, 
                    temporal_mask, 
                    face_mask_list,
                    use_token_mask=use_token_mask
                )
            elif isinstance(self.cross_attn, WanA2VCrossAttention):
                x = x + self.cross_attn(self.norm3(x), context, context_lens, temporal_mask)
            else:
                x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e, temporal_mask)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 temporal_align=True,
                 use_concat_attention=False):  # New parameter to control concat attention mode
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            temporal_align (`bool`, *optional*, defaults to True):
                Enable temporal alignment for audio features
            use_concat_attention (`bool`, *optional*, defaults to False):
                Use concatenated face and audio features for attention computation
        """

        super().__init__()

        self.checkpoint_enabled = True
        
        assert model_type in ['t2v', 'i2v', 'a2v', 'a2v_af']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.has_temporal_align = temporal_align
        self.use_concat_attention = use_concat_attention  # Save new parameter

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        attn_type = {
            't2v':'t2v_cross_attn',
            'i2v':'i2v_cross_attn',
            'a2v':'a2v_cross_attn',
            'a2v_af':'a2v_cross_attn_af'
        }

        # blocks
        cross_attn_type = attn_type[model_type]
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              self.use_concat_attention)  # Pass new parameter
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        elif model_type=='a2v':
            self.img_emb = MLPProj(1280, dim)
            self.audio_emb = AudioProjModel(seq_len=5,
                                        blocks=12, 
                                        channels=768,
                                        intermediate_dim=512,
                                        output_dim=dim,
                                        context_tokens=32,)
        elif model_type=='a2v_af':
            self.img_emb = MLPProj(1280, dim)
            self.audio_emb = AudioProjModel(seq_len=5,
                                        blocks=12, 
                                        channels=768,
                                        intermediate_dim=512,
                                        output_dim=dim,
                                        context_tokens=32,)
            self.audio_ref_emb = MLPProj(1280, dim) # Used for audio ref attention

        # initialize weights
        self.init_weights()

    def enable_gradient_checkpointing(self,use_reentrant=False):
        self.checkpoint_enabled = True
        self._use_reentrant = use_reentrant

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        audio_feature=None,
        audio_ref_features=None, # For audio ref
        face_mask_list=None, # Multi-person binding
        human_mask_list=None, # Multi-person binding (deprecated, set to None)
        masks_flattened=False,
        use_token_mask=True,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            audio_ref_features (List[Tensor], *optional*):
                Conditional audio features for audio-to-video mode

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None

        c, f, h, w = x[0].shape
        h, w = h//self.patch_size[-2], w//self.patch_size[-1]
        b = len(x)

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # x arrangement: [noisy frames, mask, ref frame + padding frames]

        # embeddings, before: [[36, F, H, W], ...]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # after: [[1, 1536, F, H/2 , W/2], ...]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x] # [[1, seq_len, 1536], ...]
        # Also flatten mask for each id
        if use_token_mask:
            if not masks_flattened and face_mask_list is not None:
                for m_index in range(len(face_mask_list)):
                    # Only take first channel
                    face_mask_list[m_index] = [m[0].flatten(0) for m in face_mask_list[m_index]]
                    face_mask_list[m_index] = torch.stack(face_mask_list[m_index]) # [B, seq_len]
                    # Add a dimension at the end
                    face_mask_list[m_index] = face_mask_list[m_index][..., None]
            
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
            # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # print("="*25,self.model_type)
        if self.model_type=="i2v":
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        elif self.model_type == 'a2v_af':
            # New list mode: supports multiple audio and faces
            if "ref_face_list" in audio_ref_features and "audio_list" in audio_ref_features:
                # Use new list mode
                ref_face_list = audio_ref_features["ref_face_list"]
                audio_list = audio_ref_features["audio_list"]
         
            # Process audio feature list
            audio_embeding_list = []
            for i, audio_feat in enumerate(audio_list):
                audio_embeding = self.audio_emb(audio_feat)
                audio_embeding_list.append(audio_embeding)

            
            # Process face feature list
            ref_context_list = []
            for i, ref_features in enumerate(ref_face_list):
                audio_ref_embeding = self.audio_ref_emb(ref_features)
                ref_context_list.append(audio_ref_embeding)
            
            # Original a2v required features
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim

            # [text, image, audio list, audio ref list]
            context = [context]
            context.append(context_clip)
            context.append(audio_embeding_list)
            context.append(ref_context_list)
        
        # Currently testing does not use temporal_mask
        self.has_temporal_align = True

        if self.has_temporal_align and len(audio_embeding_list) > 0 and audio_embeding_list[0] is not None:
            # Use first audio's shape to build temporal_mask
            audio_shape = audio_embeding_list[0].shape
            temporal_mask = torch.zeros((f, audio_shape[-3]), dtype=torch.bool, device=x.device) 
            temporal_mask[0] = True # First frame image and all speech compute attention
            # print(f"temporal_mask {temporal_mask.shape},{torch.sum(temporal_mask)}")
            for i in range(1, f):
                temporal_mask[i, (i - 1)* 4 + 1: i*4 + 1]=True # In dataloader, audio is already taken with sliding window of 5, no need to do overlap here
            temporal_mask = temporal_mask.reshape(f, 1, 1 , audio_shape[-3], 1).repeat(1, h, w, 1, audio_shape[-2])
            # print(f"temporal_mask {temporal_mask.shape},{h},{w},{torch.sum(temporal_mask)}")
            temporal_mask = rearrange(temporal_mask, 'f h w c d -> (f h w) (c d)').contiguous()[None,None,...]
            temporal_mask = temporal_mask.expand(b, 1, temporal_mask.shape[-2], temporal_mask.shape[-1])
        else: 
            temporal_mask = None


        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            temporal_mask=temporal_mask,  # For audio alignment
            face_mask_list=face_mask_list, # Multi-person binding
            human_mask_list=None,  # human_mask_list no longer used
            use_token_mask=use_token_mask
        )

        def create_custom_forward(module):
            def custom_forward(x, **kwargs):  # Explicitly accept x and **kwargs
                return module(x, **kwargs)
            return custom_forward

        for block in self.blocks:
            if self.training and self.checkpoint_enabled:
                x = checkpoint(
                    create_custom_forward(block),
                    x,  # Positional argument
                    **kwargs,  # Keyword arguments
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config: dict = None, **kwargs):
        import glob
        import os
        from omegaconf import ListConfig
        from typing import Union
        if isinstance(pretrained_model_name_or_path, str) and os.path.isdir(pretrained_model_name_or_path) and (config is None) and not pretrained_model_name_or_path.endswith('.pth'):
            print(">>> Using diffusers from_pretrained with provided config")
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        else:
            # === Custom loading logic ===
            print(">>> Using custom from_pretrained with provided config")
            from diffusers.models.model_loading_utils import load_model_dict_into_meta, load_state_dict
            import accelerate

       
            torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
  
            map_location = kwargs.pop("map_location", 'cpu')

            # step 1. Initialize model
            with accelerate.init_empty_weights():
                model = cls.from_config(config)

            # step 2. Find weight files
            if isinstance(pretrained_model_name_or_path, Union[list, ListConfig]):
                weight_files = pretrained_model_name_or_path
                
            elif os.path.isdir(pretrained_model_name_or_path):
                weight_files = glob.glob(f'{pretrained_model_name_or_path}/*.safetensors')
            else:
                weight_files = [pretrained_model_name_or_path]
                
            
            state_dict = {}
            for wf in weight_files:
             
                _state_dict = load_state_dict(wf, map_location=map_location)
                
                if "model" in _state_dict:
                    state_dict.update(_state_dict["model"])
                else:
                    state_dict.update(_state_dict)
            del _state_dict
            

            empty_state_dict = model.state_dict()
            n_miss = 0
            n_unexpect = 0
      
            for param_name in model.state_dict().keys():
                if param_name not in state_dict:
                    n_miss+=1
             
            for param_name in state_dict.keys():
                if param_name not in model.state_dict():
                    n_unexpect+=1

            # Initialize weights for missing modules
            for name, param in empty_state_dict.items():
                if name not in state_dict:
                
                    if param.dim() > 1:
                    
                        state_dict[name] = nn.init.xavier_uniform_(torch.zeros(param.shape))
                    elif '.norm_' in name:
                        state_dict[name] = nn.init.constant_(torch.zeros(param.shape), 1)
                    else:
                        state_dict[name] = nn.init.zeros_(torch.zeros(param.shape))
                    
            state_dict = {k:v.to(dtype=torch.bfloat16) for k, v in state_dict.items()}

            # step 3. Load weights
         
            load_model_dict_into_meta(model, state_dict, dtype=torch_dtype)
            n_updated = len(empty_state_dict.keys()) - n_miss
           
        
            print(f"{n_updated} parameters are loaded from {pretrained_model_name_or_path}, {n_miss} parameters are miss, {n_unexpect} parameters are unexpected.")

            del state_dict

            return model

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
