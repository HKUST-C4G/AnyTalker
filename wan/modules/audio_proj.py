import torch
from diffusers import ConfigMixin, ModelMixin
from einops import rearrange
from torch import nn
import math


class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        if audio_embeds.dim() == 4:
            audio_embeds = audio_embeds.unsqueeze(0)
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(batch_size, self.context_tokens, self.output_dim)

        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens


class LegacyAudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=8,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        batch_size = audio_embeds.shape[0]

        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        audio_embeds = audio_embeds.reshape(audio_embeds.shape[0], -1)

        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        audio_embeds_vf = audio_embeds_vf.reshape(audio_embeds_vf.shape[0], -1)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=batch_size)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=batch_size)
        audio_embeds = torch.cat([audio_embeds, audio_embeds_vf], dim=1)

        audio_embeds = audio_embeds.reshape(audio_embeds.shape[0] * audio_embeds.shape[1], -1)
        audio_embeds = torch.relu(self.proj2(audio_embeds))
        context_tokens = self.proj3(audio_embeds).reshape(
            audio_embeds.shape[0], self.context_tokens, self.output_dim
        )
        context_tokens = self.norm(context_tokens)
        return rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":

    audio_proj = AudioProjModel(
        seq_len=5,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    )
    audio = torch.randn(1, 41, 5, 12, 768)  # Example input tensor

    output = audio_proj(audio)
    print(output.shape)
