import logging
import math
import os
import subprocess
from io import BytesIO

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from audio_separator.separator import Separator
from einops import rearrange
# from funasr.download.download_from_hub import download_model
# from funasr.models.emotion2vec.model import Emotion2vec
from transformers import Wav2Vec2FeatureExtractor

# from memo.models.emotion_classifier import AudioEmotionClassifierModel
from wan.modules.wav2vec import Wav2VecModel


logger = logging.getLogger(__name__)


def resample_audio(input_audio_file: str, output_audio_file: str, sample_rate: int = 16000):
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            input_audio_file,
            "-ar",
            str(sample_rate),
            output_audio_file,
        ]
    )
    ret = p.wait()
    assert ret == 0, f"Resample audio failed! Input: {input_audio_file}, Output: {output_audio_file}"
    return output_audio_file


@torch.no_grad()
def preprocess_audio(
    wav_path: str,
    fps: int,
    wav2vec_model: str,
    vocal_separator_model: str = None,
    cache_dir: str = "",
    device: str = "cuda",
    sample_rate: int = 16000,
    num_generated_frames_per_clip: int = -1,
):
    """
    Preprocess the audio file and extract audio embeddings.

    Args:
        wav_path (str): Path to the input audio file.
        fps (int): Frames per second for the audio processing.
        wav2vec_model (str): Path to the pretrained Wav2Vec model.
        vocal_separator_model (str, optional): Path to the vocal separator model. Defaults to None.
        cache_dir (str, optional): Directory for cached files. Defaults to "".
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to "cuda".
        sample_rate (int, optional): Sampling rate for audio processing. Defaults to 16000.
        num_generated_frames_per_clip (int, optional): Number of generated frames per clip for padding. Defaults to -1.

    Returns:
        tuple: A tuple containing:
            - audio_emb (torch.Tensor): The processed audio embeddings.
            - audio_length (int): The length of the audio in frames.
    """
    # Initialize Wav2Vec model
    audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model).to(device=device)
    audio_encoder.feature_extractor._freeze_parameters()

    # Initialize Wav2Vec feature extractor
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model)

    # Initialize vocal separator if provided
    vocal_separator = None
    if vocal_separator_model is not None:
        os.makedirs(cache_dir, exist_ok=True)
        vocal_separator = Separator(
            output_dir=cache_dir,
            output_single_stem="vocals",
            model_file_dir=os.path.dirname(vocal_separator_model),
        )
        vocal_separator.load_model(os.path.basename(vocal_separator_model))
        assert vocal_separator.model_instance is not None, "Failed to load audio separation model."

    # Perform vocal separation if applicable
    if vocal_separator is not None:
        original_audio_name, _ = os.path.splitext(wav_path)
        target_audio_file = os.path.join(f"{original_audio_name}_(Vocals)_Kim_Vocal_2-16k.wav")
        if not os.path.exists(target_audio_file):
            outputs = vocal_separator.separate(wav_path)
            assert len(outputs) > 0, "Audio separation failed."
            vocal_audio_file = outputs[0]
            vocal_audio_name, _ = os.path.splitext(vocal_audio_file)
            vocal_audio_file = os.path.join(vocal_separator.output_dir, vocal_audio_file)
            vocal_audio_file = resample_audio(
                vocal_audio_file,
                target_audio_file,
                sample_rate,
            )
        else:
            print(f"vocal_audio_file: {target_audio_file} already exists, skip resample")
            vocal_audio_file = target_audio_file
    else:
        vocal_audio_file = wav_path

    # Load audio and extract Wav2Vec features
    speech_array, sampling_rate = librosa.load(vocal_audio_file, sr=sample_rate)
    audio_feature = np.squeeze(wav2vec_feature_extractor(speech_array, sampling_rate=sampling_rate).input_values)
    audio_length = math.ceil(len(audio_feature) / sample_rate * fps)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)

    # Pad audio features to match the required length
    if num_generated_frames_per_clip > 0 and audio_length % num_generated_frames_per_clip != 0:
        audio_feature = torch.nn.functional.pad(
            audio_feature,
            (
                0,
                (num_generated_frames_per_clip - audio_length % num_generated_frames_per_clip) * (sample_rate // fps),
            ),
            "constant",
            0.0,
        )
        audio_length += num_generated_frames_per_clip - audio_length % num_generated_frames_per_clip
    audio_feature = audio_feature.unsqueeze(0)

    # Extract audio embeddings
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=audio_length, output_hidden_states=True)
    assert len(embeddings) > 0, "Failed to extract audio embeddings."
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    # Concatenate embeddings with surrounding frames
    audio_emb = audio_emb.cpu().detach()
    concatenated_tensors = []
    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)] for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))
    audio_emb = torch.stack(concatenated_tensors, dim=0)

    if vocal_separator is not None:
        del vocal_separator
    del audio_encoder

    return audio_emb, audio_length

