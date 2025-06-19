import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
from typing import Iterator, Optional, Tuple
import io

from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .download_utils import load_or_download_config, load_or_download_model

class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def stream_audio(self, 
                    text: str, 
                    speaker_id: int,
                    sample_rate: Optional[int] = None,
                    bit_depth: int = 16,
                    channels: int = 1,
                    chunk_size: Optional[int] = None,
                    sdp_ratio: float = 0.2,
                    noise_scale: float = 0.6,
                    noise_scale_w: float = 0.8,
                    speed: float = 1.0,
                    quiet: bool = False) -> Iterator[Tuple[bytes, dict]]:
        """
        Stream TTS audio as raw PCM chunks as soon as each sentence is ready.
        
        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID for voice selection
            sample_rate: Output sample rate (defaults to model's sample rate)
            bit_depth: Audio bit depth (8, 16, 24, 32)
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_size: Size of each audio chunk in samples (None = sentence-based chunks)
            sdp_ratio: Stochastic duration predictor ratio
            noise_scale: Audio variation control
            noise_scale_w: Prosody variation control
            speed: Speech speed multiplier
            quiet: Suppress progress output
            
        Yields:
            Tuple[bytes, dict]: (PCM audio chunk, metadata)
                - bytes: Raw PCM audio data
                - dict: Metadata with sample_rate, bit_depth, channels, chunk_info
        """
        # Set default sample rate from model config
        if sample_rate is None:
            sample_rate = self.hps.data.sampling_rate
            
        # Validate parameters
        if bit_depth not in [8, 16, 24, 32]:
            raise ValueError("bit_depth must be 8, 16, 24, or 32")
        if channels not in [1, 2]:
            raise ValueError("channels must be 1 (mono) or 2 (stereo)")
            
        # Determine numpy dtype based on bit depth
        if bit_depth == 8:
            dtype = np.uint8
            max_val = 127
            offset = 128
        elif bit_depth == 16:
            dtype = np.int16
            max_val = 32767
            offset = 0
        elif bit_depth == 24:
            dtype = np.int32  # Will be converted to 24-bit later
            max_val = 8388607
            offset = 0
        else:  # 32-bit
            dtype = np.int32
            max_val = 2147483647
            offset = 0

        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        
        total_sentences = len(texts)
        
        for sentence_idx, t in enumerate(texts):
            if not quiet:
                print(f"Processing sentence {sentence_idx + 1}/{total_sentences}: {t[:50]}...")
                
            # Text preprocessing
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
                
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
                t, language, self.hps, device, self.symbol_to_id
            )
            
            # Generate audio for this sentence
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                speakers = torch.LongTensor([speaker_id]).to(device)
                
                audio = self.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=1. / speed,
                )[0][0, 0].data.cpu().float().numpy()
                
                # Clean up GPU memory immediately
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                torch.cuda.empty_cache()
            
            # Add pause between sentences (except last one)
            if sentence_idx < total_sentences - 1:
                pause_samples = int((sample_rate * 0.05) / speed)
                audio = np.concatenate([audio, np.zeros(pause_samples, dtype=np.float32)])
            
            # Resample if needed
            if sample_rate != self.hps.data.sampling_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=self.hps.data.sampling_rate, 
                    target_sr=sample_rate
                )
            
            # Convert to stereo if requested
            if channels == 2:
                audio = np.stack([audio, audio], axis=0).T  # Duplicate mono to stereo
            
            # Process audio in chunks if chunk_size is specified
            if chunk_size is not None:
                audio_chunks = self._split_audio_into_chunks(audio, chunk_size)
            else:
                audio_chunks = [audio]  # Single chunk per sentence
            
            # Convert each chunk to PCM and yield
            for chunk_idx, audio_chunk in enumerate(audio_chunks):
                pcm_data = self._convert_to_pcm(audio_chunk, dtype, max_val, offset, bit_depth)
                
                metadata = {
                    'sample_rate': sample_rate,
                    'bit_depth': bit_depth,
                    'channels': channels,
                    'sentence_index': sentence_idx,
                    'total_sentences': total_sentences,
                    'chunk_index': chunk_idx,
                    'total_chunks_in_sentence': len(audio_chunks),
                    'chunk_duration_ms': int((len(audio_chunk) / sample_rate) * 1000),
                    'sentence_text': t
                }
                
                yield pcm_data, metadata

    def _split_audio_into_chunks(self, audio: np.ndarray, chunk_size: int) -> list:
        """Split audio array into chunks of specified size."""
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def _convert_to_pcm(self, audio: np.ndarray, dtype: np.dtype, max_val: int, offset: int, bit_depth: int) -> bytes:
        """Convert float audio to PCM bytes."""
        # Normalize audio to [-1, 1] range
        audio = np.clip(audio, -1.0, 1.0)
        
        if bit_depth == 8:
            # 8-bit is unsigned, range [0, 255]
            pcm_audio = ((audio * max_val) + offset).astype(dtype)
        elif bit_depth == 24:
            # 24-bit requires special handling
            pcm_audio = (audio * max_val).astype(np.int32)
            # Convert to 24-bit bytes (3 bytes per sample)
            pcm_bytes = bytearray()
            for sample in pcm_audio.flatten():
                # Convert to 24-bit little-endian
                sample_bytes = sample.to_bytes(4, byteorder='little', signed=True)[:3]
                pcm_bytes.extend(sample_bytes)
            return bytes(pcm_bytes)
        else:
            # 16-bit and 32-bit
            pcm_audio = (audio * max_val).astype(dtype)
        
        return pcm_audio.tobytes()

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
