# WebUI by mrfakename <X @realmrfakename / HF @mrfakename>
# Demo also available on HF Spaces: https://huggingface.co/spaces/mrfakename/MeloTTS
import gradio as gr
import os, torch, io
import numpy as np
import threading
import queue
import time
from typing import Optional, Generator
# os.system('python -m unidic download')
print("Make sure you've downloaded unidic (python -m unidic download) for this WebUI to work.")
from melo.api import TTS
speed = 1.0
import tempfile
import click
device = 'auto'

# Initialize models with Italian support
models = {
    'EN': TTS(language='EN', device=device),
    'ES': TTS(language='ES', device=device),
    'FR': TTS(language='FR', device=device),
    'IT': TTS(language='IT', device=device),  # Added Italian support
    'ZH': TTS(language='ZH', device=device),
    'JP': TTS(language='JP', device=device),
    'KR': TTS(language='KR', device=device),
}
speaker_ids = models['EN'].hps.data.spk2id

default_text_dict = {
    'EN': 'The field of text-to-speech has seen rapid development recently.',
    'ES': 'El campo de la conversión de texto a voz ha experimentado un rápido desarrollo recientemente.',
    'FR': 'Le domaine de la synthèse vocale a connu un développement rapide récemment',
    'IT': 'Il campo della sintesi vocale ha visto un rapido sviluppo di recente.',  # Added Italian
    'ZH': 'text-to-speech 领域近年来发展迅速',
    'JP': 'テキスト読み上げの分野は最近急速な発展を遂げています',
    'KR': '최근 텍스트 음성 변환 분야가 급속도로 발전하고 있습니다.',    
}

# Global variables for streaming
streaming_active = False
audio_queue = queue.Queue()

def synthesize(speaker, text, speed, language, progress=gr.Progress()):
    """Traditional synthesis - generates complete audio file"""
    bio = io.BytesIO()
    models[language].tts_to_file(text, models[language].hps.data.spk2id[speaker], bio, speed=speed, pbar=progress.tqdm, format='wav')
    return bio.getvalue()

def synthesize_streaming(speaker, text, speed, language, sample_rate, bit_depth, channels, chunk_size, progress=gr.Progress()):
    """Streaming synthesis - generates audio chunks progressively"""
    global streaming_active, audio_queue
    
    # Clear previous queue
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break
    
    streaming_active = True
    audio_chunks = []
    total_chunks = 0
    
    try:
        # Convert parameters
        sample_rate = int(sample_rate) if sample_rate > 0 else None
        bit_depth = int(bit_depth)
        channels = int(channels)
        chunk_size = int(chunk_size) if chunk_size > 0 else None
        
        # Stream audio chunks
        for pcm_chunk, metadata in models[language].stream_audio(
            text=text,
            speaker_id=models[language].hps.data.spk2id[speaker],
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            chunk_size=chunk_size,
            speed=speed,
            quiet=True
        ):
            if not streaming_active:
                break
                
            # Convert PCM to numpy array for Gradio
            if bit_depth == 16:
                audio_np = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32767.0
            elif bit_depth == 24:
                # 24-bit requires special handling
                audio_bytes = np.frombuffer(pcm_chunk, dtype=np.uint8)
                audio_24bit = []
                for i in range(0, len(audio_bytes), 3):
                    if i + 2 < len(audio_bytes):
                        sample = int.from_bytes(audio_bytes[i:i+3], byteorder='little', signed=True)
                        audio_24bit.append(sample)
                audio_np = np.array(audio_24bit, dtype=np.float32) / 8388607.0
            elif bit_depth == 32:
                audio_np = np.frombuffer(pcm_chunk, dtype=np.int32).astype(np.float32) / 2147483647.0
            else:  # 8-bit
                audio_np = (np.frombuffer(pcm_chunk, dtype=np.uint8).astype(np.float32) - 128) / 127.0
            
            # Handle stereo
            if channels == 2:
                audio_np = audio_np.reshape(-1, 2)
                audio_np = audio_np.mean(axis=1)  # Convert to mono for Gradio
            
            audio_chunks.append(audio_np)
            total_chunks += 1
            
            # Update progress
            sentence_progress = (metadata['sentence_index'] + 1) / metadata['total_sentences']
            progress(sentence_progress, desc=f"Sentence {metadata['sentence_index'] + 1}/{metadata['total_sentences']}")
        
        # Concatenate all chunks
        if audio_chunks:
            final_audio = np.concatenate(audio_chunks)
            final_sample_rate = sample_rate or models[language].hps.data.sampling_rate
            
            # Convert to format expected by Gradio
            return (final_sample_rate, final_audio)
        else:
            return None
            
    except Exception as e:
        print(f"Streaming error: {e}")
        return None
    finally:
        streaming_active = False

def stop_streaming():
    """Stop current streaming operation"""
    global streaming_active
    streaming_active = False
    return "Streaming stopped"

def load_speakers(language, text):
    """Load speakers for selected language and update default text"""
    if text in list(default_text_dict.values()):
        newtext = default_text_dict[language]
    else:
        newtext = text
    return gr.update(value=list(models[language].hps.data.spk2id.keys())[0], choices=list(models[language].hps.data.spk2id.keys())), newtext

def toggle_streaming_options(streaming_enabled):
    """Show/hide streaming options based on checkbox"""
    return gr.update(visible=streaming_enabled)

# Create Gradio interface
with gr.Blocks(title="MeloTTS WebUI Enhanced") as demo:
    gr.Markdown('# MeloTTS WebUI Enhanced\n\nA WebUI for MeloTTS with streaming capabilities and Italian support.')
    
    with gr.Group():
        with gr.Row():
            speaker = gr.Dropdown(speaker_ids.keys(), interactive=True, value='EN-US', label='Speaker')
            language = gr.Radio(['EN', 'ES', 'FR', 'IT', 'ZH', 'JP', 'KR'], label='Language', value='EN')
        
        speed = gr.Slider(label='Speed', minimum=0.1, maximum=10.0, value=1.0, interactive=True, step=0.1)
        text = gr.Textbox(label="Text to speak", value=default_text_dict['EN'], lines=3)
        
        # Streaming options
        with gr.Row():
            enable_streaming = gr.Checkbox(label="Enable Streaming Mode", value=False)
            stop_btn = gr.Button("Stop Streaming", variant="stop", visible=False)
        
        with gr.Group(visible=False) as streaming_options:
            gr.Markdown("### Streaming Audio Parameters")
            with gr.Row():
                stream_sample_rate = gr.Number(label="Sample Rate (Hz)", value=22050, minimum=8000, maximum=48000)
                stream_bit_depth = gr.Dropdown(label="Bit Depth", choices=[8, 16, 24, 32], value=16)
            with gr.Row():
                stream_channels = gr.Dropdown(label="Channels", choices=[1, 2], value=1)
                stream_chunk_size = gr.Number(label="Chunk Size (samples, 0=sentence-based)", value=0, minimum=0)
        
        # Update language selection
        language.input(load_speakers, inputs=[language, text], outputs=[speaker, text])
        
        # Toggle streaming options visibility
        enable_streaming.change(
            toggle_streaming_options,
            inputs=[enable_streaming],
            outputs=[streaming_options]
        )
        
        enable_streaming.change(
            lambda x: gr.update(visible=x),
            inputs=[enable_streaming],
            outputs=[stop_btn]
        )
    
    # Synthesis buttons
    with gr.Row():
        btn_normal = gr.Button('Synthesize (Normal)', variant='primary')
        btn_streaming = gr.Button('Synthesize (Streaming)', variant='secondary')
    
    # Audio output
    aud = gr.Audio(interactive=False, label="Generated Audio")
    
    # Status output
    status = gr.Textbox(label="Status", interactive=False, visible=False)
    
    # Event handlers
    def conditional_synthesize(speaker, text, speed, language, enable_streaming, sample_rate, bit_depth, channels, chunk_size):
        """Choose synthesis method based on streaming checkbox"""
        if enable_streaming:
            return synthesize_streaming(speaker, text, speed, language, sample_rate, bit_depth, channels, chunk_size)
        else:
            return synthesize(speaker, text, speed, language)
    
    # Normal synthesis
    btn_normal.click(
        synthesize,
        inputs=[speaker, text, speed, language],
        outputs=[aud]
    )
    
    # Streaming synthesis
    btn_streaming.click(
        synthesize_streaming,
        inputs=[speaker, text, speed, language, stream_sample_rate, stream_bit_depth, stream_channels, stream_chunk_size],
        outputs=[aud]
    )
    
    # Stop streaming
    stop_btn.click(
        stop_streaming,
        outputs=[status]
    )
    
    # Examples
    gr.Markdown("### Examples")
    examples = [
        ["EN-US", "Hello, this is a test of the streaming text-to-speech system.", 1.0, "EN"],
        ["ES-MX", "Hola, esta es una prueba del sistema de síntesis de voz en streaming.", 1.0, "ES"],
        ["FR-FR", "Bonjour, ceci est un test du système de synthèse vocale en streaming.", 1.0, "FR"],
        ["IT-IT", "Ciao, questo è un test del sistema di sintesi vocale in streaming.", 1.0, "IT"],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[speaker, text, speed, language],
        outputs=[aud],
        fn=synthesize,
        cache_examples=False
    )
    
    gr.Markdown('---\nWebUI by [mrfakename](https://twitter.com/realmrfakename) | Enhanced with streaming by AI Assistant')

@click.command()
@click.option('--share', '-s', is_flag=True, show_default=True, default=False, help="Expose a publicly-accessible shared Gradio link usable by anyone with the link. Only share the link with people you trust.")
@click.option('--host', '-h', default=None)
@click.option('--port', '-p', type=int, default=None)
def main(share, host, port):
    demo.queue(api_open=False).launch(show_api=False, share=share, server_name=host, server_port=port)

if __name__ == "__main__":
    main()

