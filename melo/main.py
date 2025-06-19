import click
import warnings
import os


@click.command
@click.argument('text')
@click.argument('output_path')
@click.option("--file", '-f', is_flag=True, show_default=True, default=False, help="Text is a file")
@click.option('--language', '-l', default='EN', help='Language, defaults to English', type=click.Choice(['EN', 'ES', 'FR', 'IT', 'ZH', 'JP', 'KR'], case_sensitive=False))
@click.option('--speaker', '-spk', default=None, help='Speaker ID. Leave empty for default speaker. For English, defaults to "EN-Default". For other languages, uses first available speaker.')
@click.option('--list-speakers', '-ls', is_flag=True, default=False, help='List available speakers for the specified language and exit')
@click.option('--speed', '-s', default=1.0, help='Speed, defaults to 1.0', type=float)
@click.option('--device', '-d', default='auto', help='Device, defaults to auto')
def main(text, file, output_path, language, speaker, list_speakers, speed, device):
    language = language.upper()
    if language == '': 
        language = 'EN'
    
    # Initialize TTS model for the specified language
    from melo.api import TTS
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    # List speakers option
    if list_speakers:
        print(f"\nAvailable speakers for {language}:")
        print("-" * 40)
        for speaker_name, speaker_id in speaker_ids.items():
            print(f"  {speaker_name} (ID: {speaker_id})")
        print(f"\nTotal: {len(speaker_ids)} speakers available")
        
        # Show default behavior
        if language == 'EN':
            default_speaker = 'EN-Default' if 'EN-Default' in speaker_ids else list(speaker_ids.keys())[0]
            print(f"Default speaker for {language}: {default_speaker}")
        else:
            print(f"Default speaker for {language}: {list(speaker_ids.keys())[0]}")
        
        print(f"\nUsage example:")
        example_speaker = 'EN-Default' if language == 'EN' and 'EN-Default' in speaker_ids else list(speaker_ids.keys())[0]
        print(f"  python -m melo.main \"Your text here\" output.wav -l {language} -spk {example_speaker}")
        return
    
    # Handle file input
    if file:
        if not os.path.exists(text):
            raise FileNotFoundError(f'Trying to load text from file due to --file/-f flag, but file not found. Remove the --file/-f flag to pass a string.')
        else:
            with open(text, 'r', encoding='utf-8') as f:
                text = f.read().strip()
    
    if text == '':
        raise ValueError('You entered empty text or the file you passed was empty.')
    
    # Speaker selection logic - your suggested enhancement
    if language == 'EN':
        if not speaker: 
            speaker = 'EN-Default'
        spkr = speaker_ids[speaker]
    else:
        if not speaker: 
            spkr = speaker_ids[list(speaker_ids.keys())[0]]
        else:
            spkr = speaker_ids[speaker]
    
    # Validate speaker exists
    if speaker and speaker not in speaker_ids:
        print(f"Error: Speaker '{speaker}' not found for language {language}")
        print(f"Available speakers for {language}:")
        for available_speaker in speaker_ids.keys():
            print(f"  - {available_speaker}")
        print(f"\nUse --list-speakers to see all available speakers")
        raise ValueError(f"Invalid speaker '{speaker}' for language {language}")
    
    # Show what speaker is being used
    if language == 'EN':
        used_speaker = speaker if speaker else 'EN-Default'
    else:
        used_speaker = speaker if speaker else list(speaker_ids.keys())[0]
    
    print(f"Synthesizing text with:")
    print(f"  Language: {language}")
    print(f"  Speaker: {used_speaker} (ID: {spkr})")
    print(f"  Speed: {speed}")
    print(f"  Device: {device}")
    print(f"  Output: {output_path}")
    print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Generate speech
    try:
        model.tts_to_file(text, spkr, output_path, speed=speed)
        print(f"\nSynthesis completed successfully!")
        print(f"Audio saved to: {output_path}")
    except Exception as e:
        print(f"Error during synthesis: {e}")
        raise


if __name__ == "__main__":
    main()

