import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from text.cleaner import clean_text_bert
import os
import torch
from text.symbols import symbols, num_languages, num_tones

@click.command()
@click.option(
    "--metadata",
    default="data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option(
    "--config_path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
):
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        print(f"Starting preprocessing of {metadata}")
        print(f"Output will be saved to {cleaned_path}")
        
        out_file = open(cleaned_path, "w", encoding="utf-8")
        new_symbols = []
        
        lines = open(metadata, encoding="utf-8").readlines()
        print(f"Total lines to process: {len(lines)}")
        
        for line_num, line in enumerate(tqdm(lines), 1):
            try:
                print(f"\n--- Processing line {line_num} ---")
                print(f"Raw line: {line.strip()}")
                
                # Parse the line
                parts = line.strip().split("|")
                print(f"Split into {len(parts)} parts: {parts}")
                
                if len(parts) != 4:
                    print(f"ERROR: Expected 4 parts, got {len(parts)}")
                    print(f"Parts: {parts}")
                    continue
                
                utt, spk, language, text = parts
                print(f"Parsed - utt: {utt}, spk: {spk}, language: {language}")
                print(f"Text: {text}")
                
                # Check if audio file exists
                if not os.path.exists(utt):
                    print(f"WARNING: Audio file does not exist: {utt}")
                    print(f"Current working directory: {os.getcwd()}")
                    print(f"Absolute path would be: {os.path.abspath(utt)}")
                
                print("Starting text cleaning and BERT processing...")
                
                # This is where the error likely occurs
                norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cpu')
                
                print(f"Text processing successful!")
                print(f"Normalized text: {norm_text}")
                print(f"Phones: {phones[:10]}..." if len(phones) > 10 else f"Phones: {phones}")
                print(f"Tones: {tones[:10]}..." if len(tones) > 10 else f"Tones: {tones}")
                print(f"Word2ph length: {len(word2ph)}")
                print(f"BERT shape: {bert.shape}")
                
                # Check for new symbols
                for ph in phones:
                    if ph not in symbols and ph not in new_symbols:
                        new_symbols.append(ph)
                        print(f'New symbol found: {ph}')
                        print('Updated symbols list:')
                        print(new_symbols)
                        with open(f'{language}_symbol.txt', 'w', encoding='utf-8') as f:
                            f.write(f'{new_symbols}')

                # Validation checks
                print("Performing validation checks...")
                if len(phones) != len(tones):
                    print(f"ERROR: phones length ({len(phones)}) != tones length ({len(tones)})")
                    continue
                    
                if len(phones) != sum(word2ph):
                    print(f"ERROR: phones length ({len(phones)}) != sum of word2ph ({sum(word2ph)})")
                    continue
                
                print("Validation passed, writing to output file...")
                
                # Write to output file
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
                
                # Save BERT features
                bert_path = utt.replace(".wav", ".bert.pt")
                print(f"Saving BERT features to: {bert_path}")
                os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                torch.save(bert.cpu(), bert_path)
                
                print(f"Line {line_num} processed successfully!")
                
            except Exception as error:
                print(f"\n!!! ERROR processing line {line_num} !!!")
                print(f"Line content: {line.strip()}")
                print(f"Error type: {type(error).__name__}")
                print(f"Error message: {str(error)}")
                print(f"Error details: {repr(error)}")
                
                # Print more detailed error info
                import traceback
                print("Full traceback:")
                traceback.print_exc()
                
                # Continue with next line instead of stopping
                continue

        out_file.close()
        print(f"\nPreprocessing completed. Results saved to {cleaned_path}")
        metadata = cleaned_path

    # Rest of the function remains the same...
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(metadata, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = len(spk_id_map)
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
