import pickle
import os
import re

from . import symbols
from .it_phonemizer import cleaner as it_cleaner
from .it_phonemizer import it_to_ipa
from transformers import AutoTokenizer


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def text_normalize(text):
    text = it_cleaner.italian_cleaners(text)
    return text

model_id = 'dbmdz/bert-base-italian-cased'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def g2p(text, pad_start_end=True, tokenized=None):
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    # import pdb; pdb.set_trace()
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    
    phones = []
    tones = []
    word2ph = []
    # print(ph_groups)
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == '[UNK]':
            phone_list = ['UNK']
        else:
            phone_list = list(filter(lambda p: p != " ", it_to_ipa.it2ipa(w)))
        
        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
        # print(phone_list, aaa)
        # print('=' * 10)

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device=None):
    from text import italian_bert
    return italian_bert.get_bert_feature(text, word2ph, device=device)

if __name__ == "__main__":
    ori_text = 'Questo servizio gratuito è"""" 【disponibile》 in italiano 【semplificato] e altri 123'
    # ori_text = "Stavano cercando invano di far capire a mia madre che con i centomila euro che mi aveva lasciato mio padre,"
    # print(ori_text)
    text = text_normalize(ori_text)
    print(text)
    phoneme = it_to_ipa.it2ipa(text)
    print(phoneme)

    
    from TTS.tts.utils.text.phonemizers.multi_phonemizer import MultiPhonemizer
    from text.cleaner_multiling import unicleaners

    def text_normalize(text):
        text = unicleaners(text, cased=True, lang='it')
        return text

    # print(ori_text)
    text = text_normalize(ori_text)
    print(text)
    phonemizer = MultiPhonemizer({"it-it": "espeak"})
    # phonemizer.lang_to_phonemizer['it'].keep_stress = True
    # phonemizer.lang_to_phonemizer['it'].use_espeak_phonemes = True
    phoneme = phonemizer.phonemize(text, separator="", language='it-it')
    print(phoneme)
