﻿import importlib
from typing import List

import gruut
from gruut_ipa import IPA # pip install gruut_ipa

from base import BasePhonemizer
from punctuation import Punctuation

# Table for str.translate to fix gruut/TTS phoneme mismatch
GRUUT_TRANS_TABLE = str.maketrans("g", "ɡ")


class Gruut(BasePhonemizer):
    """Gruut wrapper for G2P

    Args:
        language (str):
            Valid language code for the used backend.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to `Punctuation.default_puncs()`.

        keep_puncs (bool):
            If true, keep the punctuations after phonemization. Defaults to True.

        use_espeak_phonemes (bool):
            If true, use espeak lexicons instead of default Gruut lexicons. Defaults to False.

        keep_stress (bool):
            If true, keep the stress characters after phonemization. Defaults to False.

    Example:

        >>> from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut
        >>> phonemizer = Gruut('it-it')
        >>> phonemizer.phonemize("Ciao mondo!", separator="|")
        'tʃ|a|o| m|o|n|d|o|!'
    """

    def __init__(
        self,
        language: str,
        punctuations=Punctuation.default_puncs(),
        keep_puncs=True,
        use_espeak_phonemes=False,
        keep_stress=False,
    ):
        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        self.use_espeak_phonemes = use_espeak_phonemes
        self.keep_stress = keep_stress

    @staticmethod
    def name():
        return "gruut"

    def phonemize_gruut(self, text: str, separator: str = "|", tie=False) -> str:  # pylint: disable=unused-argument
        """Convert input text to phonemes.

        Gruut phonemizes the given `str` by seperating each phoneme character with `separator`, even for characters
        that constitude a single sound.

        It doesn't affect 🐸TTS since it individually converts each character to token IDs.

        Examples::
            "ciao come stai oggi?" -> `tʃ|a|o| k|o|m|e| s|t|a|i| o|dʒ|i`

        Args:
            text (str):
                Text to be converted to phonemes.

            tie (bool, optional) : When True use a '͡' character between
                consecutive characters of a single phoneme. Else separate phoneme
                with '_'. This option requires espeak>=1.49. Default to False.
        """
        ph_list = []
        for sentence in gruut.sentences(text, lang=self.language, espeak=self.use_espeak_phonemes):
            for word in sentence:
                if word.is_break:
                    # Use actual character for break phoneme (e.g., comma)
                    if ph_list:
                        # Join with previous word
                        ph_list[-1].append(word.text)
                    else:
                        # First word is punctuation
                        ph_list.append([word.text])
                elif word.phonemes:
                    # Add phonemes for word
                    word_phonemes = []

                    for word_phoneme in word.phonemes:
                        if not self.keep_stress:
                            # Remove primary/secondary stress
                            word_phoneme = IPA.without_stress(word_phoneme)

                        word_phoneme = word_phoneme.translate(GRUUT_TRANS_TABLE)

                        if word_phoneme:
                            # Flatten phonemes
                            word_phonemes.extend(word_phoneme)

                    if word_phonemes:
                        ph_list.append(word_phonemes)

        ph_words = [separator.join(word_phonemes) for word_phonemes in ph_list]
        ph = f"{separator} ".join(ph_words)
        return ph

    def _phonemize(self, text, separator):
        return self.phonemize_gruut(text, separator, tie=False)

    def is_supported_language(self, language):
        """Returns True if `language` is supported by the backend"""
        return gruut.is_language_supported(language)

    @staticmethod
    def supported_languages() -> List:
        """Get a dictionary of supported languages.

        Returns:
            List: List of language codes.
        """
        return list(gruut.get_supported_languages())

    def version(self):
        """Get the version of the used backend.

        Returns:
            str: Version of the used backend.
        """
        return gruut.__version__

    @classmethod
    def is_available(cls):
        """Return true if ESpeak is available else false"""
        return importlib.util.find_spec("gruut") is not None


if __name__ == "__main__":
    from cleaner import italian_cleaners
    import json

    e = Gruut(language="it-it", keep_puncs=True, keep_stress=True, use_espeak_phonemes=True)
    symbols = [
        "_",
        ",",
        ".",
        "!",
        "?",
        "-",
        "~",
        "\u2026",
        "N",
        "Q",
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "\u0251",
        "\u00e6",
        "\u0283",
        "\u0291",
        "\u00e7",
        "\u026f",
        "\u026a",
        "\u0254",
        "\u025b",
        "\u0279",
        "\u00f0",
        "\u0259",
        "\u026b",
        "\u0265",
        "\u0278",
        "\u028a",
        "\u027e",
        "\u0292",
        "\u03b8",
        "\u03b2",
        "\u014b",
        "\u0266",
        "\u207c",
        "\u02b0",
        "`",
        "^",
        "#",
        "*",
        "=",
        "\u02c8",
        "\u02cc",
        "\u2192",
        "\u2193",
        "\u2191",
        " ",
        "\u0261",
        "r",
        "\u0272",
        "\u029d",
        "\u028e",
        "\u02d0",
        "t\u0283",
        "d\u0292",
        "\u0283",
        "\u0292",
        "\u025b",
        "\u0254",
        "\u032a"
    ]
    
    with open('../../data/example/metadata.list', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
     
    used_sym = []
    not_existed_sym = []
    phonemes = []

    for line in lines:
        text = line.split('|')[-1].strip()
        text = italian_cleaners(text)
        ipa =  e.phonemize(text, separator="")
        phonemes.append(ipa)
        for s in ipa:
            if s not in symbols:
                if s not in not_existed_sym:
                    print(f'not_existed char: {s}')
                    not_existed_sym.append(s)
            else:
                if s not in used_sym:
                    # print(f'used char: {s}')
                    used_sym.append(s)
    
    print(used_sym)
    print(not_existed_sym)


    with open('italian_symbols.txt', 'w',  encoding='utf-8') as g:
        g.writelines(symbols + not_existed_sym)
        
    with open('example_ipa.txt', 'w',  encoding='utf-8') as g:
        g.writelines(phonemes)

    data = {'symbols': symbols + not_existed_sym}

    with open('it_symbols.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
