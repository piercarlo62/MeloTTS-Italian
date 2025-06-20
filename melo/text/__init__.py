from .symbols import *
import unicodedata


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    import logging
    import unicodedata
    
    logger = logging.getLogger(__name__)
    
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    
    # Normalize both the input text and create a normalized symbol map
    cleaned_text = unicodedata.normalize('NFD', cleaned_text)
    
    # Create a normalized version of the symbol map
    normalized_symbol_map = {}
    for symbol, id_val in symbol_to_id_map.items():
        normalized_symbol = unicodedata.normalize('NFD', symbol)
        normalized_symbol_map[normalized_symbol] = id_val
    
    # Debug logging
    if hasattr(logging, '_logger_initialized'):
        logger.info(f"DEBUG: cleaned_text_to_sequence called with language: {language}")
        logger.info(f"DEBUG: cleaned_text length: {len(cleaned_text)}")
        logger.info(f"DEBUG: cleaned_text (first 50 chars): {repr(cleaned_text[:50])}")
        logger.info(f"DEBUG: Available languages in tone_start_map: {list(language_tone_start_map.keys())}")
        logger.info(f"DEBUG: Available languages in id_map: {list(language_id_map.keys())}")
    
    try:
        phones = [normalized_symbol_map[symbol] for symbol in cleaned_text]
    except KeyError as e:
        if hasattr(logging, '_logger_initialized'):
            logger.error(f"ERROR: Symbol not found: {e}")
            logger.error(f"ERROR: Available symbols count: {len(normalized_symbol_map)}")
            # Add detailed debugging for the missing symbol
            missing_symbol = str(e).strip("'")
            logger.error(f"ERROR: Missing symbol Unicode: U+{ord(missing_symbol):04X}")
            logger.error(f"ERROR: Missing symbol name: {unicodedata.name(missing_symbol, 'UNKNOWN')}")
            
            # Show available diacritic symbols
            diacritic_symbols = [s for s in normalized_symbol_map.keys() if unicodedata.combining(s)]
            logger.error(f"ERROR: Available diacritic symbols: {[repr(s) for s in diacritic_symbols]}")
        raise e
    
    try:
        tone_start = language_tone_start_map[language]
    except KeyError as e:
        if hasattr(logging, '_logger_initialized'):
            logger.error(f"ERROR: Language {language} not found in tone_start_map")
        raise e
        
    try:
        lang_id = language_id_map[language]
    except KeyError as e:
        if hasattr(logging, '_logger_initialized'):
            logger.error(f"ERROR: Language {language} not found in id_map")
        raise e
    
    tones = [i + tone_start for i in tones]
    lang_ids = [lang_id for i in phones]
    
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device):
    import logging
    logger = logging.getLogger(__name__)

    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert
    from .chinese_mix import get_bert_feature as zh_mix_en_bert
    from .spanish_bert import get_bert_feature as sp_bert
    from .french_bert import get_bert_feature as fr_bert
    from .korean import get_bert_feature as kr_bert
    from .italian_bert import get_bert_feature as it_bert  # Add this import

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert, 'ZH_MIX_EN': zh_mix_en_bert, 
                          'FR': fr_bert, 'SP': sp_bert, 'ES': sp_bert, "KR": kr_bert,
                          'IT': it_bert}

    # Use logging instead of print
    if hasattr(logging, '_logger_initialized'):
        logger.info(f"DEBUG: Getting BERT for language: {language}")
        logger.info(f"DEBUG: Available languages: {list(lang_bert_func_map.keys())}")
        logger.info(f"DEBUG: norm_text length: {len(norm_text) if norm_text else 'None'}")
        logger.info(f"DEBUG: word2ph length: {len(word2ph) if word2ph else 'None'}")

    if language not in lang_bert_func_map:
        if hasattr(logging, '_logger_initialized'):
            logger.error(f"ERROR: Language {language} not found in mapping!")
        raise KeyError(f"Language {language} not supported")

    bert_func = lang_bert_func_map[language]
    if hasattr(logging, '_logger_initialized'):
        logger.info(f"DEBUG: Using BERT function: {bert_func}")

    try:
        bert = bert_func(norm_text, word2ph, device)
        if hasattr(logging, '_logger_initialized'):
            logger.info(f"DEBUG: BERT processing successful, shape: {bert.shape if hasattr(bert, 'shape') else 'No shape'}")
        return bert
    except Exception as e:
        if hasattr(logging, '_logger_initialized'):
            logger.error(f"ERROR: BERT processing failed: {e}")
        raise e   
