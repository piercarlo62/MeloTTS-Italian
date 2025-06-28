import re
import hashlib
import time
import os
import logging
import json
import pandas as pd
from datetime import datetime
import random
from jellyfish import jaro_winkler_similarity  # You'll need: pip install jellyfish


class ItalianCallCenterSentenceGenerator:
    def __init__(self, model_path="/content/models/Ministral-8B-Instruct-2410/4.0bpw", output_dir="."):
        # Setup logging (file only, no console spam)
        self.setup_logging(output_dir)
        self.logger.info("🚀 Initializing Italian Call Center Sentence Generator with ExLlamaV2")
        
        # Model configuration
        self.model_path = model_path
        
        try:
            print(f"Loading quantized model from: {self.model_path}")
            print("🔧 Using ExLlamaV2 with 4.0bpw quantization for memory efficiency...")
            self.logger.info(f"Loading model from: {self.model_path}")
            
            # Initialize ExLlamaV2 configuration
            from exllamav2 import ExLlamaV2Config, ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Tokenizer
            from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
            
            self.config = ExLlamaV2Config(self.model_path)
            
            # Initialize model
            self.model = ExLlamaV2(self.config)
            # Initialize cache
            self.cache = ExLlamaV2Cache(self.model, lazy=True)
            self.model.load_autosplit(self.cache)
            # Initialize tokenizer
            self.tokenizer = ExLlamaV2Tokenizer(self.config)
            # Initialize generator
            self.generator = ExLlamaV2DynamicGenerator(
                model=self.model,
                cache=self.cache,
                tokenizer=self.tokenizer
            )
            # Create sampler settings
            self.settings = ExLlamaV2Sampler.Settings()                       
            self.settings.temperature = 0.8  # Increased for more variety
            self.settings.top_p = 0.85
            self.settings.top_k = 40
            self.settings.token_repetition_penalty = 1.15  # Increased to reduce repetition
            
            print(f"✅ Successfully loaded quantized Ministral-8B (4.0bpw)")
            print(f"🔧 Memory optimized with ExLlamaV2")
            self.logger.info(f"✅ Successfully loaded quantized model")
            
        except Exception as e:
            print(f"❌ Failed to load model from {self.model_path}: {e}")
            self.logger.error(f"❌ Failed to load model: {e}")
            raise Exception(f"Could not load model: {e}")
        
        # Generated sentences tracking
        self.generated_sentences = []
        self.sentence_hashes = set()
        
        # Statistics tracking
        self.stats = {
            "total_attempts": 0,
            "valid_sentences": 0,
            "invalid_sentences": 0,
            "duplicate_sentences": 0,
            "similarity_rejected": 0
        }
        
        # Setup output file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(output_dir, f"italian_sentences_{timestamp}.txt")
        
        # Improved domain prompts with specific instructions
        self.domain_prompts = {
            "general": [
                "Genera frasi complete e naturali per operatori di call center che salutano clienti italiani. Ogni frase deve essere unica, senza placeholder o parentesi quadre. Usa nomi propri specifici quando necessario (Mario, Giulia, etc.)",
                "Crea risposte professionali per servizio clienti che ringraziano i clienti. Ogni frase deve essere completa, specifica e senza segnaposto come [nome] o [servizio]",
                "Scrivi frasi per operatori che chiedono informazioni ai clienti. Usa dettagli specifici e concreti, evita placeholder generici",
                "Genera conferme professionali per richieste clienti. Ogni frase deve essere completa con dettagli specifici, non generici",
                "Crea frasi per trasferire chiamate a colleghi specializzati. Usa nomi di reparti specifici (Ufficio Tecnico, Amministrazione, etc.)"
            ],
            "health": [
                "Genera frasi complete per prenotazioni mediche telefoniche. Usa specializzazioni specifiche (cardiologia, dermatologia, etc.) e orari concreti",
                "Crea conferme per visite specialistiche con dettagli specifici: date, orari, preparazioni necessarie",
                "Scrivi spiegazioni per procedure mediche specifiche. Ogni frase deve essere completa e professionale senza placeholder",
                "Genera richieste per documenti sanitari specifici: referti, certificati, prescrizioni mediche",
                "Crea comunicazioni sui tempi di attesa per esami specifici: risonanza, TAC, ecografie"
            ],
            "tech_it": [
                "Genera supporto tecnico per problemi informatici specifici: password, email, software aziendali",
                "Crea guide telefoniche per software specifici: Office, browser, applicazioni aziendali",
                "Scrivi spiegazioni per procedure di backup e sicurezza informatica concrete e specifiche",
                "Genera assistenza per problemi di accesso specifici: account bloccati, password scadute, autenticazione",
                "Crea supporto per applicazioni aziendali specifiche con soluzioni concrete"
            ],
            "tech_tax": [
                "Genera assistenza per pratiche fiscali italiane specifiche: dichiarazione redditi, F24, codice fiscale",
                "Crea spiegazioni per procedure Agenzia delle Entrate concrete: SPID, cassetto fiscale, comunicazioni",
                "Scrivi guide per compilazione documenti fiscali specifici con scadenze precise",
                "Genera informazioni su pagamenti fiscali specifici: IMU, TASI, bollo auto, canone RAI",
                "Crea assistenza per documenti tributari specifici: CUD, 730, UNICO, fatturazione elettronica"
            ]
        }

    def setup_logging(self, output_dir):
        """Setup logging system (file only, no console output)"""
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"sentence_generation_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        self.logger = logging.getLogger('SentenceGenerator')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler only - NO console output
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def generate_sentence_hash(self, sentence):
        """Generate hash for duplicate detection"""
        normalized = re.sub(r'[^\w\s]', '', sentence.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def check_jaro_winkler_similarity(self, sentence, threshold=0.85):
        """Check similarity using Jaro-Winkler algorithm"""
        sentence_clean = re.sub(r'[^\w\s]', '', sentence.lower()).strip()
        
        # Check against recent sentences (last 100 for performance)
        recent_sentences = self.generated_sentences[-100:] if len(self.generated_sentences) > 100 else self.generated_sentences
        
        for existing in recent_sentences:
            existing_clean = re.sub(r'[^\w\s]', '', existing.lower()).strip()
            similarity = jaro_winkler_similarity(sentence_clean, existing_clean)
            
            if similarity > threshold:
                self.logger.debug(f"High similarity ({similarity:.3f}): '{sentence[:50]}...' vs '{existing[:50]}...'")
                return True
        
        return False

    def check_repetitive_patterns(self, sentence):
        """Check for repetitive opening patterns"""
        # Common repetitive patterns to avoid
        repetitive_patterns = [
            r'^Buongiorno,?\s+gentile\s+cliente[!,.]',
            r'^Gentile\s+cliente,?\s+grazie\s+per',
            r'^Salve,?\s+gentile\s+cliente',
            r'^Buonasera,?\s+gentile\s+cliente',
            r'^Ciao,?\s+gentile\s+cliente',
            r'^Gentile\s+signore?\s*[,/]?\s*signora',
        ]
        
        sentence_lower = sentence.lower()
        
        # Count how many sentences start with similar patterns
        pattern_count = 0
        for existing in self.generated_sentences[-50:]:  # Check last 50
            existing_lower = existing.lower()
            for pattern in repetitive_patterns:
                if re.search(pattern, sentence_lower) and re.search(pattern, existing_lower):
                    pattern_count += 1
                    break
        
        # If more than 3 sentences with similar opening pattern, reject
        if pattern_count > 3:
            self.logger.debug(f"Repetitive pattern detected: {sentence[:50]}...")
            return True
        
        return False

    def is_duplicate(self, sentence):
        """Enhanced duplicate detection with Jaro-Winkler similarity"""
        # Hash-based duplicate check
        sentence_hash = self.generate_sentence_hash(sentence)
        if sentence_hash in self.sentence_hashes:
            self.stats["duplicate_sentences"] += 1
            return True
        
        # Jaro-Winkler similarity check
        if self.check_jaro_winkler_similarity(sentence, threshold=0.85):
            self.stats["similarity_rejected"] += 1
            return True
        
        # Repetitive pattern check
        if self.check_repetitive_patterns(sentence):
            self.stats["similarity_rejected"] += 1
            return True
        
        return False

    def clean_and_validate_sentence(self, sentence):
        """Improved cleaning and validation with reduced false positives"""
        # Clean the sentence
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        # Remove common instruction artifacts
        sentence = re.sub(r'^(Ecco|Certo|Certamente|Naturalmente|Ovviamente)[,:]?\s*', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'^(Scrivi|Genera|Crea)[^.]*[.:]?\s*', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'^(Una possibile frase|Ecco una frase)[^:]*:\s*', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'^\d+[\.\)]\s*', '', sentence)  # Remove numbering
        sentence = re.sub(r'^[-•*]\s*', '', sentence)  # Remove bullets
        
        # Ensure proper capitalization
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        # Ensure proper punctuation
        if sentence and sentence[-1] not in '.!?':
            sentence += '.'
        
        # Basic validation checks
        if not sentence or len(sentence) < 15 or len(sentence) > 200:
            self.logger.debug(f"Length validation failed: {len(sentence)} chars")
            return None
        
        word_count = len(sentence.split())
        if word_count < 4 or word_count > 35:
            self.logger.debug(f"Word count validation failed: {word_count} words")
            return None
        
        # Check for placeholder patterns (more comprehensive)
        placeholder_patterns = [
            r'\[.*?\]',  # [placeholder]
            r'\{.*?\}',  # {placeholder}
            r'<.*?>',    # <placeholder>
            r'\b(NOME|COGNOME|CLIENTE|SERVIZIO|PRODOTTO)\b',  # Common placeholders
            r'\b(Area Specializzata|Nome Cliente|Tipo Servizio)\b',
            r'\.\.\.',   # Ellipsis indicating incomplete
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                self.logger.debug(f"Placeholder detected: {sentence}")
                return None
        
        # Enhanced Italian grammar validation (more lenient)
        italian_indicators = [
            # Articles
            'il', 'la', 'lo', 'gli', 'le', 'un', 'una', 'uno', 'del', 'della', 'dello', 'degli', 'delle',
            # Prepositions
            'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'nel', 'nella', 'dal', 'dalla',
            # Common verbs
            'è', 'sono', 'ha', 'hanno', 'può', 'deve', 'vuole', 'viene', 'va', 'fa', 'dice', 'chiede',
            # Pronouns
            'lei', 'lui', 'loro', 'suo', 'sua', 'suoi', 'sue', 'nostro', 'nostra', 'vostro', 'vostra',
            # Common call center words
            'cliente', 'servizio', 'assistenza', 'operatore', 'richiesta', 'informazioni', 
            'appuntamento', 'prenotazione', 'conferma', 'disponibile', 'cortesia', 'gentile',
            'buongiorno', 'buonasera', 'salve', 'grazie', 'prego', 'scusi', 'posso', 'vorrei'
        ]
        
        # Count Italian indicators (more lenient threshold)
        sentence_lower = sentence.lower()
        italian_count = sum(1 for indicator in italian_indicators if indicator in sentence_lower)
        
        if italian_count < 2:  # Reduced from previous stricter requirements
            self.logger.debug(f"Insufficient Italian indicators ({italian_count}): {sentence}")
            return None
        
        # Check for obvious grammar errors (more specific patterns)
        error_patterns = [
            r'\b\w{20,}',  # Very long words (likely errors)
            r'\b[bcdfghjklmnpqrstvwxyz]{5,}\b',  # Too many consonants
            r'\b[aeiou]{4,}\b',  # Too many vowels
            r'(.)\1{3,}',  # Repeated characters
            r'\b(xyz|qwerty|asdf)\b',  # Keyboard mashing
            r'\d{10,}',  # Very long numbers
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, sentence.lower()):
                self.logger.debug(f"Error pattern detected: {sentence}")
                return None
        
        return sentence

    def write_sentence_to_file(self, sentence):
        """Write sentence to output file immediately"""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(sentence + '\n')
                f.flush()
        except Exception as e:
            self.logger.error(f"Error writing to file: {e}")

    def build_context_prompt(self, domain, batch_size=10):
        """Build improved context-aware prompt with better instructions"""
        base_prompt = random.choice(self.domain_prompts[domain])
        
        # Add context from recent sentences to avoid duplicates
        recent_sentences = self.generated_sentences[-30:] if len(self.generated_sentences) > 30 else self.generated_sentences
        
        # Create variety instructions
        variety_instructions = [
            "Usa aperture diverse: 'Salve', 'Buongiorno', 'La ringrazio', 'Mi scusi', 'Posso aiutarla'",
            "Varia il registro: formale, cordiale, professionale ma caloroso",
            "Usa nomi specifici quando appropriato: dott. Rossi, ufficio amministrativo, reparto cardiologia",
            "Evita frasi generiche, sii specifico e concreto",
            "Non usare placeholder come [nome], [servizio], [data] - usa dettagli reali"
        ]
        
        if recent_sentences:
            # Extract common starting patterns to avoid
            starting_patterns = []
            for sentence in recent_sentences[-10:]:
                words = sentence.split()[:3]  # First 3 words
                if len(words) >= 2:
                    starting_patterns.append(' '.join(words))
            
            unique_patterns = list(set(starting_patterns))[:5]  # Top 5 unique patterns
            
            prompt = f"""{base_prompt}

ISTRUZIONI SPECIFICHE:
- Genera ESATTAMENTE {batch_size} frasi complete e diverse
- Ogni frase deve essere una singola utterance completa (non dialoghi)
- NON usare placeholder, parentesi quadre, o segnaposto generici
- Usa dettagli specifici e concreti
- {random.choice(variety_instructions)}

EVITA questi inizi già usati di recente:
{chr(10).join([f'- "{pattern}..."' for pattern in unique_patterns[:3]])}

FORMATO RICHIESTO:
- Una frase per riga
- Nessuna numerazione
- Nessuna spiegazione aggiuntiva
- Solo le {batch_size} frasi richieste

Genera ora {batch_size} frasi uniche e professionali:"""

        else:
            prompt = f"""{base_prompt}

ISTRUZIONI SPECIFICHE:
- Genera ESATTAMENTE {batch_size} frasi complete e diverse
- Ogni frase deve essere una singola utterance completa (non dialoghi)
- NON usare placeholder, parentesi quadre, o segnaposto generici
- Usa dettagli specifici e concreti
- {random.choice(variety_instructions)}

FORMATO RICHIESTO:
- Una frase per riga
- Nessuna numerazione
- Nessuna spiegazione aggiuntiva
- Solo le {batch_size} frasi richieste

Genera ora {batch_size} frasi uniche e professionali:"""
        
        return prompt

    def generate_with_exllamav2(self, prompt, max_new_tokens=400):
        """Generate text using ExLlamaV2 with Mistral template"""
        try:
            # Apply Mistral template
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            # Encode the prompt
            from exllamav2.generator import ExLlamaV2DynamicJob
            instruction_ids = self.tokenizer.encode(formatted_prompt, add_bos=True)
            
            # Create generation job
            job = ExLlamaV2DynamicJob(
                input_ids=instruction_ids,
                max_new_tokens=max_new_tokens,
                stop_conditions=[self.tokenizer.eos_token_id],
                gen_settings=self.settings
            )
            
            # Enqueue the job
            self.generator.enqueue(job)
            
            # Collect the response
            response_text = ""
            eos = False
            
            while not eos:
                results = self.generator.iterate()
                for result in results:
                    if result["stage"] == "streaming":
                        eos = result["eos"]
                        if "text" in result:
                            response_text += result["text"]
            
            return response_text.strip()
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return ""

    def generate_batch_sentences(self, domain, batch_size=10):
        """Generate a batch of sentences using ExLlamaV2 with improved parsing"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                self.stats["total_attempts"] += 1
                
                # Build context-aware prompt
                prompt = self.build_context_prompt(domain, batch_size)
                
                # Generate with ExLlamaV2
                response = self.generate_with_exllamav2(prompt, max_new_tokens=500)
                
                if not response:
                    self.stats["invalid_sentences"] += 1
                    self.logger.debug(f"❌ Empty response for domain: {domain}")
                    continue
                
                # Parse sentences from response with improved logic
                sentences = []
                lines = response.split('\n')
                
                for line in lines:
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Remove various formatting artifacts
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove numbering
                    line = re.sub(r'^[-•*]\s*', '', line)      # Remove bullets
                    line = re.sub(r'^["\'""]', '', line)       # Remove opening quotes
                    line = re.sub(r'["\'""]$', '', line)       # Remove closing quotes
                    line = re.sub(r'^Frase\s*\d*[:\.]?\s*', '', line, flags=re.IGNORECASE)  # Remove "Frase 1:"
                    line = re.sub(r'^Esempio\s*\d*[:\.]?\s*', '', line, flags=re.IGNORECASE)  # Remove "Esempio:"
                    
                    # Skip lines that are instructions or meta-text
                    skip_patterns = [
                        r'^(Ecco|Certamente|Naturalmente)',
                        r'^(Le frasi|Queste frasi)',
                        r'^(Spero|Confido)',
                        r'^\w+\s*:$',  # Single word followed by colon
                        r'^(Nota|NB|PS)[:\.]',
                    ]
                    
                    should_skip = False
                    for skip_pattern in skip_patterns:
                        if re.search(skip_pattern, line, re.IGNORECASE):
                            should_skip = True
                            break
                    
                    if should_skip:
                        continue
                    
                    # Process the line
                    if line:
                        cleaned = self.clean_and_validate_sentence(line)
                        if cleaned and not self.is_duplicate(cleaned):
                            sentences.append(cleaned)
                            self.generated_sentences.append(cleaned)
                            self.sentence_hashes.add(self.generate_sentence_hash(cleaned))
                            self.stats["valid_sentences"] += 1
                            
                            # Write immediately to file
                            self.write_sentence_to_file(cleaned)
                            
                            # Log to file only (no console output)
                            self.logger.debug(f"✅ Generated: {cleaned}")
                        else:
                            self.stats["invalid_sentences"] += 1
                            if cleaned:
                                # Log to file only (no console output)
                                self.logger.debug(f"❌ Duplicate/Similar: {cleaned[:50]}...")
                            else:
                                # Log to file only (no console output)
                                self.logger.debug(f"❌ Invalid grammar: {line[:50]}...")
                
                return sentences
                
            except Exception as e:
                self.stats["invalid_sentences"] += 1
                # Log to file only (no console output)
                self.logger.debug(f"❌ Batch generation error: {str(e)}")
                continue
        
        return []

    def update_progress_line(self, current, target, domain):
        """Update progress on same line (overwrite previous)"""
        progress_pct = (current / target) * 100
        valid = self.stats["valid_sentences"]
        invalid = self.stats["invalid_sentences"]
        duplicates = self.stats["duplicate_sentences"]
        similar = self.stats["similarity_rejected"]
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * current // target)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Overwrite the same line
        progress_text = f"\r[{bar}] {progress_pct:.1f}% | {domain} | Valid: {valid} | Invalid: {invalid} | Dups: {duplicates} | Similar: {similar} | Current: {current}/{target}"
        
        print(progress_text, end='', flush=True)

    def generate_domain_sentences(self, domain, target_count):
        """Generate sentences for a specific domain using larger batches"""
        sentences = []
        failed_batches = 0
        max_failed_batches = 20
        batch_size = 10  # Increased batch size for better efficiency
        
        while len(sentences) < target_count and failed_batches < max_failed_batches:
            remaining = target_count - len(sentences)
            current_batch_size = min(batch_size, remaining)
            
            batch_sentences = self.generate_batch_sentences(domain, current_batch_size)
            
            if batch_sentences:
                sentences.extend(batch_sentences)
                failed_batches = 0
                
                # Update progress line (overwrite same line)
                self.update_progress_line(len(sentences), target_count, domain)
                
            else:
                failed_batches += 1
                # Still update progress even on failed batches
                self.update_progress_line(len(sentences), target_count, domain)
        
        return sentences

    def generate_all_sentences(self, total_target=1000):
        """Generate all sentences with improved distribution and tracking"""
        print(f"🎯 Target: {total_target} sentences")
        print(f"📁 Output file: {self.output_file}")
        print(f"🤖 Using quantized Ministral-8B (4.0bpw) with ExLlamaV2")
        print("🚀 Starting generation with enhanced quality controls...\n")
        
        # Distribution
        distribution = {
            "health": int(total_target * 0.40),
            "general": int(total_target * 0.30),
            "tech_it": int(total_target * 0.20),
            "tech_tax": int(total_target * 0.10)
        }
        
        all_sentences = []
        start_time = time.time()
        
        # Generate for each domain
        for domain, count in distribution.items():
            domain_sentences = self.generate_domain_sentences(domain, count)
            all_sentences.extend(domain_sentences)
            
            # New line after each domain completion
            print()  # Move to next line
        
        end_time = time.time()
        
        # Final summary with enhanced metrics
        print(f"\n{'='*80}")
        print("GENERATION COMPLETED!")
        print(f"{'='*80}")
        print(f"⏱️  Time: {end_time - start_time:.2f} seconds")
        print(f"📊 Total valid: {self.stats['valid_sentences']} sentences")
        print(f"❌ Total invalid: {self.stats['invalid_sentences']} attempts")
        print(f"🔄 Duplicates rejected: {self.stats['duplicate_sentences']}")
        print(f"📝 Similarity rejected: {self.stats['similarity_rejected']}")
        print(f"📁 Output file: {self.output_file}")
        
        if self.stats['total_attempts'] > 0:
            print(f"🎯 Success rate: {self.stats['valid_sentences']/self.stats['total_attempts']*100:.1f}%")
        
        if len(self.generated_sentences) > 0:
            print(f"🔄 Uniqueness: {len(set(self.generated_sentences))/len(self.generated_sentences)*100:.1f}%")
        
        # Quality analysis
        if all_sentences:
            lengths = [len(s) for s in all_sentences]
            word_counts = [len(s.split()) for s in all_sentences]
            
            print(f"\n📏 QUALITY METRICS:")
            print(f"   Length range: {min(lengths)}-{max(lengths)} chars")
            print(f"   Word range: {min(word_counts)}-{max(word_counts)} words")
            print(f"   Average length: {sum(lengths)/len(lengths):.1f} chars")
            print(f"   Average words: {sum(word_counts)/len(word_counts):.1f} words")
        
        return all_sentences

def _get_sentence_domain(sentence_index, domain_distribution):
    """Helper function to determine sentence domain based on index"""
    current_idx = 0
    for domain, count in domain_distribution.items():
        if sentence_index < current_idx + count:
            return domain
        current_idx += count
    return "unknown"

def main(model_path="/content/models/Ministral-8B-Instruct-2410/4.0bpw", save_path=".", target_sentences=1000):
    """Main execution function with improved error handling"""
    print("🇮🇹 IMPROVED ITALIAN CALL CENTER SENTENCE GENERATOR (ExLlamaV2 + Quantized Ministral)")
    print("=" * 90)
    
    try:
        # Initialize generator
        generator = ItalianCallCenterSentenceGenerator(
            model_path=model_path,
            output_dir=save_path
        )
        
        # Generate sentences
        sentences = generator.generate_all_sentences(target_sentences)
        
        # Create final CSV and JSON files
                    # Create final CSV and JSON files
        if sentences:
            df = pd.DataFrame({
                'sentence': sentences,
                'length': [len(s) for s in sentences],
                'word_count': [len(s.split()) for s in sentences]
            })
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save CSV
            csv_path = os.path.join(save_path, f"italian_sentences_{timestamp}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"💾 CSV saved: {csv_path}")
            
            # Save JSON with comprehensive stats
            json_path = os.path.join(save_path, f"italian_sentences_{timestamp}.json")
            
            # Calculate detailed statistics
            length_stats = {
                "min": int(df['length'].min()),
                "max": int(df['length'].max()),
                "mean": float(df['length'].mean()),
                "median": float(df['length'].median()),
                "std": float(df['length'].std())
            }
            
            word_stats = {
                "min": int(df['word_count'].min()),
                "max": int(df['word_count'].max()),
                "mean": float(df['word_count'].mean()),
                "median": float(df['word_count'].median()),
                "std": float(df['word_count'].std())
            }
            
            # Domain distribution
            domain_distribution = {
                "health": int(target_sentences * 0.40),
                "general": int(target_sentences * 0.30),
                "tech_it": int(target_sentences * 0.20),
                "tech_tax": int(target_sentences * 0.10)
            }
            
            # Calculate Jaro-Winkler diversity metrics
            diversity_scores = []
            if len(sentences) > 1:
                sample_size = min(100, len(sentences))  # Sample for performance
                sample_sentences = random.sample(sentences, sample_size)
                
                for i, sent1 in enumerate(sample_sentences):
                    for sent2 in sample_sentences[i+1:]:
                        try:
                            score = jaro_winkler_similarity(sent1.lower(), sent2.lower())
                            diversity_scores.append(1 - score)  # Convert similarity to diversity
                        except:
                            continue
            
            diversity_stats = {
                "mean_diversity": float(sum(diversity_scores) / len(diversity_scores)) if diversity_scores else 0.0,
                "min_diversity": float(min(diversity_scores)) if diversity_scores else 0.0,
                "max_diversity": float(max(diversity_scores)) if diversity_scores else 0.0,
                "sample_size": len(diversity_scores)
            }
            
            json_data = {
                "metadata": {
                    "total_sentences": len(sentences),
                    "target_sentences": target_sentences,
                    "success_rate": len(sentences) / target_sentences * 100,
                    "timestamp": timestamp,
                    "model_used": "Ministral-8B-Instruct-2410",
                    "model_type": "exllamav2_quantized_4bpw",
                    "model_path": model_path,
                    "unique_sentences": len(set(sentences)),
                    "uniqueness_rate": len(set(sentences)) / len(sentences) * 100,
                    "length_statistics": length_stats,
                    "word_statistics": word_stats,
                    "diversity_statistics": diversity_stats,
                    "generation_stats": generator.stats,
                    "domain_distribution": domain_distribution,
                    "quality_improvements": {
                        "jaro_winkler_similarity": "enabled",
                        "repetitive_pattern_detection": "enabled",
                        "placeholder_filtering": "enhanced",
                        "batch_size": 10,
                        "temperature": 0.8,
                        "repetition_penalty": 1.15
                    }
                },
                "sentences": [
                    {
                        "id": i + 1,
                        "text": sentence,
                        "length": len(sentence),
                        "word_count": len(sentence.split()),
                        "domain": _get_sentence_domain(i, domain_distribution)
                    }
                    for i, sentence in enumerate(sentences)
                ]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"💾 JSON saved: {json_path}")
            
            # Display sample sentences by domain with quality analysis
            print(f"\n🔍 SAMPLE SENTENCES BY DOMAIN:")
            print("-" * 80)
            
            # Show samples from each domain
            current_idx = 0
            for domain, count in domain_distribution.items():
                print(f"\n📋 {domain.upper()} ({count} sentences):")
                end_idx = min(current_idx + 3, len(sentences))  # Show 3 samples per domain
                for i in range(current_idx, end_idx):
                    if i < len(sentences):
                        print(f"  {i+1:2d}. {sentences[i]}")
                current_idx += count
            
            # Enhanced quality metrics
            uniqueness = len(set(sentences)) / len(sentences) * 100
            print(f"\n📊 ENHANCED QUALITY METRICS:")
            print(f"   Uniqueness: {uniqueness:.1f}%")
            print(f"   Length range: {df['length'].min()}-{df['length'].max()} chars")
            print(f"   Word range: {df['word_count'].min()}-{df['word_count'].max()} words")
            print(f"   Average length: {df['length'].mean():.1f} chars")
            print(f"   Average words: {df['word_count'].mean():.1f} words")
            
            if diversity_scores:
                print(f"   Jaro-Winkler diversity: {diversity_stats['mean_diversity']:.3f}")
                print(f"   Diversity range: {diversity_stats['min_diversity']:.3f}-{diversity_stats['max_diversity']:.3f}")
            
            # Enhanced performance metrics
            if generator.stats['total_attempts'] > 0:
                efficiency = generator.stats['valid_sentences'] / generator.stats['total_attempts'] * 100
                print(f"\n⚡ ENHANCED PERFORMANCE:")
                print(f"   Efficiency: {efficiency:.1f}% success rate")
                print(f"   Total attempts: {generator.stats['total_attempts']}")
                print(f"   Valid sentences: {generator.stats['valid_sentences']}")
                print(f"   Invalid attempts: {generator.stats['invalid_sentences']}")
                print(f"   Duplicates rejected: {generator.stats['duplicate_sentences']}")
                print(f"   Similarity rejected: {generator.stats['similarity_rejected']}")
            
            # Italian language quality indicators
            italian_words = ['il', 'la', 'di', 'che', 'per', 'con', 'può', 'deve', 'sono', 'è', 'cliente', 'servizio']
            italian_coverage = []
            for sentence in sentences[:100]:  # Check first 100 sentences
                words_found = sum(1 for word in italian_words if word in sentence.lower())
                italian_coverage.append(words_found / len(italian_words))
            
            if italian_coverage:
                avg_italian_coverage = sum(italian_coverage) / len(italian_coverage) * 100
                print(f"\n🇮🇹 ITALIAN LANGUAGE QUALITY:")
                print(f"   Italian indicators coverage: {avg_italian_coverage:.1f}%")
            
            # Pattern analysis
            opening_patterns = {}
            for sentence in sentences:
                words = sentence.split()[:2]  # First 2 words
                if len(words) >= 2:
                    pattern = ' '.join(words)
                    opening_patterns[pattern] = opening_patterns.get(pattern, 0) + 1
            
            # Show most common patterns
            sorted_patterns = sorted(opening_patterns.items(), key=lambda x: x[1], reverse=True)
            print(f"\n🔄 PATTERN DIVERSITY ANALYSIS:")
            print(f"   Unique opening patterns: {len(opening_patterns)}")
            print(f"   Most common patterns:")
            for pattern, count in sorted_patterns[:5]:
                percentage = (count / len(sentences)) * 100
                print(f"     '{pattern}': {count} times ({percentage:.1f}%)")
            
            # ExLlamaV2 specific metrics
            print(f"\n🔧 EXLLAMAV2 ENHANCED METRICS:")
            print(f"   Model: Quantized Ministral-8B (4.0bpw)")
            print(f"   Memory efficient: ✅ Quantized inference")
            print(f"   Template: Mistral [INST] format")
            print(f"   Context length: 8192 tokens")
            print(f"   Temperature: 0.8 (increased for variety)")
            print(f"   Repetition penalty: 1.15 (enhanced)")
            print(f"   Batch size: 10 sentences per generation")
            print(f"   Jaro-Winkler similarity: ✅ Enabled (threshold: 0.85)")
            print(f"   Pattern detection: ✅ Enhanced repetition filtering")
        
        return sentences
        
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Generate Italian Call Center Sentences (Enhanced ExLlamaV2 + Quantized Ministral)')
    parser.add_argument('--target', type=int, default=1000, help='Target number of sentences (default: 1000)')
    parser.add_argument('--output', type=str, default='.', help='Output directory (default: current directory)')
    parser.add_argument('--model-path', type=str, default='/content/models/Ministral-8B-Instruct-2410/4.0bpw', 
                       help='Path to quantized model directory (default: /content/models/Ministral-8B-Instruct-2410/4.0bpw)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Enhanced Italian Call Center Sentence Generator...")
    print(f"🎯 Target: {args.target} sentences")
    print(f"📁 Output: {args.output}")
    print(f"🤖 Model: {args.model_path}")
    print("🧠 Using Enhanced ExLlamaV2 with quantized Ministral-8B (4.0bpw)")
    print("🔧 New features: Jaro-Winkler similarity, pattern detection, enhanced validation")
    print("📝 Progress will show detailed metrics (debug in logs only)")
    
    sentences = main(
        model_path=args.model_path,
        save_path=args.output, 
        target_sentences=args.target
    )
    
    if sentences:
        print(f"\n🎉 SUCCESS! Generated {len(sentences)} high-quality Italian sentences!")
        print(f"📊 Files saved in: {args.output}")
        print(f"📋 Check the log file in {args.output}/logs/ for detailed debug information")
        print("\n💡 ENHANCED FEATURES:")
        print("   ✅ Jaro-Winkler similarity detection (threshold: 0.85)")
        print("   ✅ Repetitive pattern filtering")
        print("   ✅ Enhanced placeholder detection")
        print("   ✅ Improved Italian grammar validation")
        print("   ✅ Larger batch generation (10 sentences/batch)")
        print("   ✅ Better prompt engineering with variety instructions")
        print("\n📁 OUTPUT FILES:")
        print("   • italian_sentences_TIMESTAMP.txt (raw sentences)")
        print("   • italian_sentences_TIMESTAMP.csv (structured data)")
        print("   • italian_sentences_TIMESTAMP.json (full metadata + diversity metrics)")
        print("   • logs/sentence_generation_TIMESTAMP.log (detailed debug logs)")
        print("\n🔍 QUALITY IMPROVEMENTS:")
        print("   • Reduced false positives in validation")
        print("   • Better duplicate detection with similarity scoring")
        print("   • Enhanced pattern diversity analysis")
        print("   • Improved placeholder filtering")
        print("   • More specific Italian grammar checks")
        print("\n⚙️ INSTALLATION REQUIREMENTS:")
        print("   pip install jellyfish  # For Jaro-Winkler similarity")
        print("   pip install exllamav2  # For model inference")
        print("   pip install pandas     # For data processing")
    else:
        print("\n💥 FAILED! Check the error messages above.")
        print("💡 TROUBLESHOOTING:")
        print("   • Install jellyfish: pip install jellyfish")
        print("   • Ensure model path exists: /content/models/Ministral-8B-Instruct-2410/4.0bpw")
        print("   • Check if model files are properly downloaded")
        print("   • Try with smaller target: --target 100")
        print("   • Check logs directory for detailed error information")
        print("   • Ensure ExLlamaV2 is properly installed: pip install exllamav2")
















