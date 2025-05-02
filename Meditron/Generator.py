from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import time
import gc
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model_name = "epfl-llm/meditron-7b"

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        logger.info("Tokenizer loaded successfully")

        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        logger.info("Model loaded successfully")

    def truncate_at_sentence(self, text, max_length=300):
        """Truncate text at the last complete sentence within max_length."""
        if len(text) <= max_length:
            return text

        # Find the last period within max_length
        truncated = text[:max_length]
        last_period = truncated.rfind('.')

        if last_period != -1:
            # Check if there's a space after the period
            if last_period + 1 < len(truncated) and truncated[last_period + 1] == ' ':
                return text[:last_period + 1].strip()
            # If no space, find the next space
            next_space = text[last_period:].find(' ')
            if next_space != -1:
                return text[:last_period + next_space].strip()

        # If no period found, try other sentence endings
        endings = ['. ', '? ', '! ']
        for ending in endings:
            last_end = truncated.rfind(ending)
            if last_end != -1:
                return text[:last_end + 1].strip()

        # If no sentence boundary found, truncate at last space
        last_space = truncated.rfind(' ')
        if last_space != -1:
            return text[:last_space].strip()

        return truncated.strip()

    def clean_text(self, text):
        """Simple text cleaning to fix basic spacing issues."""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix basic punctuation spacing
        text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
        return text.strip()

    def format_medical_text(self, text):
        """Format medical text with proper structure and spacing."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Process each sentence
        formatted_sentences = []
        for sentence in sentences:
            # Clean the sentence
            cleaned = self.clean_text(sentence)

            # Capitalize first letter
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]

            formatted_sentences.append(cleaned)

        # Join sentences with proper spacing
        return ' '.join(formatted_sentences)

    def preprocess_input(self, text):
        """Preprocess input text to fix mangled words and spacing."""
        # First, split all run-together words
        # This pattern looks for transitions between lowercase and uppercase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Split words that are run together without proper spacing
        # This looks for sequences of letters that should be separate words
        text = re.sub(r'([a-zA-Z])([a-zA-Z]{2,})', r'\1 \2', text)

        # Fix common medical term spacing
        medical_terms = {
            'Serotonergic': 'Serotonergic',
            'Central': 'Central',
            'Nervous': 'Nervous',
            'System': 'System',
            'Synaptic': 'Synaptic',
            'Concentrations': 'Concentrations',
            'Endogenous': 'Endogenous',
            'Neurotransmitter': 'Neurotransmitter',
            'Serotonin': 'Serotonin',
            'Clinical': 'Clinical',
            'Symptoms': 'Symptoms',
            'Receptors': 'Receptors',
            'Tremor': 'Tremor',
            'Shivering': 'Shivering',
            'Hyperreflexia': 'Hyperreflexia',
            'Clonus': 'Clonus',
            'Rigidity': 'Rigidity',
            'Muscle': 'Muscle',
            'Twitching': 'Twitching',
            'Ocular': 'Ocular',
            'Clonus': 'Clonus',
            'Dysarthria': 'Dysarthria',
            'Mental': 'Mental',
            'Confusion': 'Confusion',
            'Seizures': 'Seizures',
            'Agitation': 'Agitation',
            'Hyperthermia': 'Hyperthermia',
            'Diaphoresis': 'Diaphoresis',
            'Tachycardia': 'Tachycardia',
            'Hypertension': 'Hypertension',
            'Flushing': 'Flushing',
            'Headache': 'Headache',
            'Delirium': 'Delirium',
            'Vomiting': 'Vomiting',
            'Diarrhea': 'Diarrhea',
            'Gastrointestinal': 'Gastrointestinal',
            'Bleeding': 'Bleeding',
            'Coma': 'Coma',
            'Respiratory': 'Respiratory',
            'Arrest': 'Arrest',
            'Death': 'Death',
            'Mechanisms': 'Mechanisms',
            'Development': 'Development',
            'Disorder': 'Disorder',
            'Direct': 'Direct',
            'Binding': 'Binding',
            'Postsynaptic': 'Postsynaptic'
        }

        # Apply medical term fixes
        for term, replacement in medical_terms.items():
            text = re.sub(rf'\b{term}\b', replacement, text, flags=re.IGNORECASE)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Ensure proper sentence spacing
        text = re.sub(r'\.([A-Z])', r'. \1', text)

        return text.strip()

    def generate(self, query, contexts):
        start_time = time.time()
        logger.info(f"Processing query: {query}")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Filter out empty or invalid contexts
        valid_contexts = []
        for ctx in contexts[:2]:  # Only use top 2 contexts
            if isinstance(ctx, str) and len(ctx.strip()) > 0:
                valid_contexts.append(self.truncate_at_sentence(ctx))
            elif isinstance(ctx, dict) and 'text' in ctx and isinstance(ctx['text'], str) and len(ctx['text'].strip()) > 0:
                valid_contexts.append(self.truncate_at_sentence(ctx['text']))

        logger.info(f"Found {len(valid_contexts)} valid contexts")

        if not valid_contexts:
            return "I cannot answer based on available medical literature."

        # Create a more focused prompt
        context_str = "\n".join([
            f"Context {i + 1}: {ctx}" for i, ctx in enumerate(valid_contexts)
        ])

        prompt = f"""Based on the following medical contexts, please provide a concise answer to the question.

{context_str}

Question: {query}

Answer:"""

        logger.info("Tokenizing input...")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        logger.info("Input tokenized successfully")

        try:
            logger.info("Generating response...")
            with torch.inference_mode():  # More efficient than no_grad
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,  # Increased for complete responses
                    temperature=0.7,
                    do_sample=False,  # Disable sampling for more consistent output
                    num_beams=1,  # Use greedy decoding
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            logger.info("Response generated successfully")

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the answer part
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

            # Simple text cleaning
            answer = self.clean_text(answer)

            end_time = time.time()
            logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

            # Clear memory again
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."