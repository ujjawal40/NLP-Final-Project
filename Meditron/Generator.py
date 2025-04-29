from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class MedicalGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "epfl-llm/meditron-7b"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    def generate(self, query, contexts):
        # Filter out empty or invalid contexts
        valid_contexts = [ctx for ctx in contexts if ctx and isinstance(ctx, str) and len(ctx.strip()) > 0]

        if not valid_contexts:
            return "I cannot answer based on available medical literature."

        # Create a more natural prompt
        context_str = "\n\n".join([
            f"Medical context {i + 1}: {ctx}" for i, ctx in enumerate(valid_contexts)
        ])

        prompt = f"""Based on the following medical contexts, please answer the question:

{context_str}

Question: {query}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the answer part
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

            return answer

        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."
