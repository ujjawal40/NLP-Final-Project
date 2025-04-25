from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
import torch


class MeditronGenerator:
    def __init__(self):
        self.config = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.paths["models"]["meditron"])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.paths["models"]["meditron"],
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    def generate_answer(self, question, contexts):
        """Generate answer using Meditron with RAG contexts"""
        context_str = "\n".join([f"Context {i + 1}: {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""You are a medical expert. Use these contexts:
{context_str}

Question: {question}
Format your answer as:
1. Final Decision: [yes/no/maybe]
2. Justification: [2-3 sentence explanation]
3. Context References: [List relevant context numbers]"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)