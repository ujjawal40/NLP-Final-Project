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
            torch_dtype=torch.bfloat16,  # <---- THIS SOLVES OOM
            trust_remote_code=True
        )

    def generate(self, query, contexts):
        context_str = "\n".join([
            f"---CONTEXT {i + 1}---\n{str(ctx)[:1000]}\n" for i, ctx in enumerate(contexts)
        ])

        prompt = f"""<MEDICAL_QA>
You must strictly follow these rules:
1. Answer ONLY using the provided contexts.
2. If contexts are irrelevant, say: "I cannot answer based on available medical literature."
3. Cite the contexts like [CONTEXT 1] for each claim.
4. Never invent facts.

CONTEXTS:
{context_str}

QUESTION: {query}

ANSWER:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False,
            num_beams=5
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
