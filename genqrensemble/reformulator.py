from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class HFReformulator:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.device = device

    def generate_keywords(self, instruction: str, query: str) -> str:
        prompt = f"{instruction}: {query}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.92,
            top_k=200,
            repetition_penalty=1.2,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
