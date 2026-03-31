from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class HFReformulator:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if device == "cuda":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            ).to(device)
        self.device = device

    def generate_keywords(self, instruction: str, query: str) -> str:
        prompt = f"{instruction}: {query}"
        input_device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(input_device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.92,
            top_k=200,
            repetition_penalty=1.2,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
