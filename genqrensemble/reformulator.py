from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch


class HFReformulator:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        config = AutoConfig.from_pretrained(model_id)
        self.is_encoder_decoder = getattr(config, 'is_encoder_decoder', False)

        if self.is_encoder_decoder:
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
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16, 
            ).to(device)
        
        self.device = device

    def generate_keywords(self, instruction: str, query: str) -> str:
        prompt = f"{instruction}: {query}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.5,
        )
        
        if self.is_encoder_decoder:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            input_length = inputs["input_ids"].shape[-1]
            return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)