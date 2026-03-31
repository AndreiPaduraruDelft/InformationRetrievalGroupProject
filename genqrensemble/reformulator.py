from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Code below is changed to work on my AMD GPU. Changing it back will probably work
class HFReformulator:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if device == "dml":
            import torch_directml
            # Select dedicated GPU over integrated (e.g. RX 7900 XT over Radeon(TM) Graphics)
            torch_device = None
            for i in range(torch_directml.device_count()):
                name = torch_directml.device_name(i)
                if "7900" in name or ("RX" in name and "Radeon(TM) Graphics" not in name):
                    torch_device = torch_directml.device(i)
                    print(f"  Selected DirectML device {i}: {name}")
                    break
            if torch_device is None:
                torch_device = torch_directml.device(torch_directml.device_count() - 1)
                print(f"  Falling back to DirectML device: {torch_directml.device_name(torch_directml.device_count()-1)}")
            use_fp16 = True
        elif device == "cpu":
            torch_device = torch.device("cpu")
            use_fp16 = False
        elif device == "cuda":
            torch_device = torch.device("cuda")
            use_fp16 = True
        else:
            torch_device = torch.device(device)
            use_fp16 = False

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=torch.float16 if use_fp16 else torch.float32,
            low_cpu_mem_usage=True,
        )
        print(f"  Moving model to {torch_device} ...")
        self.model = self.model.to(torch_device)
        self.model.eval()
        self.device = torch_device

        if device == "dml":
            # Monkey-patch: DirectML can't handle masked_fill with float16 min
            # in _prepare_4d_causal_attention_mask_with_cache_position.
            # Returning None makes the decoder use default causal behavior.
            self.model.decoder._update_causal_mask = lambda *args, **kwargs: None

        print(f"  Model ready on {torch_device}")

    def generate_keywords(self, instruction: str, query: str) -> str:
        prompt = f"{instruction}: {query}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.92,
                top_k=200,
                repetition_penalty=1.2,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
