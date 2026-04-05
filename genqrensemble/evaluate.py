import re

import ir_measures
import pyterrier as pt
from tqdm import tqdm

_DATASET_METRICS = {
    "trec-robust04": [ir_measures.nDCG @ 10, ir_measures.RR(rel=2), ir_measures.AP(rel=2), ir_measures.P @ 10],
}
_DEFAULT_METRICS = [ir_measures.nDCG @ 10, ir_measures.RR(rel=2), ir_measures.AP(rel=2), ir_measures.P @ 10]


def _query_swapper(reformulated_topics):
    """Transformer that replaces queries with pre-reformulated ones, keyed by qid."""
    qid_to_query = dict(zip(
        reformulated_topics["qid"].astype(str),
        reformulated_topics["query"],
    ))

    def swap(df):
        df = df.copy()
        df["query"] = df["qid"].astype(str).map(qid_to_query).fillna(df["query"])
        return df

    return pt.apply.generic(swap)


def _weighted_ensemble_swapper(original_topics, ensemble_topics, beta=0.05):
    """Build structured query_toks from original + reformulated queries.

    Original query terms contribute +1.0 per occurrence.
    Reformulated/ensemble query terms contribute +beta per occurrence.

    This means duplicates from both original and reformulated queries are kept
    and accumulated.

    Example:
        original:  "black hole formation"
        ensemble:  "black hole collapse black dwarf stellar"
        beta=0.05

        result:
        {
            "black": 1.05,
            "hole": 1.05,
            "formation": 1.0,
            "collapse": 0.05,
            "dwarf": 0.05,
            "stellar": 0.05
        }
    """
    orig_map = dict(zip(original_topics["qid"].astype(str), original_topics["query"]))
    ens_map = dict(zip(ensemble_topics["qid"].astype(str), ensemble_topics["query"]))

    def tokenise(text):
        return re.sub(r"[^\w\s]", "", text).lower().split()

    def build_query_toks(qid):
        orig_q = orig_map.get(qid, "")
        ens_q = ens_map.get(qid, orig_q)

        query_toks = {}

        # Original terms: +1.0 per occurrence
        for token in tokenise(orig_q):
            query_toks[token] = query_toks.get(token, 0.0) + 1.0

        # Reformulated terms: +beta per occurrence
        for token in tokenise(ens_q):
            query_toks[token] = query_toks.get(token, 0.0) + beta

        return query_toks

    def swap(df):
        df = df.copy()
        df["query_toks"] = df["qid"].astype(str).map(build_query_toks)
        return df

    return pt.apply.generic(swap)


def run_experiment(bm25, dataset_name, topics, qrels, flanqr_topics, ensemble_topics,
                   rerank=False, rerank_depth=1000, get_text_pipe=None, device="cpu"):
    flanqr_pipe          = _query_swapper(flanqr_topics)                                  >> bm25
    ensemble_pipe        = _query_swapper(ensemble_topics)                                >> bm25
    flanqr_weighted_pipe = _weighted_ensemble_swapper(topics, flanqr_topics,   beta=0.05) >> bm25
    weighted_pipe        = _weighted_ensemble_swapper(topics, ensemble_topics, beta=0.05) >> bm25

    pipelines = [bm25, flanqr_pipe, flanqr_weighted_pipe, ensemble_pipe, weighted_pipe]
    names     = ["BM25", "FlanQR", "FlanQR_beta_0_05", "GenQREnsemble", "GenQREnsemble_beta_0_05"]

    eval_metrics = _DATASET_METRICS.get(dataset_name, _DEFAULT_METRICS)

    if rerank:
        print(f"  reranking has started (depth={rerank_depth})")
        # Force safetensors loading to avoid CVE-2025-32434 torch.load block on torch < 2.6
        from transformers import T5ForConditionalGeneration
        _orig_from_pretrained = T5ForConditionalGeneration.from_pretrained.__func__
        @classmethod
        def _safe_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault("use_safetensors", True)
            return _orig_from_pretrained(cls, *args, **kwargs)
        T5ForConditionalGeneration.from_pretrained = _safe_from_pretrained

        # Compatibility shim: newer transformers removed batch_encode_plus;
        # pyterrier_t5 still calls it, so alias it to __call__.
        from transformers import T5Tokenizer as _T5Tok
        if not hasattr(_T5Tok, 'batch_encode_plus'):
            _T5Tok.batch_encode_plus = lambda self, *args, **kwargs: self(*args, **kwargs)

        from pyterrier_t5 import MonoT5ReRanker
        mono_t5  = MonoT5ReRanker(model="castorini/monot5-base-msmarco", verbose=False)

        # Override device if DirectML requested
        if device == "dml":
            import torch_directml
            dml_device = None
            for i in range(torch_directml.device_count()):
                name = torch_directml.device_name(i)
                if "7900" in name or ("RX" in name and "Radeon(TM) Graphics" not in name):
                    dml_device = torch_directml.device(i)
                    print(f"  MonoT5 using DirectML device {i}: {name}")
                    break
            if dml_device is None:
                dml_device = torch_directml.device(torch_directml.device_count() - 1)
                print(f"  MonoT5 falling back to DirectML device: {torch_directml.device_name(torch_directml.device_count()-1)}")

            # Patch T5Stack causal mask to avoid masked_fill (incompatible with DirectML)
            from transformers.models.t5.modeling_t5 import T5Stack
            @staticmethod
            def _dml_prepare_4d_causal_mask(
                attention_mask, sequence_length, target_length, dtype, cache_position, batch_size,
            ):
                import torch as _torch
                if attention_mask is not None and attention_mask.dim() == 4:
                    return attention_mask
                min_dtype = _torch.finfo(dtype).min
                causal_mask = _torch.full(
                    (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device,
                )
                if sequence_length != 1:
                    causal_mask = _torch.triu(causal_mask, diagonal=1)
                causal_mask *= (_torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)).to(dtype)
                causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
                if attention_mask is not None:
                    causal_mask = causal_mask.clone()
                    mask_length = attention_mask.shape[-1]
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
                    padding_mask = padding_mask == 0
                    min_val = _torch.tensor(min_dtype, dtype=dtype, device=causal_mask.device)
                    causal_mask[:, :, :, :mask_length] = _torch.where(
                        padding_mask, min_val, causal_mask[:, :, :, :mask_length],
                    )
                return causal_mask
            T5Stack._prepare_4d_causal_attention_mask_with_cache_position = _dml_prepare_4d_causal_mask

            mono_t5.device = dml_device
            mono_t5.model.to(dml_device)
        if get_text_pipe is None:
            get_text_pipe = pt.text.get_text(pt.get_dataset(f"irds:{dataset_name}"), "text")
        pipelines += [
            (bm25          % rerank_depth) >> get_text_pipe >> mono_t5,
            (flanqr_pipe   % rerank_depth) >> get_text_pipe >> mono_t5,
            (ensemble_pipe % rerank_depth) >> get_text_pipe >> mono_t5,
        ]
        names += ["BM25+MonoT5", "FlanQR+MonoT5", "GenQREnsemble+MonoT5"]

    return pt.Experiment(
        pipelines,
        topics,
        qrels,
        eval_metrics=eval_metrics,
        names=names,
        baseline=1,
        correction="holm",
    )
