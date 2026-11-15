"""Shared fixtures of the tests: a synthetic corpus and a tiny randomly initialised decoder (nothing is downloaded)."""
import os
import sys

import numpy as np
import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DIMS = {"t": 12, "a": 6, "v": 5}
T = 8


def make_corpus(path, n=(48, 16, 16), unimodal=True, label_range=(-1.0, 1.0), seed=0):
    """Write a processed corpus in the format of ``cover.preprocess``."""
    rng = np.random.RandomState(seed)
    blob = {"meta": {"dataset": "sims" if unimodal else "mosi", "label_range": list(label_range), "T": T, "dims": dict(DIMS),
                     "has_unimodal_labels": unimodal}}
    for split, k in zip(("train", "valid", "test"), n):
        hi = label_range[1]
        y = rng.uniform(-hi, hi, size=k).astype(np.float32)
        s = {m: torch.tensor(rng.randn(k, T, d).astype(np.float32) + y[:, None, None]).half() for m, d in DIMS.items()}
        s["y"] = torch.tensor(y)
        s["id"] = [f"{split}_{i:04d}" for i in range(k)]
        if unimodal:
            for m in DIMS:
                s[f"y_{m}"] = torch.tensor(np.clip(y + 0.3 * rng.randn(k), -hi, hi).astype(np.float32))
        blob[split] = s
    torch.save(blob, path)
    return path


class TinyTokenizer:
    """Whitespace tokenizer with the call signature the model uses."""
    pad_token = "<pad>"
    eos_token = "<pad>"

    def __init__(self):
        self.vocab = {"<pad>": 0}

    def __call__(self, texts, padding=True, return_tensors="pt", padding_side="left"):
        ids = [[self.vocab.setdefault(w, len(self.vocab)) for w in t.split()] for t in texts]
        L = max(len(x) for x in ids)
        input_ids = torch.zeros(len(ids), L, dtype=torch.long)
        attention = torch.zeros(len(ids), L, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, L - len(x):] = torch.tensor(x)
            attention[i, L - len(x):] = 1
        return {"input_ids": input_ids, "attention_mask": attention}


def tiny_model(lora_r=4, num_fusion_tokens=2, seed=0, gate_alpha=None):
    """LLM-Fusion (with a learned reliability gate if ``gate_alpha`` is given) on a small randomly initialised decoder."""
    from transformers import Qwen2Config, Qwen2Model

    from cover.learned_gate import LearnedGateFusion
    from cover.model import LLMFusion
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    kwargs = dict(T=T, num_fusion_tokens=num_fusion_tokens, lora_r=lora_r, llm=Qwen2Model(cfg), tokenizer=TinyTokenizer())
    if gate_alpha is not None:
        return LearnedGateFusion(None, DIMS, gate_alpha=gate_alpha, **kwargs)
    return LLMFusion(None, DIMS, **kwargs)
