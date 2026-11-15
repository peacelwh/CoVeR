"""Hyper-parameters of the paper (Sec. 4.1 and supplementary Sec. A).

Shared by LLM-Fusion and CoVeR: Qwen3-0.6B with LoRA on the attention projections, K = 4 fusion tokens, T = 50 tokens
per modality, lambda_task = 10, lambda_con = lambda_reg = 1, AdamW with a cosine schedule, batch 32, bf16 and
activation checkpointing, a default budget of 30 epochs (15 on CMU-MOSEI) on one 24 GB GPU.

Two recipes are provided for each method:

``selected``  per-dataset hyper-parameters chosen on the validation split only (comparisons with the state of the
              art, IEMOCAP, MR-Bench-H and permanent missingness). CoVeR: tau = 0.4, lambda_uni = 2, the MSE loss and
              lr 2e-4 everywhere, rank 16 except on CH-SIMS (32), weight averaging (decay 0.99) except on CH-SIMSv2;
              CMU-MOSEI reuses the CMU-MOSI setting. LLM-Fusion receives the same search (rank, learning rate,
              epochs and weight averaging); its selected values are listed in ``SELECTED`` below.
``default``   the default recipe of the backbone for every row of the analyses (test-time override, comparison of
              corrections, ablation): rank 16, lr 2e-4, no weight averaging; CoVeR uses tau = 0.3, lambda_uni = 1.
"""
from copy import deepcopy

DATASETS = ("mosi", "mosei", "sims", "simsv2", "iemocap", "meld", "mustardpp")
DATASET_NAMES = {"mosi": "CMU-MOSI", "mosei": "CMU-MOSEI", "sims": "CH-SIMS", "simsv2": "CH-SIMSv2",
                 "iemocap": "IEMOCAP", "meld": "MELD", "mustardpp": "MUStARD++"}
#: corpora with human unimodal labels; the others use out-of-fold expert predictions as unimodal judgments
HUMAN_UNIMODAL = ("sims", "simsv2")

COMMON = dict(
    llm_path="Qwen/Qwen3-0.6B", num_fusion_tokens=4, lora_dropout=0.1, dropout=0.15,
    lambda_task=10.0, lambda_con=1.0, lambda_reg=1.0,
    batch=32, lr=2e-4, weight_decay=1e-4, warmup_epochs=2, lr_floor=0.1, grad_clip=1.0,
    regression="mse", ema=0.0, lora_r=16,
)
EPOCHS = {"mosi": 30, "mosei": 15, "sims": 30, "simsv2": 30, "iemocap": 20, "meld": 15, "mustardpp": 30}
#: data-loader workers (also the number of mask streams of the random-missing-rate evaluation)
NUM_WORKERS = {"mosi": 2, "mosei": 4, "sims": 2, "simsv2": 2, "iemocap": 4, "meld": 4, "mustardpp": 4}

DEFAULT = {
    "llm_fusion": dict(),
    "cover": dict(tau=0.3, lambda_uni=1.0),
    "learned_gate": dict(),
}
#: learned reliability gate (comparison of corrections): the gates are supervised with the human unimodal labels
#: where the corpus has them, otherwise with the predictions of the views themselves
LEARNED_GATE = {
    True: dict(gate_alpha=0.5, gate_targets="unimodal", gate_agreement="polarity"),
    False: dict(gate_alpha=0.15, gate_targets="views", gate_agreement="close"),
}

_COVER = dict(tau=0.4, lambda_uni=2.0, lora_r=16, lr=2e-4, ema=0.99)
_FUSION = dict(lora_r=32, lr=2e-4, ema=0.99)
SELECTED = {
    "cover": {
        "mosi": dict(_COVER),
        "mosei": dict(_COVER),
        "sims": dict(_COVER, lora_r=32),
        "simsv2": dict(_COVER, ema=0.0),
        "iemocap": dict(_COVER),
        "meld": dict(_COVER),
        "mustardpp": dict(_COVER),
    },
    "llm_fusion": {
        "mosi": dict(_FUSION),
        "mosei": dict(_FUSION),
        "sims": dict(lora_r=16, lr=1e-4, ema=0.99, epochs=50),
        "simsv2": dict(lora_r=16, lr=1e-4, ema=0.99),
        "iemocap": dict(_FUSION),
        "meld": dict(_FUSION),
        "mustardpp": dict(_FUSION),
    },
}


def get_recipe(method: str, dataset: str, recipe: str = "selected") -> dict:
    """Hyper-parameters of ``method`` ('llm_fusion' | 'cover' | 'learned_gate') on ``dataset`` under ``recipe``
    ('selected' | 'default'; the learned gate has the default recipe only)."""
    assert method in ("llm_fusion", "cover", "learned_gate"), method
    assert dataset in DATASETS, dataset
    assert recipe in ("selected", "default"), recipe
    hp = deepcopy(COMMON)
    hp["epochs"] = EPOCHS[dataset]
    hp["num_workers"] = NUM_WORKERS[dataset]
    hp.update(tau=None, lambda_uni=0.0)
    if method == "learned_gate":
        hp.update(LEARNED_GATE[dataset in HUMAN_UNIMODAL])
    else:
        hp.update(DEFAULT[method] if recipe == "default" else SELECTED[method][dataset])
    hp.update(method=method, dataset=dataset, recipe=recipe)
    return hp
