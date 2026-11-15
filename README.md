<h1 align="center">Words Speak Louder? Exploring the Language Shortcut of LLM-Driven Incomplete Multimodal Learning</h1>

<p align="center">
  Wenhao&nbsp;Li<sup>1,2</sup>, Chong&nbsp;Xiao<sup>1</sup>, Xiaokun&nbsp;Bai<sup>1</sup>, Rundong&nbsp;He<sup>3</sup>, Zhuang&nbsp;Qi<sup>4</sup>, Xusheng&nbsp;Liang<sup>5,2</sup>, Zhibin&nbsp;Wu<sup>1</sup>, Qiangchang&nbsp;Wang<sup>1</sup>
  <br>
  <sup>1</sup>Shandong&nbsp;University &ensp; <sup>2</sup>Shenzhen&nbsp;Loop&nbsp;Area&nbsp;Institute &ensp; <sup>3</sup>The&nbsp;Hong&nbsp;Kong&nbsp;Polytechnic&nbsp;University &ensp; <sup>4</sup>National&nbsp;University&nbsp;of&nbsp;Singapore &ensp; <sup>5</sup>City&nbsp;University&nbsp;of&nbsp;Hong&nbsp;Kong
</p>

<p align="center">
  <a href="https://peacelwh.github.io/CoVeR/"><img src="https://img.shields.io/badge/Project-CoVeR-9cf" alt="Project page: CoVeR"></a>
  <a href="https://huggingface.co/datasets/Bluexxx/MR-Bench-H"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MR--Bench--H-dfb317" alt="Dataset: MR-Bench-H"></a>
  <a href="https://huggingface.co/Bluexxx/CoVeR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-CoVeR-dfb317" alt="Checkpoints: CoVeR"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-97ca00" alt="License: MIT"></a>
</p>

Code for **CoVeR** (Counterfactual-View Reliability learning), its backbone **LLM-Fusion**, and the evaluation on
**MR-Bench** (MR-Bench-A and MR-Bench-H).

Incomplete multimodal learning increasingly uses large language models as fusion engines: modality features enter as
token blocks and missing modalities become placeholders. LLM-driven fusion can override non-verbal evidence it can
read. Contrasting the full-input prediction with *counterfactual views* — the same model shown one modality or no
content — identifies these language shortcuts. CoVeR trains the views as unimodal judges and withholds fused
supervision where they disagree; it keeps the architecture and the inference cost of the backbone.

## Repository structure

```
README.md      this file
data/          datasets (not distributed), processed tensors, MR-Bench-A annotations, held-out clip lists -> data/README.md
src/cover/     core code: model, objectives, training, evaluation, benchmarks, diagnosis
scripts/       preprocessing, training, evaluation, plotting and batch scripts
results/       outputs: run directories, tables, figures, benchmark predictions          -> results/README.md
tests/         minimal runnable tests (synthetic data, tiny randomly initialised decoder)
```

| Module | Content | Paper |
|---|---|---|
| `cover/data.py` | availability patterns, dynamic masking, permanent missingness | Sec. 2.1, Sec. 4.1, Supp. E |
| `cover/model.py` | LLM-Fusion, counterfactual views, disagreement gate | Eqs. 1, 3, 4 |
| `cover/losses.py` | backbone objective, unimodal-judge loss, CoVeR objective | Eqs. 5, 6 |
| `cover/train.py`, `cover/recipes.py` | training loop and hyper-parameters | Sec. 4.1, Supp. A |
| `cover/evaluate.py`, `cover/metrics.py` | seven patterns, missing rates, accuracy / robust mode, metrics | Sec. 4.1, Sec. 4.3 |
| `cover/experts.py`, `cover/mrbench_a.py` | unimodal experts, case typing, Cons. / Conf. / ShortErr | Sec. 2.1, Sec. 2.2, Supp. B |
| `cover/mrbench_h.py` | fusion-model predictions and scoring on the 900 clips | Sec. 2.3, Supp. C |
| `cover/diagnosis.py` | prior-centred reliance, conflict-detection AUROC | Sec. 3.2, Eq. 2 |
| `cover/steering.py` | discrepancy-gated activation steering (inference-time comparator) | Sec. 3.2, Supp. C |
| `cover/learned_gate.py` | learned reliability gate (training-time comparator) | Sec. 4.3 |
| `cover/features.py` | feature extraction for IEMOCAP, MELD and MUStARD++ | Supp. A |

## Installation

```bash
conda create -n cover python=3.10 -y && conda activate cover
pip install -r requirements.txt        # PyTorch 2.4, transformers 4.57, peft 0.17
pip install -e .                       # optional: the scripts also run from a plain checkout
python -m unittest discover -s tests   # minimal tests, no download, CPU is enough
```

Training uses one 24 GB GPU (bf16, activation checkpointing). The LLM is
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B); pass a local copy with `--llm_path`.

## Data

**Corpora.** CMU-MOSI, CMU-MOSEI, CH-SIMS and CH-SIMSv2 with the standard MMSA features, and IEMOCAP (Session 5 test),
scored by its averaged valence, with BERT, openSMILE ComParE and ViT facial-expression features. Each modality is
pooled to T = 50 tokens. See [data/README.md](data/README.md) for sources, the expected layout and the feature recipe.

```bash
python scripts/preprocess.py --raw_dir data/raw --datasets mosi mosei sims simsv2      # -> data/processed/<dataset>.pt

# IEMOCAP, MELD and MUStARD++ have no MMSA features: extract them first (pip install -r requirements-features.txt)
python scripts/extract_features.py --source iemocap --root /path/to/IEMOCAP_full_release
python scripts/preprocess.py --datasets iemocap
```

**MR-Bench-A** types every sample of a corpus by its unimodal judgments (consensus / conflict / weak-modality conflict
/ other). The annotations of the four MMSA corpora are in `data/mrbench_a/` (derived annotations keyed by the original
sample identifiers); they are rebuilt with

```bash
python scripts/build_mrbench_a.py --dataset sims     # human unimodal labels (CH-SIMS, CH-SIMSv2)
python scripts/build_mrbench_a.py --dataset mosi     # single-modality experts, out-of-fold on the training split
```

**MR-Bench-H** (900 human-validated clips: 620 language-shortcut, 280 non-verbal-shortcut; seven sources) is released
as a separate dataset, [Bluexxx/MR-Bench-H](https://huggingface.co/datasets/Bluexxx/MR-Bench-H). The scripts below take
its item list `data/test.jsonl` through `--items`:

```bash
hf download Bluexxx/MR-Bench-H --repo-type dataset --local-dir MR-Bench-H
```

## Method

**LLM-Fusion.** A sample is serialised into `Z(c, p) = [P(p); S_t(c_t); S_a(c_a); S_v(c_v); E_fus]`: a prompt stating
which modalities are available, one block per modality (projected features between boundary tokens, or `T` copies of a
learnable `<miss_j>` embedding) and `K` fusion tokens. A LoRA-tuned decoder-only LLM produces the fusion-token states
and a mask-aware dual-path head decodes them into the score. The backbone is trained with
`L_L = λ_task ℓ(ŷ, y) + λ_con L_con + λ_reg L_reg`.

**Counterfactual views.** A view `V = (c, p)` specifies which blocks carry features and what the prompt claims. For a
sample with availability `m`: the full view `(m, m)`, the single-modality views `(e_j ⊙ m, e_j ⊙ m)` and the
content-free view `(0, m)`. All views share one token layout and run in one batched pass.

**CoVeR** changes only how the model is trained:

1. *Views as unimodal judges.* Every observed single-modality view is supervised with its unimodal label
   (`L_uni`, Eq. 5): human unimodal annotations on CH-SIMS / CH-SIMSv2, out-of-fold expert predictions elsewhere.
2. *Disagreement-gated supervision.* A non-text view confidently disagrees with the text view if their predictions
   have opposite polarity and the view's prediction is at least `τ/2` from the neutral score (Eq. 3). The gated
   prediction `ỹ` (Eq. 4) equals `ŷ` if no view disagrees and the mean of the disagreeing views' predictions (with
   stopped gradient) otherwise; the task loss is put on `ỹ` (Eq. 6), so the fused path receives no task gradient from
   a sample on which a judge disagrees, whereas all views keep their unimodal supervision.

**Inference.** *Accuracy mode* outputs `ŷ` of the full view in one forward pass, with the architecture and the
inference cost of LLM-Fusion (all main results). The optional *robust mode* applies Eqs. (3, 4) with a test-time margin `τ_test`. Applied
to a trained LLM-Fusion, the same rule is the training-free *test-time override*.

## Training and evaluation

```bash
# LLM-Fusion and CoVeR with the hyper-parameters selected on the validation split
python scripts/train.py --dataset mosi --method llm_fusion --seed 42
python scripts/train.py --dataset mosi --method cover --seed 42

# default recipe of the backbone (analyses): tau = 0.3, lambda_uni = 1
python scripts/train.py --dataset sims --method cover --recipe default --seed 42
python scripts/train.py --dataset sims --method cover --recipe default --no_gate --seed 42   # judges without the gate
```

A run writes `results/runs/<dataset>/<name>_s<seed>/` with `config.json`, `history.json`, `best.pt` (the checkpoint
with the best validation Avg Corr; one checkpoint serves every metric) and `results.json`:

* `accuracy` / `robust`: metrics per availability pattern and their average (`Avg`), under random missing rates
  (`Rate`), and the MR-Bench-A statistics with all modalities observed (`Cons`, `Conf`, `ShortErr`, conflict Acc2
  without text and by the visual-only view).

```bash
python scripts/evaluate.py --run results/runs/mosi/cover_s42                                  # re-evaluate a run
python scripts/evaluate.py --run results/runs/mosi/llm_fusion_default_s42 --tau_test 0.3 \
    --out results/runs/mosi/llm_fusion_default_s42/results_override.json                        # test-time override
python scripts/evaluate.py --run results/runs/mosi/llm_fusion_default_s42 --steering           # activation steering
python scripts/make_tables.py                                                                 # results/tables/*.md
```

### Checkpoints

The CoVeR checkpoints of seed 42 (selected recipe) for the seven source corpora of MR-Bench-H are released at
[Bluexxx/CoVeR](https://huggingface.co/Bluexxx/CoVeR): CMU-MOSI, CMU-MOSEI, CH-SIMS, CH-SIMSv2, IEMOCAP, MELD and
MUStARD++. Each run directory holds `config.json`, `best.pt` and `results.json` in the layout of `results/runs/`:

```bash
hf download Bluexxx/CoVeR --local-dir results/runs --include "*/cover_s42/*"
python scripts/evaluate.py --run results/runs/mosi/cover_s42            # seven patterns, missing rates, MR-Bench-A

# MR-Bench-H: one prediction file per source, then the pooled score (CoVeR in accuracy mode and robust mode)
for ds in mosi mosei sims simsv2 iemocap meld mustardpp; do
  python scripts/predict_mrbench_h.py --run results/runs/$ds/cover_s42 --items MR-Bench-H/data/test.jsonl
done
python scripts/score_mrbench_h.py --items MR-Bench-H/data/test.jsonl --pred "results/mrbench_h/cover_*_s42.jsonl"
```

The paper reports means over three seeds; these checkpoints are the runs of seed 42.

### Hyper-parameters

Shared by LLM-Fusion and CoVeR: Qwen3-0.6B with LoRA on the attention projections, `K = 4` fusion tokens,
`λ_task = 10`, `λ_con = λ_reg = 1`, AdamW with a cosine schedule, batch 32, bf16 and activation checkpointing,
and a default budget of 30 epochs (15 on CMU-MOSEI, 20 on IEMOCAP). Hyper-parameters are selected per dataset on the
validation split only (`τ ∈ {0.2, 0.3, 0.4}`, `λ_uni ∈ {0.5, 1, 2}`, rank `r ∈ {16, 32}`, then learning rate, epochs and
weight averaging); LLM-Fusion receives the same search. The selected values of both methods are in
[`src/cover/recipes.py`](src/cover/recipes.py):

| CoVeR, `--recipe selected` | CMU-MOSI | CMU-MOSEI | CH-SIMS | CH-SIMSv2 | IEMOCAP |
|---|---|---|---|---|---|
| margin `τ` (labels in [0, 1], `c = 0.5`) | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| `λ_uni` | 2 | 2 | 2 | 2 | 2 |
| LoRA rank | 16 | 16 | 32 | 16 | 16 |
| learning rate | 2e-4 | 2e-4 | 2e-4 | 2e-4 | 2e-4 |
| weight averaging (decay) | 0.99 | 0.99 | 0.99 | – | 0.99 |
| epochs | 30 | 15 | 30 | 30 | 20 |

| LLM-Fusion, `--recipe selected` | CMU-MOSI | CMU-MOSEI | CH-SIMS | CH-SIMSv2 | IEMOCAP |
|---|---|---|---|---|---|
| LoRA rank | 32 | 32 | 16 | 16 | 32 |
| learning rate | 2e-4 | 2e-4 | 1e-4 | 1e-4 | 2e-4 |
| weight averaging (decay) | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| epochs | 30 | 15 | 50 | 30 | 20 |

`--recipe default` is the default recipe of the backbone for every row of the analyses (rank 16, lr 2e-4, no weight
averaging; CoVeR: `τ = 0.3`, `λ_uni = 1`). Every hyper-parameter can be overridden on the command line
(`python scripts/train.py -h`).

## Evaluation protocol

* **Incomplete training.** A missing modality is unavailable both as an input and as a target. The main protocol draws
  one of the seven non-empty availability patterns, the complete one included, at each visit of a training sample.
  `--missing permanent` assigns every training and validation sample one pattern for the whole run and zeroes the
  features of its unobserved modalities when the data are loaded (`--missing permanent_incomplete`: the six incomplete
  patterns only).
* **Test.** Every availability pattern and random missing rates `r ∈ {0.1, …, 0.7}` on the complete test split.
  Acc2, weighted F1, MAE and Pearson correlation are averaged over the patterns (Avg) or the missing rates (Rate).
  Acc2 and F1 exclude neutral labels; averages are arithmetic means, except for the correlation, which is averaged
  through Fisher's z-transform.
* **MR-Bench-A.** With all modalities observed: Acc2 on consensus and conflict cases, and ShortErr, the fraction of
  conflict cases on which the model predicts the polarity of the text judgment.
* **MR-Bench-H.** Every model is scored on all 900 clips; an answer that is not a valid polarity counts as wrong.
  A fusion model reads features, so each clip is scored, with all modalities observed, by the model trained on the
  training split of its source corpus; results are the mean and standard deviation over three seeds of the accuracy
  pooled over the seven sources.
* **Seeds.** Means over three seeds for the comparisons with the state of the art and the diagnosis, over five seeds
  for the analyses of CoVeR.

## Reproducing the experiments

`bash scripts/run_paper.sh {main|analysis|permanent|tables}` runs the stages below for all datasets and seeds.

| Experiment | Commands |
|---|---|
| Comparison under incomplete modalities, per-pattern and missing-rate results (LLM-Fusion, CoVeR) | `train.py --method {llm_fusion,cover}` (selected recipe, seeds 42–44), `make_tables.py` |
| Correcting the collapse on MR-Bench-A: LLM-Fusion, discrepancy-gated steering, test-time override, learned reliability gate, conflict re-weighting, unimodal multi-task, CoVeR in accuracy and robust mode | default recipe, seeds 42–46: `train.py --method llm_fusion --recipe default`; `evaluate.py --steering`; `evaluate.py --tau_test 0.3`; `train.py --method learned_gate`; `train.py --method llm_fusion --recipe default --conflict_weight 5`; `train.py --method cover --recipe default --no_gate`; `train.py --method cover --recipe default` |
| Supervision ablation on CH-SIMS | `train.py --dataset sims --method cover --recipe default --no_gate` (unimodal loss, no gate) and `train.py --dataset sims --method cover --recipe default` (CoVeR), seeds 42–46; the LLM-Fusion row is the three-seed reference of the comparison under incomplete modalities |
| Test-time override sweep | `evaluate.py --run <LLM-Fusion run> --tau_test {0.5, 0.4, 0.3, 0.2, 0.1}` |
| Scaling the LLM backbone | `train.py --dataset sims --method llm_fusion --recipe default --llm_path <LLM>` |
| Where the decision collapses (reliance per layer, AUROC) | `diagnose.py --run <run>`, `plot_reliance.py` |
| Permanent missingness | `train.py --missing permanent` (`permanent_incomplete`); expert judgments from the visible samples: `build_mrbench_a.py --missing permanent` |
| IEMOCAP; MELD and MUStARD++ (sources of MR-Bench-H) | `extract_features.py --source {iemocap,meld,mustardpp}`, `preprocess.py`, `build_mrbench_a.py`, then `train.py` as above |
| MR-Bench-H, fusion models | `predict_mrbench_h.py --run <run> --items <test.jsonl>` per source and seed (`--steering` for the state-steering comparator), then `score_mrbench_h.py` |
| MR-Bench-H, zero-shot MLLMs | `eval_mllm.py --model <name> --model_path <path> --items <test.jsonl> --clips <clips>`, then `score_mrbench_h.py` |

This repository contains the implementation of our method and analyses. The nine incomplete-multimodal-learning
methods compared in the paper are ported from the authors' code and run under the same protocol (same frozen features,
availability patterns and tests); these ports are not part of this repository.

## Citation

```bibtex
@article{li2026words,
  title  = {Words Speak Louder? Exploring the Language Shortcut of LLM-Driven Incomplete Multimodal Learning},
  author = {Li, Wenhao and Xiao, Chong and Bai, Xiaokun and He, Rundong and Qi, Zhuang and Liang, Xusheng and Wu, Zhibin and Wang, Qiangchang},
  year   = {2026}
}
```

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The MR-Bench-A annotations in `data/mrbench_a/`
are derived annotations keyed by the original sample identifiers and inherit the licences of their source corpora.
The diagnosis tools are intended for auditing models; we do not recommend deploying affect recognition in high-stakes
decisions about individuals.
