"""CoVeR: Counterfactual-View Reliability learning for LLM-driven incomplete multimodal learning.

Modules
-------
data         availability patterns, dynamic masking and permanent missingness (Sec. 2.1, Sec. 4.1)
model        LLM-Fusion backbone, counterfactual views and the disagreement gate (Eqs. 1, 3, 4)
losses       backbone objective L_L, unimodal-judge loss L_uni and the CoVeR objective (Eqs. 5, 6)
train        training loop shared by LLM-Fusion and CoVeR
evaluate     evaluation under the seven availability patterns and random missing rates
recipes      hyper-parameters of the paper
experts      single-modality BiLSTM experts (unimodal judgments on corpora without human unimodal labels)
mrbench_a    MR-Bench-A: case typing and its metrics (consensus / conflict Acc2, ShortErr)
mrbench_h    MR-Bench-H: item-level predictions of fusion models and scoring on the 900 clips
steering     discrepancy-gated activation steering (inference-time comparator)
learned_gate learned reliability gate (training-time comparator)
diagnosis    prior-centred reliance (Eq. 2) and conflict-detection AUROC
features     feature extraction for IEMOCAP, MELD and MUStARD++
"""

__version__ = "1.0.0"
