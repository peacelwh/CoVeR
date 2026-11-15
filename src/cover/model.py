"""LLM-Fusion with counterfactual views and the disagreement gate (Secs. 2.1, 4.2 and 4.3 of the paper).

Input sequence (Eq. 1), identical for every view so that the views are aligned token by token::

    Z(c, p) = [ P(p) ; S_t(c_t) ; S_a(c_a) ; S_v(c_v) ; E_fus ]

``P(p)`` is a prompt stating which modalities are available, ``S_j(1) = [<j_s>, phi_j(X^j), <j_e>]`` is a projected
feature block with boundary tokens, ``S_j(0)`` replaces the features by T copies of a learnable ``<miss_j>`` embedding
and ``E_fus`` are K fusion tokens. A LoRA-tuned decoder-only LLM produces the fusion-token states and a mask-aware
dual-path head decodes the last-layer states into the score.

A *view* V = (c, p) specifies which blocks carry features (content mask c) and what the prompt claims (prompt mask p).
For a sample with availability m:

    full view            V_f = (m, m)
    single-modality view V_j = (e_j * m, e_j * m)   for j in {t, a, v}
    content-free view    V_0 = (0, m)               (keeps the prompt, hides all content)

All views of a mini-batch run in one batched pass. LLM-Fusion and accuracy-mode inference use the full view alone;
CoVeR training, the test-time override and robust mode add the three single-modality views (4B sequences).
"""
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cover.data import MODALITIES, MODALITY_NAMES

VIEWS = ("full", "t", "a", "v", "cf")
JUDGES = ("t", "a", "v")
NEUTRAL = 0.5  # neutral point c of the normalised label scale [0, 1]

PROMPT = ("Given the available {avail} features from a video clip. The {miss} modality is missing. "
          "Based on the available modalities, please infer and reconstruct the useful latent representations "
          "for the missing {miss} modalities at the designated positions. Then integrate and enhance all "
          "multimodal features for multimodal sentiment analysis. Output the fused multi-dimensional feature "
          "representations at the designated feature dimension positions: ")


def disagreement_gate(y_views: Dict[str, torch.Tensor], mask: torch.Tensor, tau: float) -> torch.Tensor:
    """Eq. (3) with budgeted weights: which non-anchor views confidently disagree with the anchor (text) view.

    y_views: predictions of the single-modality views in [0, 1] (the text view is content-free when m_t = 0);
    mask: availability [B, 3]; tau: margin on the normalised label scale. Returns the weights w_j [2, B] for
    j in (a, v), with w_j = m_j g_j / max(1, sum_k m_k g_k).
    """
    y_t = y_views["t"]
    g = []
    for j, m in enumerate(MODALITIES):
        if m == "t":
            continue
        y_j = y_views[m]
        fire = ((y_j - NEUTRAL) * (y_t - NEUTRAL) < 0) & ((y_j - NEUTRAL).abs() >= tau / 2)
        g.append(fire.float() * mask[:, j])
    g = torch.stack(g, 0)
    return g / g.sum(0, keepdim=True).clamp(min=1.0)


class LLMFusion(nn.Module):
    """LLM-driven fusion model. ``llm`` / ``tokenizer`` can be passed directly (unit tests); otherwise they are
    loaded from ``llm_path``."""

    def __init__(self, llm_path: Optional[str], dims: Dict[str, int], T: int = 50, num_fusion_tokens: int = 4,
                 lora_r: int = 16, lora_alpha: Optional[int] = None, lora_dropout: float = 0.1, dropout: float = 0.15,
                 grad_checkpointing: bool = False, attn_implementation: str = "sdpa",
                 llm_dtype: torch.dtype = torch.bfloat16, llm: Optional[nn.Module] = None, tokenizer=None):
        super().__init__()
        self.T, self.K, self.dims = T, num_fusion_tokens, dict(dims)

        # ---------------- decoder-only LLM (no LM head), LoRA on the attention projections ----------------
        if llm is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(llm_path)
            llm = AutoModel.from_pretrained(llm_path, dtype=llm_dtype, attn_implementation=attn_implementation)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.D = llm.config.hidden_size
        self.num_layers = llm.config.num_hidden_layers
        for p in llm.parameters():
            p.requires_grad = False
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            # LoRA with alpha = 2r
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=lora_dropout,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
            self.llm = get_peft_model(llm, cfg)
        else:
            self.llm = llm
        self.embed = self.decoder().get_input_embeddings()
        if grad_checkpointing:  # activation checkpointing of the decoder layers
            self.decoder().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.decoder().config.use_cache = False

        # ---------------- trainable special tokens ----------------
        # 0..5: <j_s> / <j_e> for j = t, a, v ; 6..8: <miss_j> ; 9..9+K-1: fusion tokens
        self.special = nn.Embedding(6 + 3 + self.K, self.D)
        nn.init.normal_(self.special.weight, std=0.02)
        self.idx_start = {m: 2 * i for i, m in enumerate(MODALITIES)}
        self.idx_end = {m: 2 * i + 1 for i, m in enumerate(MODALITIES)}
        self.idx_missing = {m: 6 + i for i, m in enumerate(MODALITIES)}
        self.idx_fusion = list(range(9, 9 + self.K))

        # ---------------- modality projections phi_j: two 1x1 convolutions ----------------
        def projection(d_in):
            return nn.Sequential(
                nn.Conv1d(d_in, self.D, 1), nn.BatchNorm1d(self.D), nn.ReLU(True), nn.Dropout(dropout),
                nn.Conv1d(self.D, self.D, 1), nn.BatchNorm1d(self.D))
        self.proj = nn.ModuleDict({m: projection(dims[m]) for m in MODALITIES})

        # ---------------- mask-aware dual-path head ----------------
        # main path: fusion tokens; auxiliary path: modality blocks weighted by availability
        self.LN = nn.LayerNorm(self.D)
        self.token_role_weight = nn.Parameter(torch.ones(self.K))
        self.cond_head = nn.Sequential(nn.Linear(self.D + 3, self.D), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.D, self.D))
        self.cond_gate = nn.Sequential(nn.Linear(self.D + 3, self.D), nn.Sigmoid())
        self.regressor = nn.Sequential(nn.Linear(self.D, self.D // 2), nn.LayerNorm(self.D // 2), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(self.D // 2, 1))
        self.modal_norms = nn.ModuleDict({m: nn.LayerNorm(self.D) for m in MODALITIES})
        self.cross_modal_attn = nn.MultiheadAttention(self.D, 4, dropout=dropout, batch_first=True)
        self.modal_gate = nn.Sequential(nn.Linear(self.D * 3, self.D), nn.GELU(), nn.Linear(self.D, 3), nn.Softmax(-1))
        self.missing_confidence = nn.Parameter(torch.tensor([0.6, 0.6, 0.6]))
        self.aux_regressor = nn.Sequential(nn.Linear(self.D, self.D // 2), nn.LayerNorm(self.D // 2), nn.GELU(),
                                           nn.Dropout(dropout), nn.Linear(self.D // 2, 1))
        self.w1 = nn.Parameter(torch.tensor([0.7]))
        self._prompt_cache = {}

    # ------------------------------------------------------------------ helpers
    def decoder(self):
        """The decoder-only LLM without the LoRA wrapper."""
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
            return self.llm.base_model.model
        return self.llm

    def decoder_layers(self):
        return self.decoder().layers

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def prompt_text(mask3: Sequence[int]) -> str:
        avail = [MODALITY_NAMES[m] for m, b in zip(MODALITIES, mask3) if b]
        miss = [MODALITY_NAMES[m] for m, b in zip(MODALITIES, mask3) if not b]
        return PROMPT.format(avail=", ".join(avail) if avail else "no available modality",
                             miss=", ".join(miss) if miss else "none")

    def _prompt_ids(self, device):
        """The eight prompt variants (one per availability pattern, the empty one included), tokenised once and
        left-padded to a common length, so that every view has an identical token layout."""
        if "ids" not in self._prompt_cache:
            texts = [self.prompt_text([(k >> 2) & 1, (k >> 1) & 1, k & 1]) for k in range(8)]
            enc = self.tokenizer(texts, padding=True, return_tensors="pt", padding_side="left")
            self._prompt_cache["ids"] = enc["input_ids"]
            self._prompt_cache["attn"] = enc["attention_mask"]
        return self._prompt_cache["ids"].to(device), self._prompt_cache["attn"].to(device)

    # ------------------------------------------------------------------ views
    def project(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """feats: m -> [B, T, d_m]. Returns the projected blocks [B, 3, T, D]."""
        return torch.stack([self.proj[m](feats[m].transpose(1, 2)).transpose(1, 2) for m in MODALITIES], 1)

    @staticmethod
    def view_masks(mask: torch.Tensor, name: str):
        """(content mask c, prompt mask p) of view ``name`` for the availability ``mask`` [B, 3]."""
        if name == "full":
            return mask, mask
        if name == "cf":
            return torch.zeros_like(mask), mask
        e = torch.zeros_like(mask)
        j = MODALITIES.index(name)
        e[:, j] = mask[:, j]
        return e, e

    def assemble(self, feat: torch.Tensor, mask: torch.Tensor, names: Sequence[str]):
        """Builds Z(c, p) for every view in ``names``. feat: [B, 3, T, D]. Returns the input embeddings [V*B, S, D]
        (views stacked along the batch axis), the attention mask, the position ids and the token layout."""
        device, B = feat.device, feat.shape[0]
        pairs = [self.view_masks(mask, n) for n in names]
        C = torch.cat([c for c, _ in pairs], 0)
        P = torch.cat([p for _, p in pairs], 0)
        R = C.shape[0]
        feat_rep = feat.repeat(len(names), 1, 1, 1)
        miss_emb = torch.stack([self.special.weight[self.idx_missing[m]] for m in MODALITIES], 0)  # [3, D]
        cm = C[:, :, None, None]
        content = cm * feat_rep + (1 - cm) * miss_emb[None, :, None, :]

        p_ids, p_attn = self._prompt_ids(device)
        keys = (P[:, 0] * 4 + P[:, 1] * 2 + P[:, 2]).long()
        prompt_emb = self.embed(p_ids[keys]).float()
        prompt_attn = p_attn[keys].float()
        Lp = prompt_emb.shape[1]

        parts, attns = [prompt_emb], [prompt_attn]
        ones1 = torch.ones(R, 1, device=device)
        onesT = torch.ones(R, self.T, device=device)
        layout = {"prompt": (0, Lp)}
        cur = Lp
        for j, m in enumerate(MODALITIES):
            parts.append(self.special.weight[self.idx_start[m]].expand(R, 1, -1))
            attns.append(ones1)
            cur += 1
            layout[m] = (cur, cur + self.T)
            parts.append(content[:, j])
            attns.append(onesT)
            cur += self.T
            parts.append(self.special.weight[self.idx_end[m]].expand(R, 1, -1))
            attns.append(ones1)
            cur += 1
        layout["fusion"] = (cur, cur + self.K)
        parts.append(self.special.weight[self.idx_fusion].unsqueeze(0).expand(R, -1, -1))
        attns.append(torch.ones(R, self.K, device=device))
        embeds = torch.cat(parts, 1)
        attn = torch.cat(attns, 1)
        pos_ids = (attn.cumsum(-1) - 1).clamp(min=0).long()
        layout.update({"views": list(names), "B": B, "C": C})
        return embeds, attn, pos_ids, layout

    def run_views(self, feats, mask, names: Sequence[str], return_hidden: bool = False, feat: Optional[torch.Tensor] = None):
        """One batched pass of the LLM over the views ``names``. Returns (LLM output, layout). ``feat``: projected
        blocks, if already computed."""
        embeds, attn, pos_ids, layout = self.assemble(self.project(feats) if feat is None else feat, mask, names)
        out = self.llm(inputs_embeds=embeds.to(self.embed.weight.dtype), attention_mask=attn, position_ids=pos_ids,
                       output_hidden_states=return_hidden)
        return out, layout

    # ------------------------------------------------------------------ head
    def head(self, hidden: torch.Tensor, layout, adjust_readout=None) -> Dict[str, torch.Tensor]:
        """Mask-aware dual-path head on the last-layer states [R, S, D]; predictions are in [0, 1].
        ``adjust_readout``: optional function applied to the fusion-token readout before the main path."""
        fs, fe = layout["fusion"]
        C = layout["C"]
        fusion = hidden[:, fs:fe, :].float()  # [R, K, D]
        w = F.softmax(self.token_role_weight, 0)
        z = readout = self.LN((fusion * w[None, :, None]).sum(1))
        if adjust_readout is not None:
            z = adjust_readout(z)
        cond_in = torch.cat([z, C], -1)
        z = z + self.cond_gate(cond_in) * self.cond_head(cond_in)
        y_main = torch.sigmoid(self.regressor(z).squeeze(-1))

        blocks = []
        for m in MODALITIES:
            s, e = layout[m]
            blocks.append(self.modal_norms[m](hidden[:, s:e, :].float().mean(1)))
        seq = torch.stack(blocks, 1)
        attn_out, _ = self.cross_modal_attn(seq, seq, seq)
        wgt = self.modal_gate(torch.cat(blocks, -1))
        conf = C + (1 - C) * torch.sigmoid(self.missing_confidence)[None, :]
        z_aux = (attn_out * wgt[..., None] * conf[..., None]).sum(1)
        y_aux = torch.sigmoid(self.aux_regressor(z_aux).squeeze(-1))
        y = self.w1 * y_main + (1 - self.w1) * y_aux
        return {"y": y, "y_main": y_main, "y_aux": y_aux, "fusion": fusion, "readout": readout}

    # ------------------------------------------------------------------ forward
    def forward(self, feats: Dict[str, torch.Tensor], mask: torch.Tensor, views: Optional[str] = None,
                tau: Optional[float] = None, gate_scale: float = 1.0, return_hidden: bool = False):
        """feats: m -> [B, T, d_m]; mask: availability [B, 3].

        views: ``None``      the full view alone (LLM-Fusion; accuracy mode of CoVeR)
               ``"judges"``  full view + the three single-modality views, processed jointly (4B sequences)
               ``"all"``     ``"judges"`` + the content-free view (diagnosis)
        tau:   margin of the disagreement gate; if given (with views), ``y_tilde`` is the gated prediction of
               Eq. (4), otherwise ``y_tilde`` is the full-view prediction ``y``.

        Returns a dict with ``y`` (full view), ``y_tilde``, ``y_main`` / ``y_aux`` (the two decoding paths of the
        full view), ``fusion`` (fusion-token states of the full view) and, with views, ``y_views`` (j -> prediction
        of the single-modality view), ``view_valid`` ([3, B]: modality j is observed) and ``gate_weights``.
        """
        names: List[str] = ["full"]
        if views in ("judges", "all"):
            names += list(JUDGES)
        if views == "all":
            names.append("cf")
        out, layout = self.run_views(feats, mask, names, return_hidden)
        h = self.head(out.last_hidden_state, layout)
        B, V = mask.shape[0], len(names)
        res = {k: h[k].view(V, B, *h[k].shape[1:])[0] for k in ("y", "y_main", "y_aux", "fusion")}
        res["layout"] = layout
        res["y_tilde"] = res["y"]
        if V > 1:
            y_all = h["y"].view(V, B)
            res["y_views"] = {m: y_all[1 + j] for j, m in enumerate(JUDGES)}
            res["view_valid"] = mask.t()
            if "cf" in names:
                res["y_cf"] = y_all[names.index("cf")]
            if tau is not None:
                # Eq. (4): y_tilde = y + sum_j w_j (sg[y^(j)] - y); if no view disagrees, y_tilde = y, otherwise
                # y_tilde is the mean of the disagreeing views' predictions.
                y_sg = {m: y.detach().float() for m, y in res["y_views"].items()}
                w = disagreement_gate(y_sg, mask, tau)
                mix = sum(gate_scale * w[i] * (y_sg[m] - res["y"]) for i, m in enumerate(("a", "v")))
                res["y_tilde"] = res["y"] + mix
                res["gate_weights"] = w
        if return_hidden:
            res["hidden_states"] = out.hidden_states
        return res
