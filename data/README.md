# Data

No source data are distributed with this repository. Obtain every corpus from its distributor and under its terms;
do not commit raw data, features or processed tensors (`data/raw/` and `data/processed/` are git-ignored).

```
data/
  raw/            feature pickles of the corpora (you provide them)
  processed/      <dataset>.pt written by scripts/preprocess.py
  mrbench_a/      MR-Bench-A annotations of CMU-MOSI, CMU-MOSEI, CH-SIMS and CH-SIMSv2 (distributed)
  mrbench_h_holdout/         identifiers of the MR-Bench-H candidates held out of training (IEMOCAP, MUStARD++)
  iemocap_female_side.json   side of the female actor in every IEMOCAP dialogue (speaker's face)
```

## Corpora and features

| Corpus | `--dataset` | Train / valid / test | Label scale | Unimodal judgments | Feature pickle under `data/raw/` |
|---|---|---|---|---|---|
| CMU-MOSI | `mosi` | 1,284 / 229 / 686 | [-3, 3] | experts | `MOSI/aligned_50.pkl` |
| CMU-MOSEI | `mosei` | 16,326 / 1,871 / 4,659 | [-3, 3] | experts | `MOSEI/aligned_50.pkl` |
| CH-SIMS | `sims` | 1,368 / 456 / 457 | [-1, 1] | human | `SIMS/unaligned_39.pkl` |
| CH-SIMSv2 | `simsv2` | 2,722 / 647 / 1,034 | [-1, 1] | human | `SIMSv2/sims_unaligned.pkl` |
| IEMOCAP | `iemocap` | 6,740 / 617 / 2,170 | [-2, 2] | experts | `IEMOCAP/iemocap_50.pkl` |
| MELD | `meld` | see below | [-1, 1] | experts | `MELD/meld_50.pkl` |
| MUStARD++ | `mustardpp` | see below | [-1, 1] | experts | `MUSTARDPP/mustardpp_50.pkl` |

**CMU-MOSI, CMU-MOSEI, CH-SIMS, CH-SIMSv2** use the standard MMSA features and splits (the processed pickles of the
[MMSA](https://github.com/thuiar/MMSA) distribution).

**IEMOCAP, MELD and MUStARD++** have no MMSA features. `scripts/extract_features.py` writes pickles in the same
format (`pip install -r requirements-features.txt`; ffmpeg is needed for MELD and MUStARD++):

```bash
python scripts/extract_features.py --source iemocap --root /path/to/IEMOCAP_full_release   # folder with Session1..5
python scripts/extract_features.py --source meld --root /path/to/MELD                      # *_sent_emo.csv, MELD.Raw/
python scripts/extract_features.py --source mustardpp --root /path/to/MUStARD++            # csv and utterance videos
python scripts/preprocess.py --datasets iemocap meld mustardpp
```

The features are:

* text: BERT-base last hidden states of the transcript (at most 50 word pieces, the MMSA recipe);
* audio: openSMILE ComParE_2016 low-level descriptors (65-d, 100 fps) of the utterance;
* visual: the CLS state and the 7 expression logits of a ViT fine-tuned on FER2013 (775-d) on the speaker's face,
  detected by MTCNN at 5 fps. On MELD and MUStARD++ the clips do not identify the speaker, so the largest confident
  face of each frame is embedded. IEMOCAP shows both actors in frame; the female actor's side is read off five frames
  per dialogue by Qwen2.5-VL-7B (asked twice, on the frame and its mirror image; `scripts/iemocap_sides.py`, result in
  `iemocap_female_side.json`), and the utterance id gives the speaker's gender.

Scores and splits:

* IEMOCAP: the score is the averaged valence (1–5) minus 3, so polarity is its sign and neutral utterances are excluded
  from Acc2 / F1. Sessions 1–4 form the training set with 10% of their dialogues as validation, Session 5 is the test
  set, and the 512 MR-Bench-H candidates of Sessions 1–4 (`mrbench_h_holdout/iemocap.txt`) are held out of training
  (`pool` split).
* MELD: the official training split minus 10% of its dialogues (validation) is used for training, with the sentiment
  label in {-1, 0, 1} as the score; all MR-Bench-H clips come from the official development (`pool`) and test splits.
* MUStARD++: every MR-Bench-H candidate (`mrbench_h_holdout/mustardpp.txt`) is held out (`pool`) and the remaining
  utterances are split 80/10/10, with the score (valence − 5) / 4.

### Pickle format

A pickle is a dict with the splits `train`, `valid`, `test` (and optionally `pool`); every split is a dict with

| Key | Shape | Content |
|---|---|---|
| `text` | [N, L, 768] | BERT features |
| `text_bert` | [N, 3, L] | BERT ids / attention mask / type ids (row 1 gives the valid text length); optional |
| `audio`, `vision` | [N, L, d] | audio / visual features |
| `audio_lengths`, `vision_lengths` | [N] | valid lengths; optional |
| `regression_labels` | [N] | sentiment score |
| `regression_labels_T`, `_A`, `_V` | [N] | human unimodal labels (CH-SIMS, CH-SIMSv2) |
| `id` | [N] | sample identifiers |

`scripts/preprocess.py` resamples each modality to T = 50 steps by adaptive average pooling over its valid length and
z-scores the audio and visual features with training-split statistics.

## MR-Bench-A (`data/mrbench_a/`)

`<dataset>_<split>.jsonl` for the four MMSA corpora, one object per sample:

| Field | Meaning |
|---|---|
| `id` | sample identifier, identical to the MMSA pickles |
| `case_type` | `consensus`, `conflict`, `weak_conflict` or `other` |
| `label_polarity` | sign of the label (-1 / 0 / +1) |
| `polarity` | sign of the text / audio / visual judgment |
| `shortcut_polarity` | polarity of the text judgment: the answer of a model that follows the words |
| `unimodal` | expert judgments on the label scale (CMU-MOSI, CMU-MOSEI) |

Judgments: the human unimodal labels on CH-SIMS / CH-SIMSv2 (read from the corpus, not copied here) and, on CMU-MOSI /
CMU-MOSEI, single-modality BiLSTM experts (three seeds; validation and test from experts trained on the training
split, training split out-of-fold). A judgment is confident if its magnitude is at least κ (0.2 on CH-SIMS / CH-SIMSv2,
0.5 on CMU-MOSI / CMU-MOSEI). Among samples with a non-neutral label:

* `consensus`: all judgments share the polarity of the label;
* `conflict`: the text judgment is confidently wrong and an audio or visual judgment is confidently right;
* `weak_conflict`: the text judgment is right and an audio or visual judgment is confidently wrong;
* `other`: otherwise.

Counts per split (train / validation / test), also in `stats.json`:

| Dataset | Samples | Consensus | Conflict | Weak |
|---|---|---|---|---|
| CMU-MOSI | 1284 / 229 / 686 | 428 / 92 / 187 | 55 / 9 / 36 | 297 / 55 / 186 |
| CMU-MOSEI | 16326 / 1871 / 4659 | 6387 / 675 / 1669 | 68 / 4 / 20 | 675 / 83 / 265 |
| CH-SIMS | 1368 / 456 / 457 | 621 / 211 / 221 | 105 / 37 / 31 | 99 / 38 / 32 |
| CH-SIMSv2 | 2722 / 647 / 1034 | 1423 / 326 / 457 | 148 / 55 / 73 | 146 / 42 / 54 |

The expert judgments of the training split are also the unimodal labels of CoVeR's judges on CMU-MOSI / CMU-MOSEI.
Any model can be scored with `python scripts/score_mrbench_a.py --dataset <dataset> --pred <predictions.jsonl>`.
For the other corpora, build the annotations with `python scripts/build_mrbench_a.py --dataset <dataset>`.

MR-Bench-A adds derived annotations keyed by the original sample identifiers and inherits the licences and annotation
biases of its sources.

## MR-Bench-H

MR-Bench-H is distributed as a separate dataset, [Bluexxx/MR-Bench-H](https://huggingface.co/datasets/Bluexxx/MR-Bench-H)
(annotations, permitted transcripts, clip references and tools to rebuild the clips from the sources). Scripts of this repository read its item list `data/test.jsonl`
(`--items`) and, for zero-shot MLLMs, the rebuilt clips `<uid>.mp4` (`--clips`).
