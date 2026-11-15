"""Feature extraction for IEMOCAP, MELD and MUStARD++ (supplementary Sec. A, "Features extracted by us").

These corpora have no MMSA features, so we extract them in the layout of the MMSA pickles:

    text    BERT-base last hidden states of the transcript (at most 50 word pieces, the MMSA recipe)   [N, 50, 768]
    audio   openSMILE ComParE_2016 low-level descriptors (65-d, 100 fps) of the utterance              [N, 50, 65]
    visual  the CLS state and the 7 expression logits of a ViT fine-tuned on FER2013 (775-d) on the
            speaker's face, detected by MTCNN at 5 fps                                                  [N, 50, 775]

Every stream is pooled to T = 50.

IEMOCAP shows both actors in frame. The actors keep their positions for a whole dialogue and the utterance id gives
the speaker's gender, so the side of the female actor (``data/iemocap_female_side.json``, read off five frames per
dialogue by a vision-language model, see scripts/iemocap_sides.py) gives the speaker's face of every utterance. The
score is the averaged valence (1-5) minus 3. Sessions 1-4 form the training set with 10% of their dialogues as
validation, Session 5 is the test set, and the MR-Bench-H candidates of Sessions 1-4 are held out of training.

MELD and MUStARD++ clips do not identify the speaker, so the largest confident face of each frame is embedded.
MELD: the official training split minus 10% of its dialogues (validation), with the sentiment label in {-1, 0, 1} as
the score; the official development split is held out. MUStARD++: every MR-Bench-H candidate is held out and the
remaining utterances are split 80/10/10, with the score (valence - 5) / 4.

Held-out clips are stored in the ``pool`` split and never used for training or validation.
"""
import csv
import glob
import os
import pickle
import re
import subprocess
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F

T = 50
VIS_DIM = 768 + 7
OUT = {"iemocap": "IEMOCAP/iemocap_50.pkl", "meld": "MELD/meld_50.pkl", "mustardpp": "MUSTARDPP/mustardpp_50.pkl"}


def pool_T(x, T=T):
    """[L, d] -> [T, d] by adaptive average pooling over L."""
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if x.shape[0] == T:
        return x
    return F.adaptive_avg_pool1d(x.T.unsqueeze(0), T)[0].T


# ------------------------------------------------------------------------------------------------------------ text
@torch.no_grad()
def text_features(texts, device, bert="bert-base-uncased", max_len=50, bs=128):
    """BERT last hidden states [N, max_len, 768] and (ids, attention mask, type ids) [N, 3, max_len]."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bert)
    model = AutoModel.from_pretrained(bert).to(device).eval()
    feats, ids = [], []
    for i in range(0, len(texts), bs):
        chunk = [t if t.strip() else "." for t in texts[i:i + bs]]
        enc = tok(chunk, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        h = model(**{k: v.to(device) for k, v in enc.items()}).last_hidden_state.float().cpu()
        feats.append(h.half())
        ids.append(torch.stack([enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]], 1))
    return torch.cat(feats).numpy(), torch.cat(ids).numpy().astype(np.int64)


# ----------------------------------------------------------------------------------------------------------- audio
_SMILE = None


def audio_features(wav):
    """ComParE_2016 low-level descriptors of a wav file pooled to T steps: ([T, 65], number of frames) or (None, 0)."""
    global _SMILE
    import opensmile
    if _SMILE is None:
        _SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                 feature_level=opensmile.FeatureLevel.LowLevelDescriptors, num_workers=1)
    try:
        x = _SMILE.process_file(wav).to_numpy(dtype=np.float32)
    except Exception:
        return None, 0
    if x.shape[0] == 0:
        return None, 0
    return pool_T(x).numpy(), int(x.shape[0])


def clip_audio_features(job):
    """(video, wav): extract the 16 kHz mono soundtrack of a clip with ffmpeg (once), then its descriptors."""
    video, wav = job
    if not os.path.exists(wav):
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        r = subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1",
                            "-ar", "16000", wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0 or not os.path.exists(wav):
            return None, 0
    return audio_features(wav)


# ---------------------------------------------------------------------------------------------------------- visual
class FaceEmbedder:
    """MTCNN face detection and the expression embedding (CLS state + 7 logits) of a ViT fine-tuned on FER2013."""

    def __init__(self, model_dir, device):
        from facenet_pytorch import MTCNN
        from transformers import AutoImageProcessor, ViTForImageClassification
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device, select_largest=False, post_process=False, min_face_size=30)
        proc = AutoImageProcessor.from_pretrained(model_dir)
        size = proc.size
        self.size = size["height"] if isinstance(size, dict) and "height" in size else (size["shortest_edge"] if isinstance(size, dict) else size)
        self.mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
        self.vit = ViTForImageClassification.from_pretrained(model_dir, torch_dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def detect(self, frames):
        return self.mtcnn.detect(frames)  # boxes: list of [k, 4] or None; probabilities: list of [k] or None

    @torch.no_grad()
    def embed(self, crops):
        import cv2
        x = torch.from_numpy(np.stack([cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_LINEAR) for c in crops]))
        x = x.to(self.device).permute(0, 3, 1, 2).float().div(255).sub(self.mean).div(self.std).half()
        out = []
        for i in range(0, len(x), 256):
            cls = self.vit.vit(pixel_values=x[i:i + 256]).last_hidden_state[:, 0]
            out.append(torch.cat([cls, self.vit.classifier(cls)], -1).float().cpu())
        return torch.cat(out).numpy()


def crop_face(frame, box, margin=0.25):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - margin * w)), int(min(W, x2 + margin * w))
    y1, y2 = int(max(0, y1 - margin * h)), int(min(H, y2 + margin * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def _gray_crop(gray, box, size):
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    H, W = gray.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.resize(gray[y1:y2, x1:x2], size, interpolation=cv2.INTER_AREA)


def clip_face_features(video, emb, fps_v=5.0, min_prob=0.9, det_scale=0.5, max_frames=150, chunk=32):
    """Expression embeddings [n, 775] of the largest confident face in the frames sampled at ``fps_v`` (None if no
    face is found). Used for clips that do not identify the speaker (MELD, MUStARD++)."""
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / fps_v)))
    frames, i = [], 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not frames:
        return None
    if len(frames) > max_frames:
        frames = [frames[j] for j in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    H, W = frames[0].shape[:2]
    crops = []
    for s in range(0, len(frames), chunk):
        part = frames[s:s + chunk]
        small = [cv2.resize(f, (int(W * det_scale), int(H * det_scale)), interpolation=cv2.INTER_AREA) for f in part]
        boxes, probs = emb.detect(small)
        for f, bx, pr in zip(part, boxes, probs):
            if bx is None:
                continue
            faces = [np.asarray(b) / det_scale for p, b in zip(pr, bx) if p is not None and p >= min_prob]
            if not faces:
                continue
            c = crop_face(f, max(faces, key=lambda z: (z[2] - z[0]) * (z[3] - z[1])))
            if c is not None:
                crops.append(c)
    return emb.embed(crops) if crops else None


def dialogue_face_features(video, utts, emb, fps_v=5.0, chunk=48, min_prob=0.9, det_every=5, det_scale=0.5):
    """IEMOCAP dialogue video with both actors in frame. utts: [(uid, start, end)].

    Frames are sampled at ``fps_v`` within every utterance; faces are detected on every ``det_every``-th sampled
    frame at ``det_scale`` resolution and carried forward in between (the actors keep their positions). Returns, per
    utterance and side ('L' / 'R' by x-position), the expression embeddings [n, 775] and grey face / mouth crops
    (lip activity)."""
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    need = defaultdict(list)
    for uid, s, e in utts:
        if e - s < 0.4:
            c = (s + e) / 2
            s, e = c - 0.2, c + 0.2
        for t in np.arange(s, e, 1.0 / fps_v):
            need[min(max(int(round(t * fps)), 0), max(n_frames - 1, 0))].append(uid)
    keys = ("L", "R", "L_face", "R_face", "L_mouth", "R_mouth")
    res = {uid: {k: [] for k in keys} for uid, _, _ in utts}
    buf, buf_fi = [], []
    state = {"cur": 0, "count": 0, "last_faces": []}

    def flush():
        if not buf:
            return
        det_pos = [i for i in range(len(buf)) if (state["count"] + i) % det_every == 0 or not state["last_faces"] and i == 0]
        det = {}
        if det_pos:
            small = [cv2.resize(buf[i], (int(W * det_scale), int(H * det_scale)), interpolation=cv2.INTER_AREA) for i in det_pos]
            boxes, probs = emb.detect(small)
            for i, bx, pr in zip(det_pos, boxes, probs):
                faces = [] if bx is None else [(float(p), np.asarray(b) / det_scale) for p, b in zip(pr, bx) if p is not None and p >= min_prob]
                det[i] = sorted(faces, key=lambda z: -z[0])[:2]
        crops, owners = [], []
        for i, (frame, fi) in enumerate(zip(buf, buf_fi)):
            if i in det and det[i]:
                state["last_faces"] = det[i]
            faces = state["last_faces"]
            if not faces:
                continue
            if len(faces) == 2:
                faces = sorted(faces, key=lambda z: (z[1][0] + z[1][2]) / 2)
                sides = ["L", "R"]
            else:
                sides = ["L" if (faces[0][1][0] + faces[0][1][2]) / 2 < W / 2 else "R"]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            for (p, b), side in zip(faces, sides):
                c = crop_face(frame, b)
                if c is None:
                    continue
                x1, y1, x2, y2 = b
                w, h = x2 - x1, y2 - y1
                face48 = _gray_crop(gray, b, (48, 48))
                mouth = _gray_crop(gray, (x1 + 0.2 * w, y1 + 0.6 * h, x2 - 0.2 * w, y2), (32, 24))
                if face48 is None or mouth is None:
                    continue
                crops.append(c)
                owners.append((fi, side, face48, mouth))
        if crops:
            for v, (fi, side, face48, mouth) in zip(emb.embed(crops), owners):
                for uid in need[fi]:
                    res[uid][side].append(v)
                    res[uid][side + "_face"].append(face48)
                    res[uid][side + "_mouth"].append(mouth)
        state["count"] += len(buf)
        buf.clear()
        buf_fi.clear()

    for fi in sorted(need):
        while state["cur"] < fi:
            if not cap.grab():
                break
            state["cur"] += 1
        ok, frame = cap.read()
        if not ok:
            continue
        state["cur"] += 1
        buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buf_fi.append(fi)
        if len(buf) >= chunk:
            flush()
    flush()
    cap.release()
    out = {}
    for uid, d in res.items():
        out[uid] = {s: (np.stack(d[s]) if d[s] else np.zeros((0, VIS_DIM), np.float32)) for s in ("L", "R")}
        for k in ("L_face", "R_face", "L_mouth", "R_mouth"):
            out[uid][k] = np.stack(d[k]).astype(np.uint8) if d[k] else np.zeros((0, 1, 1), np.uint8)
    return out


def _lip_activity(d, side):
    """Lip activity of one side within an utterance: mean absolute difference between consecutive mouth crops,
    relative to the same quantity on the whole face (cancels head motion and compression noise)."""
    m, f = d.get(side + "_mouth"), d.get(side + "_face")
    if m is None or len(m) < 2:
        return np.nan
    dm = np.abs(np.diff(m.astype(np.int16), axis=0)).mean()
    df = np.abs(np.diff(f.astype(np.int16), axis=0)).mean()
    return float(dm / (df + 1e-3))


def speaker_sequences(dialogue_res, female_side=None):
    """{uid: ([n, 775], flag)}: the expression sequence of the speaker's face for every utterance of a dialogue.

    The utterance id gives the speaker's gender (..._F000 / ..._M000) and ``female_side`` ('L' / 'R') the side of the
    female actor. Without it, the side is decided from the lip-activity contrast between the female and the male
    utterances. If the speaker's side has no detected face, the other side is used."""
    diffs = {"F": [], "M": []}
    for uid, d in dialogue_res.items():
        sL, sR = _lip_activity(d, "L"), _lip_activity(d, "R")
        if np.isfinite(sL) and np.isfinite(sR):
            diffs[uid.split("_")[-1][0]].append(sL - sR)
    dF = float(np.mean(diffs["F"])) if diffs["F"] else np.nan
    dM = float(np.mean(diffs["M"])) if diffs["M"] else np.nan
    if np.isfinite(dF) and np.isfinite(dM):
        lip_side = "L" if dF > dM else "R"
    elif np.isfinite(dF):
        lip_side = "L" if dF > 0 else "R"
    elif np.isfinite(dM):
        lip_side = "R" if dM > 0 else "L"
    else:
        lip_side = None
    f_side = female_side or lip_side or "L"
    out = {}
    for uid, d in dialogue_res.items():
        side = f_side if uid.split("_")[-1][0] == "F" else ("R" if f_side == "L" else "L")
        other = "R" if side == "L" else "L"
        if len(d[side]) > 0:
            out[uid] = (d[side], "ok")
        elif len(d[other]) > 0:
            out[uid] = (d[other], "other_side")
        else:
            out[uid] = (np.zeros((1, VIS_DIM), np.float32), "no_face")
    return out, {"female_side": f_side, "lip_side": lip_side}


# --------------------------------------------------------------------------------------------------------- corpora
def iemocap_utterances(root):
    """uid -> (session, dialogue, start, end, emotion, valence, transcript) from the evaluation and transcription
    files of ``root`` (the folder that contains Session1 ... Session5). Non-verbal annotations such as [LAUGHTER] are
    removed from the transcripts."""
    out = {}
    for sess in range(1, 6):
        sd = os.path.join(root, f"Session{sess}", "dialog")
        trans = {}
        for f in glob.glob(os.path.join(sd, "transcriptions", "*.txt")):
            for line in open(f, encoding="utf-8", errors="replace"):
                m = re.match(r"(Ses\S+) \[(\d+\.\d+)-(\d+\.\d+)\]: (.*)", line.strip())
                if m:
                    trans[m.group(1)] = re.sub(r"\s+", " ", re.sub(r"\[[A-Za-z_ ]+\]", " ", m.group(4))).strip()
        for f in glob.glob(os.path.join(sd, "EmoEvaluation", "*.txt")):
            dialogue = os.path.basename(f)[:-4]
            for line in open(f, encoding="utf-8", errors="replace"):
                m = re.match(r"\[(\d+\.\d+) - (\d+\.\d+)\]\s+(\S+)\s+(\S+)\s+\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]", line.strip())
                if m and m.group(3) in trans:
                    out[m.group(3)] = (sess, dialogue, float(m.group(1)), float(m.group(2)), m.group(4), float(m.group(5)),
                                       trans[m.group(3)])
    return out


def meld_clips(root):
    """MELD utterances of ``root`` (the folder with train/dev/test_sent_emo.csv and MELD.Raw): score = sentiment."""
    folders = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}
    items = []
    for split in ("train", "dev", "test"):
        for r in csv.DictReader(open(os.path.join(root, f"{split}_sent_emo.csv"), encoding="utf-8", errors="replace")):
            uid = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
            text = r["Utterance"].replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            items.append({"split": split, "id": uid, "transcript": text,
                          "video": os.path.join(root, "MELD.Raw", folders[split], f"{uid}.mp4"),
                          "label": float({"positive": 1, "negative": -1}.get(r["Sentiment"].strip().lower(), 0))})
    return items


def mustardpp_clips(root):
    """MUStARD++ target utterances of ``root`` (the folder with mustard++_text_official.csv and the utterance
    videos): score = (valence - 5) / 4."""
    videos = next((d for d in (os.path.join(root, "videos", "final_utterance_videos"), os.path.join(root, "final_utterance_videos"))
                   if os.path.isdir(d)), os.path.join(root, "final_utterance_videos"))
    items = []
    for r in csv.DictReader(open(os.path.join(root, "mustard++_text_official.csv"), encoding="utf-8", errors="replace")):
        if (r.get("Sarcasm") or "").strip() == "":
            continue  # context line
        valence = float(r["Valence"]) if r["Valence"].strip() else 5.0
        items.append({"split": "all", "id": r["KEY"], "transcript": r["SENTENCE"],
                      "video": os.path.join(videos, f"{r['KEY']}.mp4"), "label": (valence - 5.0) / 4.0})
    return items


def read_ids(path):
    """Identifiers of held-out clips, one per line."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _split_dict(ids, text, text_bert, audio, vision, labels):
    return {"id": list(ids), "text": text.astype(np.float32), "text_bert": text_bert, "audio": audio,
            "audio_lengths": np.full(len(ids), T, np.int64), "vision": vision.astype(np.float16),
            "vision_lengths": np.full(len(ids), T, np.int64), "regression_labels": np.asarray(labels, np.float32)}


def _save(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=4)
    for split in ("train", "valid", "test", "pool"):
        y = out[split]["regression_labels"]
        print(f"[split] {split}: N={len(y)} pos={int((y > 0).sum())} neutral={int((y == 0).sum())} neg={int((y < 0).sum())}", flush=True)
    print(f"saved -> {path} ({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)


# ---------------------------------------------------------------------------------------------------------- IEMOCAP
def build_iemocap(root, out_path, holdout=(), female_sides=None, fer="trpakov/vit-face-expression",
                  bert="bert-base-uncased", fps=5.0, workers=16, val_frac=0.1, limit_dialogues=0, shard=None,
                  vision_only=False):
    """Extract IEMOCAP and write the feature pickle. ``holdout``: utterance ids kept out of training (``pool``);
    ``female_sides``: {dialogue: 'L' | 'R'}; ``shard`` = (i, n) fills the per-dialogue vision cache of dialogues with
    index % n == i (several shards can run concurrently, then one run without ``vision_only`` assembles the pickle)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = os.path.join(os.path.dirname(out_path), "cache")
    os.makedirs(os.path.join(cache, "vision"), exist_ok=True)
    utt = iemocap_utterances(root)
    by_dialogue = defaultdict(list)
    for uid in sorted(utt):
        sess, dialogue = utt[uid][0], utt[uid][1]
        if os.path.exists(os.path.join(root, f"Session{sess}", "sentences", "wav", dialogue, f"{uid}.wav")):
            by_dialogue[dialogue].append(uid)
    dialogues = sorted(by_dialogue)[:limit_dialogues or None]
    uids = [u for d in dialogues for u in by_dialogue[d]]
    print(f"[iemocap] {len(uids)} utterances in {len(dialogues)} dialogues", flush=True)
    female_sides = female_sides or {}

    emb = FaceEmbedder(fer, device)
    t0 = time.time()
    for di, dialogue in enumerate(dialogues):
        cp = os.path.join(cache, "vision", f"{dialogue}.pkl")
        if os.path.exists(cp) or (shard and di % shard[1] != shard[0]):
            continue
        sess = utt[by_dialogue[dialogue][0]][0]
        video = os.path.join(root, f"Session{sess}", "dialog", "avi", "DivX", f"{dialogue}.avi")
        res = dialogue_face_features(video, [(u, utt[u][2], utt[u][3]) for u in by_dialogue[dialogue]], emb, fps_v=fps)
        with open(cp, "wb") as f:
            pickle.dump(res, f, protocol=4)
        print(f"  [visual] {di + 1}/{len(dialogues)} {dialogue} ({time.time() - t0:.0f}s)", flush=True)
    if vision_only:
        return None
    index = {u: i for i, u in enumerate(uids)}
    vision, flags = np.zeros((len(uids), T, VIS_DIM), np.float32), defaultdict(int)
    for dialogue in dialogues:
        with open(os.path.join(cache, "vision", f"{dialogue}.pkl"), "rb") as f:
            seqs, _ = speaker_sequences(pickle.load(f), female_sides.get(dialogue))
        for u, (seq, flag) in seqs.items():
            vision[index[u]] = pool_T(seq).numpy()
            flags[flag] += 1
    print(f"[visual] {vision.shape} {dict(flags)}", flush=True)

    text, text_bert = text_features([utt[u][6] for u in uids], device, bert)
    print(f"[text] {text.shape}", flush=True)

    audio, bad = np.zeros((len(uids), T, 65), np.float32), []
    wavs = [os.path.join(root, f"Session{utt[u][0]}", "sentences", "wav", utt[u][1], f"{u}.wav") for u in uids]
    with get_context("spawn").Pool(workers) as pool:  # 'spawn': forked workers deadlock after CUDA initialisation
        for i, (x, _) in enumerate(pool.imap(audio_features, wavs, chunksize=8)):
            if x is None:
                bad.append(uids[i])
            else:
                audio[i] = x
    print(f"[audio] {audio.shape}, failed {len(bad)}", flush=True)

    train_dialogues = sorted(d for d in dialogues if utt[by_dialogue[d][0]][0] <= 4)
    rng = np.random.RandomState(0)
    val = set(rng.choice(train_dialogues, size=max(1, int(round(val_frac * len(train_dialogues)))), replace=False).tolist()) if train_dialogues else set()
    holdout = set(holdout)
    split_of = {u: "test" if utt[u][0] == 5 else ("pool" if u in holdout else ("valid" if utt[u][1] in val else "train")) for u in uids}
    out = {}
    for split in ("train", "valid", "test", "pool"):
        ii = [i for i, u in enumerate(uids) if split_of[u] == split]
        out[split] = _split_dict([uids[i] for i in ii], text[ii], text_bert[ii], audio[ii], vision[ii],
                                 [utt[uids[i]][5] - 3.0 for i in ii])
    _save(out, out_path)
    return out


# ------------------------------------------------------------------------------------------------- MELD, MUStARD++
def build_clips(source, root, out_path, holdout=(), fer="trpakov/vit-face-expression", bert="bert-base-uncased",
                fps=5.0, workers=12, val_frac=0.1, limit=0, shard=None, vision_only=False):
    """Extract MELD (``source='meld'``) or MUStARD++ (``'mustardpp'``) and write the feature pickle."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = os.path.join(os.path.dirname(out_path), "cache")
    os.makedirs(os.path.join(cache, "vision"), exist_ok=True)
    items = [it for it in (meld_clips if source == "meld" else mustardpp_clips)(root) if os.path.exists(it["video"])]
    items = items[:limit or None]
    key = lambda it: f"{it['split']}__{it['id']}"
    print(f"[{source}] {len(items)} clips with video", flush=True)

    todo = [(i, it) for i, it in enumerate(items) if not os.path.exists(os.path.join(cache, "vision", key(it) + ".npy"))
            and (shard is None or i % shard[1] == shard[0])]
    if todo:
        emb = FaceEmbedder(fer, device)
        for n, (i, it) in enumerate(todo):
            try:
                v = clip_face_features(it["video"], emb, fps_v=fps)
            except Exception as e:
                print(f"  [visual] failed {key(it)}: {e}", flush=True)
                v = None
            np.save(os.path.join(cache, "vision", key(it) + ".npy"), np.zeros((0, VIS_DIM), np.float32) if v is None else v.astype(np.float32))
            if (n + 1) % 500 == 0:
                print(f"  [visual] {n + 1}/{len(todo)}", flush=True)
    if vision_only:
        return None

    text, text_bert = text_features([it["transcript"] for it in items], device, bert)
    print(f"[text] {text.shape}", flush=True)

    audio, bad = np.zeros((len(items), T, 65), np.float32), []
    jobs = [(it["video"], os.path.join(cache, "wav", key(it) + ".wav")) for it in items]
    with get_context("spawn").Pool(workers) as pool:
        for i, (x, _) in enumerate(pool.imap(clip_audio_features, jobs, chunksize=8)):
            if x is None:
                bad.append(key(items[i]))
            else:
                audio[i] = x
    print(f"[audio] {audio.shape}, failed {len(bad)}", flush=True)

    vision, no_face = np.zeros((len(items), T, VIS_DIM), np.float32), 0
    for i, it in enumerate(items):
        v = np.load(os.path.join(cache, "vision", key(it) + ".npy"))
        if len(v) == 0:
            no_face += 1
        else:
            vision[i] = pool_T(v).numpy()
    print(f"[visual] {vision.shape}, clips without a face: {no_face}", flush=True)

    rng = np.random.RandomState(0)
    holdout = set(holdout)
    split_of = {}
    if source == "meld":  # official training split minus 10% of its dialogues; the development split is held out
        dialogues = sorted({it["id"].split("_")[0] for it in items if it["split"] == "train"})
        val = set(rng.choice(dialogues, size=max(1, int(round(val_frac * len(dialogues)))), replace=False).tolist()) if dialogues else set()
        for it in items:
            split_of[key(it)] = {"test": "test", "dev": "pool"}.get(it["split"]) or ("valid" if it["id"].split("_")[0] in val else "train")
    else:                 # held-out candidates, then 80 / 10 / 10
        rest = [it for it in items if it["id"] not in holdout]
        n_val = int(round(val_frac * len(rest)))
        for r, j in enumerate(rng.permutation(len(rest))):
            split_of[key(rest[j])] = "valid" if r < n_val else ("test" if r < 2 * n_val else "train")
        for it in items:
            if it["id"] in holdout:
                split_of[key(it)] = "pool"
    out = {}
    for split in ("train", "valid", "test", "pool"):
        ii = [i for i, it in enumerate(items) if split_of[key(it)] == split]
        out[split] = _split_dict([items[i]["id"] for i in ii], text[ii], text_bert[ii], audio[ii], vision[ii],
                                 [items[i]["label"] for i in ii])
    _save(out, out_path)
    return out
