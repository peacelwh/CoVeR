#!/usr/bin/env python
"""Zero-shot evaluation of MLLMs on MR-Bench-H (Sec. 2.3, supplementary Sec. A, "MLLM evaluation").

Each clip is presented to an MLLM under every view it supports:

    full              clip with its audio track (V+A)
    transcript        transcript only (T)
    muted             muted video (V)
    audio             audio only (A)
    muted+transcript  muted video with the transcript (V+T), for vision-language models

The question is "Is the overall sentiment expressed by the speaker in this clip positive or negative? Answer with
exactly one word: positive or negative.", prefixed by the transcript for transcript views; Chinese clips are prompted
in English. Decoding is greedy with at most six new tokens; the first occurrence of "positive" or "negative" is the
answer, and any other answer counts as wrong. Models run in bf16 on one GPU with their official chat templates.

    python scripts/eval_mllm.py --model qwen2.5-omni-7b --model_path /models/Qwen2.5-Omni-7B \
        --items /path/to/MR-Bench-H/data/test.jsonl --clips /path/to/clips --out results/mrbench_h/qwen2.5-omni-7b.jsonl

``--clips`` is the folder written by the benchmark's ``tools/build_clips.py`` (``<uid>.mp4``). Score the output with
scripts/score_mrbench_h.py.
"""
import argparse
import json
import os
import re
import time

QUESTION = ("Is the overall sentiment expressed by the speaker in this clip positive or negative? "
            "Answer with exactly one word: positive or negative.")
TEXT_PREFIX = "Here is what the speaker says in a short video clip: \"{t}\"\n"


def question(view, transcript):
    if view == "transcript":
        return TEXT_PREFIX.format(t=transcript) + QUESTION.replace("in this clip ", "")
    if view == "muted+transcript":
        return TEXT_PREFIX.format(t=transcript) + QUESTION
    return QUESTION


def parse_polarity(answer):
    """First occurrence of positive / negative; any other answer is invalid (None) and counts as wrong."""
    m = re.search(r"\b(positive|negative)\b", (answer or "").strip().lower())
    return {"positive": 1, "negative": -1}[m.group(1)] if m else None


def _to(inputs, model):
    return inputs.to(model.device, dtype=model.dtype) if hasattr(inputs, "to") else inputs


class QwenOmni:
    """Qwen2.5-Omni (thinker, text output): video sampled at 2 fps with at most 360 x 420 pixels per frame; the audio
    is interleaved with ``use_audio_in_video``."""
    VIEWS = ["full", "transcript", "muted", "audio"]
    SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory "
              "and visual inputs, as well as generating text and speech.")

    def __init__(self, path, max_pixels=360 * 420, fps=2.0):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        self.torch = torch
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.proc = Qwen2_5OmniProcessor.from_pretrained(path)
        self.max_pixels, self.fps = max_pixels, fps

    def ask(self, view, clip, transcript):
        from qwen_omni_utils import process_mm_info
        q = question(view, transcript)
        if view == "transcript":
            content, use_av = [{"type": "text", "text": q}], False
        elif view == "audio":
            content, use_av = [{"type": "audio", "audio": clip}, {"type": "text", "text": q}], False
        else:
            content = [{"type": "video", "video": clip, "max_pixels": self.max_pixels, "fps": self.fps},
                       {"type": "text", "text": q}]
            use_av = view == "full"
        conv = [{"role": "system", "content": [{"type": "text", "text": self.SYSTEM}]}, {"role": "user", "content": content}]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=use_av)
        inputs = self.proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                           use_audio_in_video=use_av).to(self.model.device).to(self.model.dtype)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, use_audio_in_video=use_av, max_new_tokens=6, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class VideoVLM:
    """Vision-language models without audio input (Qwen2.5-VL, InternVL3, LLaVA-OneVision): 16 uniformly sampled
    frames; 'muted' = frames only, 'muted+transcript' = frames with the transcript."""
    VIEWS = ["muted", "muted+transcript", "transcript"]

    def __init__(self, path, num_frames=16):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.num_frames = torch, num_frames
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def ask(self, view, clip, transcript):
        if view == "transcript":
            content, kw = [{"type": "text", "text": question(view, transcript)}], {}
        else:
            content = [{"type": "video", "path": clip}, {"type": "text", "text": question(view, transcript)}]
            kw = {"num_frames": self.num_frames}
        inputs = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors="pt", **kw)
        inputs = _to(inputs, self.model)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=6, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class QwenAudio:
    """Qwen2-Audio-7B: 'audio' = the soundtrack (voice and words), 'transcript' = text only."""
    VIEWS = ["audio", "transcript"]

    def __init__(self, path):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def ask(self, view, clip, transcript):
        import librosa
        sr = self.proc.feature_extractor.sampling_rate
        if view == "transcript":
            conv = [{"role": "user", "content": [{"type": "text", "text": question(view, transcript)}]}]
            audios = None
        else:
            conv = [{"role": "user", "content": [{"type": "audio", "audio_url": clip}, {"type": "text", "text": QUESTION}]}]
            audios = [librosa.load(clip, sr=sr)[0]]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = _to(self.proc(text=text, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True), self.model)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=6, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class TextLLM:
    """Text-only LLM on the transcript (the text-only reader and the reader panel of stage 2)."""
    VIEWS = ["transcript"]

    def __init__(self, path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def ask(self, view, clip, transcript):
        conv = [{"role": "user", "content": question("transcript", transcript)}]
        # enable_thinking=False switches off the reasoning block of Qwen3; other chat templates ignore the variable
        enc = self.tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                           enable_thinking=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=6, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class MiniCPMo:
    """MiniCPM-o 2.6 (remote code; use the environment of its model card): one frame and one second of 16 kHz audio
    per unit, at least four units for the frames-only view; 'muted' keeps the frames, 'audio' the audio."""
    VIEWS = ["full", "transcript", "muted", "audio"]

    def __init__(self, path, min_units=4):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.min_units = torch, min_units
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
                                               init_vision=True, init_audio=True, init_tts=False).eval().cuda()
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    def _units(self, clip, n_min=1):
        """One unit per second = (frame at the middle of the second, 1 s of 16 kHz audio; the last second zero-padded)."""
        import librosa
        import numpy as np
        from decord import VideoReader, cpu
        from PIL import Image
        vr = VideoReader(clip, ctx=cpu(0))
        fps = max(vr.get_avg_fps(), 1e-3)
        dur = len(vr) / fps
        n = max(n_min, int(np.ceil(dur)))
        times = (np.arange(n) + 0.5) * dur / n
        frames = vr.get_batch([min(int(t * fps), len(vr) - 1) for t in times]).asnumpy()
        try:
            audio = librosa.load(clip, sr=16000, mono=True)[0]
        except Exception:
            audio = np.zeros(int(16000 * dur), dtype=np.float32)
        audio = np.pad(audio, (0, max(0, 16000 * n - len(audio))))[: 16000 * n].astype(np.float32)
        return [(Image.fromarray(f.astype("uint8")), audio[16000 * i: 16000 * (i + 1)]) for i, f in enumerate(frames)], audio

    def ask(self, view, clip, transcript):
        q = question(view, transcript)
        kw = dict(tokenizer=self.tok, sampling=False, max_new_tokens=6, use_tts_template=False, generate_audio=False,
                  max_slice_nums=1, use_image_id=False)
        if view == "transcript":
            res = self.model.chat(msgs=[{"role": "user", "content": [q]}], **kw)
        else:
            units, audio = self._units(clip, n_min=1 if view != "muted" else self.min_units)
            if view == "full":
                content = [x for img, a in units for x in ("<unit>", img, a)] + [q]
                msgs = [self.model.get_sys_prompt(mode="omni", language="en"), {"role": "user", "content": content}]
                res = self.model.chat(msgs=msgs, omni_input=True, **kw)
            elif view == "muted":
                res = self.model.chat(msgs=[{"role": "user", "content": [img for img, _ in units] + [q]}], **kw)
            else:
                res = self.model.chat(msgs=[{"role": "user", "content": [q, audio]}], **kw)
        return res if isinstance(res, str) else getattr(res, "text", str(res))


MODELS = {
    # zero-shot MLLMs of the benchmark results
    "qwen2-audio-7b": QwenAudio, "minicpm-o-2.6": MiniCPMo, "qwen2.5-omni-7b": QwenOmni, "qwen2.5-omni-3b": QwenOmni,
    "internvl3-8b": VideoVLM, "llava-ov-7b": VideoVLM, "qwen2.5-vl-7b": VideoVLM, "qwen2.5-7b": TextLLM,
    # the other members of the text-only reader panel (stage 2 of the construction)
    "qwen3-8b": TextLLM, "yi-1.5-9b": TextLLM, "phi-3.5-mini": TextLLM,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--items", required=True, help="item list of the benchmark (data/test.jsonl)")
    ap.add_argument("--clips", required=True, help="folder with <uid>.mp4 (benchmark tools/build_clips.py)")
    ap.add_argument("--views", nargs="+", default=None, help="default: every view the model supports")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    args = ap.parse_args()
    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[: args.limit]
    done = set()
    if os.path.exists(args.out):  # resume
        with open(args.out, encoding="utf-8") as f:
            done = {(r["uid"], r["view"]) for r in map(json.loads, f)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model = MODELS[args.model](args.model_path)
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for n, it in enumerate(items):
            clip = os.path.join(args.clips, f"{it['uid']}.mp4")
            for view in (args.views or model.VIEWS):
                if (it["uid"], view) in done:
                    continue
                if "transcript" in view and not it.get("transcript"):
                    raise SystemExit(f"{it['uid']}: no transcript (fill withheld transcripts with the benchmark's tools/fill_transcripts.py)")
                if view != "transcript" and not os.path.exists(clip):
                    raise SystemExit(f"missing clip {clip}")
                try:
                    answer = model.ask(view, clip, it.get("transcript"))
                except Exception as e:  # unreadable clip, no audio stream: an invalid answer, counted as wrong
                    answer = f"ERROR: {type(e).__name__}: {str(e)[:120]}"
                f.write(json.dumps({"uid": it["uid"], "model": args.model, "view": view, "answer": answer,
                                    "polarity": parse_polarity(answer)}, ensure_ascii=False) + "\n")
                f.flush()
            if n % 50 == 0:
                print(f"{n}/{len(items)} clips, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
