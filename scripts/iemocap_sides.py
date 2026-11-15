#!/usr/bin/env python
"""Which side of the IEMOCAP frame shows the female actor, per dialogue, read off by a vision-language model.

IEMOCAP dialogues are dyads of one female and one male actor who keep their positions for the whole dialogue, and the
utterance id gives the speaker's gender. Knowing the side of the female actor therefore gives the speaker's face of
every utterance (``cover.features``). Qwen2.5-VL-7B is asked on five frames per dialogue, twice per frame: on the frame
and on its mirror image (the answer must flip). The result is distributed as ``data/iemocap_female_side.json``.

    python scripts/iemocap_sides.py --root /data/IEMOCAP_full_release --model /models/Qwen2.5-VL-7B-Instruct
"""
import argparse
import glob
import json
import os
import re

import numpy as np

PROMPT = ("This picture shows two people, a woman and a man. Is the woman on the left or on the right side of the picture? "
          "Answer with one word: left or right.")


def frames_of(video, n):
    import cv2
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for q in np.linspace(0.1, 0.9, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(q * total))
        ok, frame = cap.read()
        if ok:
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with Session1 ... Session5")
    ap.add_argument("--model", required=True, help="Qwen2.5-VL-7B-Instruct (hub id or local path)")
    ap.add_argument("--out", default="data/iemocap_female_side.json")
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    videos = sorted(glob.glob(os.path.join(args.root, "Session*", "dialog", "avi", "DivX", "*.avi")))
    if args.limit:
        videos = videos[:args.limit]
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def ask(img):
        conv = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img)}, {"type": "text", "text": PROMPT}]}]
        inputs = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        answer = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
        m = re.search(r"left|right", answer)
        return m.group(0) if m else "?"

    mirror = {"left": "right", "right": "left", "?": "?"}
    sides, consistent = {}, 0
    for vi, video in enumerate(videos):
        dialogue = os.path.basename(video)[:-4]
        votes, flipped = [], []
        for frame in frames_of(video, args.n_frames):
            votes.append(ask(frame))
            flipped.append(ask(np.ascontiguousarray(frame[:, ::-1])))
        every = votes + [mirror[v] for v in flipped]
        n_left, n_right = every.count("left"), every.count("right")
        sides[dialogue] = "L" if n_left > n_right else ("R" if n_right > n_left else None)
        agree = sum(1 for a, b in zip(votes, flipped) if a != "?" and mirror[b] == a)
        consistent += int(agree == len(votes))
        print(f"[{vi + 1}/{len(videos)}] {dialogue}: {sides[dialogue]} ({agree}/{len(votes)} frames consistent)", flush=True)
        with open(args.out, "w") as f:
            json.dump(sides, f, indent=0)
    print(f"{len(sides)} dialogues, {consistent} fully consistent -> {args.out}")


if __name__ == "__main__":
    main()
