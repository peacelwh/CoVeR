#!/usr/bin/env python
"""Extract the features of IEMOCAP, MELD or MUStARD++ into an MMSA-style pickle (see ``cover.features``).

    text: BERT-base last hidden states   audio: openSMILE ComParE_2016 descriptors   visual: ViT facial-expression
    embedding of the speaker's face (IEMOCAP) or of the largest confident face (MELD, MUStARD++), MTCNN at 5 fps

    python scripts/extract_features.py --source iemocap --root /data/IEMOCAP_full_release
    python scripts/extract_features.py --source meld --root /data/MELD
    python scripts/extract_features.py --source mustardpp --root /data/MUStARD++
    python scripts/preprocess.py --datasets iemocap meld mustardpp

``--root``: IEMOCAP: the folder with Session1 ... Session5; MELD: the folder with {train,dev,test}_sent_emo.csv and
MELD.Raw/; MUStARD++: the folder with mustard++_text_official.csv and the utterance videos. The visual stream is the
slow part: fill its cache with several ``--vision_only --shard i/n`` processes, then run once without these options.
Requires ffmpeg, opensmile, facenet-pytorch and opencv-python.
"""
import argparse
import json
import os

import _path  # noqa: F401
from cover import features


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=list(features.OUT))
    ap.add_argument("--root", required=True, help="folder of the corpus (see above)")
    ap.add_argument("--out_dir", default="data/raw")
    ap.add_argument("--holdout", default=None,
                    help="ids of the clips held out of training, one per line (default: data/mrbench_h_holdout/<source>.txt)")
    ap.add_argument("--sides", default="data/iemocap_female_side.json", help="IEMOCAP: side of the female actor per dialogue")
    ap.add_argument("--fer", default="trpakov/vit-face-expression", help="ViT fine-tuned on FER2013 (hub id or local path)")
    ap.add_argument("--bert", default="bert-base-uncased")
    ap.add_argument("--fps", type=float, default=5.0, help="frames per second of the visual stream")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--shard", default="", help="i/n: only fill the visual cache of items with index %% n == i")
    ap.add_argument("--vision_only", action="store_true", help="only fill the visual cache")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the first N dialogues (IEMOCAP) or clips")
    args = ap.parse_args()
    out_path = os.path.join(args.out_dir, features.OUT[args.source])
    holdout = features.read_ids(args.holdout or os.path.join("data", "mrbench_h_holdout", f"{args.source}.txt"))
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    common = dict(holdout=holdout, fer=args.fer, bert=args.bert, fps=args.fps, workers=args.workers, val_frac=args.val_frac,
                  shard=shard, vision_only=args.vision_only)
    if args.source == "iemocap":
        sides = json.load(open(args.sides)) if os.path.exists(args.sides) else {}
        print(f"[iemocap] {len(holdout)} held-out utterances; side of the female actor known for {len(sides)} dialogues")
        features.build_iemocap(args.root, out_path, female_sides=sides, limit_dialogues=args.limit, **common)
    else:
        print(f"[{args.source}] {len(holdout)} held-out clips")
        features.build_clips(args.source, args.root, out_path, limit=args.limit, **common)


if __name__ == "__main__":
    main()
