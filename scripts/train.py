#!/usr/bin/env python
"""Train LLM-Fusion or CoVeR and evaluate the selected checkpoint (see ``cover.train`` for the options).

    python scripts/train.py --dataset mosi --method cover --seed 42
"""
import _path  # noqa: F401
from cover.train import main

if __name__ == "__main__":
    main()
