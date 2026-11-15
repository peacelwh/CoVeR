# Results

Outputs of the scripts are written here. Nothing in this directory is needed to run the code; checkpoints and run
directories are git-ignored.

```
results/
  runs/<dataset>/<name>_s<seed>/     one training run (scripts/train.py)
      config.json                    hyper-parameters of the run
      history.json                   training losses and validation metrics per epoch
      best.pt                        checkpoint with the best validation Avg Corr (trainable tensors)
      results.json                   test metrics: accuracy / robust mode -> patterns, rates, mrbench_a
      results_override.json          test-time override of an LLM-Fusion run (scripts/evaluate.py --tau_test)
      results_steering.json          activation steering of an LLM-Fusion run (scripts/evaluate.py --steering)
      diagnosis_<pattern>.json       reliance per layer and conflict-detection AUROC (scripts/diagnose.py)
  tables/                            main.md, mrbench_a.md, patterns.md, summary.json (scripts/make_tables.py)
  figures/                           reliance curves (scripts/plot_reliance.py)
  mrbench_h/                         item-level predictions on MR-Bench-H (scripts/predict_mrbench_h.py,
                                     scripts/eval_mllm.py)
  mrbench_h.json                     MR-Bench-H scores (scripts/score_mrbench_h.py)
```

`results.json` stores fractions (Acc2, F1, Cons, Conf, ShortErr) and raw MAE / correlation; the tables report Acc2, F1
and the MR-Bench-A metrics in % and, where stated, the correlation ×100. Averages over the seven availability patterns
(`Avg`) and the missing rates (`Rate`) are arithmetic means, except for the correlation, which is averaged through
Fisher's z-transform; tables are means over the seeds of a run name.
