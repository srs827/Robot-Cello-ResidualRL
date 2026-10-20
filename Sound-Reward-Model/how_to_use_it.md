# How to use it

Setup (once): Python 3.10, then `pip install -r requirements.txt`. Run every command from inside the `Sound-Reward-Model` folder.

The recordings (`.wav`) are not in GitHub. To rebuild the dataset or retrain, copy `bryan/data/local/` from the team Google Drive folder "Robotic Cellist Glockenspiel 2026 Fall" into `data/local/` here. Using the reward does not need them.

---

## run_all.py
- **Input:** nothing (downloads public data itself; uses `data/local/` and `data/human_pairs.csv`).
- **What it does:** runs the whole pipeline in order: download → build dataset → train 12 models → distil into 1 fast model → score reference takes → evaluate → explanation analysis. Finished models are skipped, so it can be restarted.
- **Output:** new models in `retrained/` (the shipped models in `checkpoints/` are not touched), results in `outputs/evaluation_retrained/`, `outputs/explain_retrained/`, `outputs/examples_retrained/`.
- **How to run:** `python run_all.py` (CPU: about 15–20 h for training + ~1 h for the rest). Only some stages: `python run_all.py --stages "dataset,train"`.

## requirements.txt
- **Input:** –
- **What it does:** lists the Python packages and versions.
- **Output:** –
- **How to run:** `pip install -r requirements.txt`

## writeup-oct21.md
- **Input:** –
- **What it does:** what was done this week, the results, and next steps.
- **Output:** –
- **How to run:** read it. Slides: `bryan/sound_explain.pptx` in the team Google Drive folder.

---

## reward/reward.py  ← use this as the RL reward
- **Input:** a waveform (numpy array, mono or stereo, float or int16, any length ≥ 0.1 s) and its sample rate.
- **What it does:** scores the take with the trained model(s) and explains the score.
  - `reward(y, sr)` → float in [0, 1]: chance this take beats a typical good take (0.5 = as good as a normal good take). 0.0 for empty / too short / silent audio. Never crashes.
  - `explain(y, sr)` → reward + score of the start / middle / end of the note + 6 aspects (tone cleanliness, worst-part cleanliness, volume steadiness, ending, tone colour steadiness, noisiness), each compared with 384 good takes, with a status (ok / weak / problem) and what to improve.
  - `compare(y_a, y_b, sr)` → which take is better, probability, and the aspects where the winner is better.
  - Model choice: `Reward("fast")` = 1 network (about 25 ms for a 2-second take on a laptop CPU, for the RL loop); `Reward("ensemble12")` = 12 networks averaged (about 12× slower, most accurate). `explain()` costs a little more than `reward()`. Any single model name in `checkpoints/` also works, e.g. `Reward("m1_pairwise")`.
- **Output:** a float (`reward`) or a dict (`explain`, `compare`).
- **How to run:**
  ```python
  import sys; sys.path.insert(0, "path/to/bryan/reward")
  from reward import Reward
  R = Reward("fast")
  r = R.reward(audio, sr=22050)        # RL reward
  info = R.explain(audio, sr=22050)    # info["summary"], info["aspects"], info["improve"], info["parts"]
  ```
  From the command line: `python reward/reward.py take.wav` (explain) or `python reward/reward.py a.wav b.wav` (compare).

## reward/build_reference.py
- **Input:** the trained models in `checkpoints/` and the built dataset (`data/proc/`).
- **What it does:** scores the 384 good reference takes with every model and stores their acoustic features; `reward.py` needs this to turn raw scores into 0–1 and to compare aspects.
- **Output:** `checkpoints/reference.json`
- **How to run:** `python reward/build_reference.py` (only needed after retraining; already included for the shipped models).

## checkpoints/
- **Input:** –
- **What it does:** the trained models, ready to use without retraining:
  - 12 ensemble members: `m1_pairwise`, `pw_ctl_s1`–`pw_ctl_s5` (trained without human answers), `pw_v1_s0`, `pw_v1_s1`, `pw_v2_s2`, `pw_v2_s3`, `pw_v3_s4`, `pw_v3_s5` (trained with human answers).
  - Distilled single models: `distill8` → `distill8b` → `distill12` (= `"fast"`, recommended for RL).
  - `reference.json`: reference scores and features used by `reward.py`.
- **Output:** –
- **How to run:** loaded automatically by `reward.py`, `evaluate.py` and the analysis scripts.

---

## data/download_public.py
- **Input:** internet connection.
- **What it does:** downloads the public datasets: Good-sounds (cello part, ~1.9 GB, Zenodo) and the Iowa cello arco notes (~200 MB).
- **Output:** `data/raw/good_sounds/`, `data/raw/iowa_cello/`
- **How to run:** `python data/download_public.py`

## data/build_dataset.py
- **Input:** `data/local/` (our recordings + labels) and `data/raw/` (public data).
- **What it does:** converts every recording to 22.05 kHz mono, trims silence, and makes one table of all clips with source, labels and ratings.
- **Output:** `data/proc/manifest.csv` and `data/proc/audio/<clip_id>.npy` (2,799 clips).
- **How to run:** `python data/build_dataset.py`

## data/make_human_pairs.py
- **Input:** raw A-vs-B answers, `data/human_judgments.csv` (columns `a, b, winner` with winner = a / b / tie).
- **What it does:** merges answers into one soft label per pair (`p_a` = share of votes for a, ties count half). Pairs that touch a test clip are marked `eval` and are never trained on.
- **Output:** `data/human_pairs.csv`
- **How to run:** `python data/make_human_pairs.py` (run again when new answers are added).

## data/local/
- **Input:** –
- **What it does:** the good/bad label files for our own recordings: robot recordings (`record_audio`, `Robot/data2`–`data4`) and human recordings (`Robot/human_data`). The `.wav` files come from the team Google Drive (see Setup).
- **Output:** –
- **How to run:** read by `build_dataset.py`.

## data/human_judgments.csv, data/human_pairs.csv
- **Input:** –
- **What it does:** 200 A-vs-B listening answers (raw) and the prepared pairs (75 train, 125 eval).
- **Output:** –
- **How to run:** read by `train_reward.py` and `evaluate.py`.

---

## training/train_reward.py
- **Input:** the built dataset (`data/proc/`) and `data/human_pairs.csv`.
- **What it does:** trains one reward model from "A is better than B" pairs: clean vs synthetic defects at 3 severities, Good-sounds expert ratings and labels, robot good vs bad, and human A-vs-B answers. Saves a checkpoint every 100 steps and resumes if stopped.
- **Output:** `<out>/<name>_seed<seed>.pt`
- **How to run:** `python training/train_reward.py --name my_model --seed 0` (add `--human_pairs none` to train without human answers; ~1–2 h on CPU).

## training/train_distill.py
- **Input:** teacher models and a starting model in `--models`, the built dataset.
- **What it does:** trains one network to copy the average score of several models, so the RL loop needs only one forward pass.
- **Output:** `<out>/<name>_seed0.pt`
- **How to run:** `python training/train_distill.py --name distill12 --teachers ensemble12 --init distill8b --mode real`

---

## evaluation/evaluate.py
- **Input:** models in `checkpoints/`, the built dataset, `data/human_pairs.csv`.
- **What it does:** for each model: agreement with 57 held-out human A-vs-B answers (with 95% interval and McNemar test vs a baseline), robot test AUC (B3), human test AUC (B4), synthetic defect detection (B5 seen / unseen types), and the hack test (share of bad takes that beat the median good take after the best of 20 cheap audio tricks).
- **Output:** `outputs/evaluation/evaluation.json`, `outputs/evaluation/summary.csv`
- **How to run:** `python evaluation/evaluate.py --models "ensemble12,fast,m1_pairwise"` (add `--skip_bench` for only the human test).

## evaluation/benchmark.py
- **Input:** a scorer object and the built dataset.
- **What it does:** the benchmark tests and hack test used by `evaluate.py`.
- **Output:** dicts of metrics.
- **How to run:** imported by `evaluate.py`.

---

## analysis/explain_features.py
- **Input:** a model, the built dataset.
- **What it does:** Spearman correlation between the model score and 12 acoustic features on all 327 real recordings, plus good vs bad group means.
- **Output:** `outputs/explain/explain_features.json`
- **How to run:** `python analysis/explain_features.py`

## analysis/explain_position.py
- **Input:** a model, the built dataset.
- **What it does:** checks whether the start, middle or end of the note drives the score (rank agreement of each part with the full score).
- **Output:** `outputs/explain/explain_position.json`
- **How to run:** `python analysis/explain_position.py`

## analysis/example_outputs.py
- **Input:** a model, the built dataset.
- **What it does:** runs `explain()` and `compare()` on one good and one bad real take.
- **Output:** `outputs/examples/examples_<model>.json`
- **How to run:** `python analysis/example_outputs.py --model fast`

---

## core/common.py, core/model.py, core/synth.py, core/splits.py
- **Input:** –
- **What it does:** shared code: paths and model names; the network, acoustic features and scoring; synthetic defects and recording effects; fixed train/test splits.
- **Output:** –
- **How to run:** imported by the other scripts.

## outputs/
- **Input:** –
- **What it does:** results of the shipped models: `evaluation/` (summary.csv, evaluation.json, and `all_models_original_run.md` = every model from the original runs incl. the old RL reward), `explain/` (feature and position analysis), `examples/` (explain / compare outputs for a good and a bad take).
- **Output:** –
- **How to run:** –
