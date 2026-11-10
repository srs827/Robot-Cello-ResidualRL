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

## reward/piece_reward.py  ← score a whole song note by note
- **Input:** a recording of a whole piece (numpy array or wav / mp3 file) plus where the notes are: the MIDI score (`--midi`), the robot run log csv (`--log`, uses `time_elapsed_sec`, `current_note_number`, `remaining_duration_sec`), or a list of `(start_s, end_s, midi_pitch)`. With none of them it cuts notes by detected pitch (less exact: repeated notes merge).
- **What it does:** finds every note (MIDI: lines the score up with the recording by pitch and splits repeated notes at bow changes), then gives each note a reward and its main issue (same aspects as `explain()`, without the "ending" one, which the cause test showed the model does not really use) and how far its pitch is from the score in cents. With a MIDI score it also checks rhythm: each note's length to the next note vs the score, after removing the overall tempo.
- **Output:** a dict: `reward` (average over notes), `share_good` (share of notes ≥ 0.5), `issue_counts`, `worst_notes`, with MIDI also `slower_than_score`, `timing_dev_median`, `share_on_time` (within ±20 %), and `notes` = one row per note (`start`, `end`, `note`, `reward`, `issue`, `cents_off`, `timing_dev`).
- **How to run:**
  ```python
  import sys; sys.path.insert(0, "path/to/reward")
  from piece_reward import PieceReward, notes_from_log
  R = PieceReward("fast")
  res = R.piece(audio, sr=22050, midi="song.mid")                                  # with the score
  res = R.piece(audio, sr=22050, notes=notes_from_log("run.csv"), check_pitch=False)  # in the RL loop, ~80 ms per note
  ```
  From the command line: `python reward/piece_reward.py song.mp3 --midi song.mid --out result.json`. Needs `pretty_midi` only for `--midi`.

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

## training/train_vcello.py
- **Input:** the fast model (`distill12`), the built dataset, and the virtual cello calibration in `outputs/virtual_cello/virtual_cello.json`.
- **What it does:** experiment: fine-tunes the fast model so clean bowing on the virtual cello scores higher than double-slip or crushed bowing, while keeping its scores on real recordings the same. Trains on the G and A strings only (D is kept out for testing). `--lam` sets how strong the virtual cello part is.
- **Output:** `candidates/<name>_seed0.pt`
- **How to run:** `python training/train_vcello.py --name fast_vc --lam 1.0` (about 10 minutes on CPU). Then `python reward/build_reference.py --models fast_vc --model_dir candidates` before using it.

## candidates/
- **Input:** –
- **What it does:** experimental models, NOT used by default: `fast_vc` (strong virtual-cello retrain, lam 1.0) and `fast_vc03` (gentle, lam 0.3), plus their `reference.json`. Results: `outputs/evaluation_candidate/`, `outputs/virtual_cello_candidate/`, `outputs/virtual_cello_candidate03/`, `outputs/pieces/pieces_candidate.json`. Both lose some accuracy on real recordings, so the shipped `fast` stays the recommended model.
- **Output:** –
- **How to run:** `Reward("fast_vc", model_dir="candidates")`. To compare with the current model in `evaluate.py`, copy `checkpoints/distill12_seed0.pt` into `candidates/` and run `python evaluation/evaluate.py --models "fast_vc03,fast_vc,distill12" --baseline distill12 --model_dir candidates --out outputs/evaluation_candidate`.

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

## analysis/one_feature_test.py
- **Input:** a model, the built dataset.
- **What it does:** cause test: takes 75 held-out good takes, changes only one thing at 4 strengths (added noise, volume wobble, fading end, tone-colour wobble, plus two controls: quieter and fixed EQ) and measures how much the reward drops and whether `explain()` flags the matching aspect.
- **Output:** `outputs/explain/one_feature_test.json`
- **How to run:** `python analysis/one_feature_test.py`

## analysis/mp3_test.py
- **Input:** a model, the built dataset.
- **What it does:** checks whether MP3 compression changes the reward: held-out good and bad takes are turned into MP3 (about 80 and 50 kbps, like the robot song files) and scored again.
- **Output:** `outputs/explain/mp3_test.json` (mean reward of good / bad takes, good-vs-bad AUC, change vs original)
- **How to run:** `python analysis/mp3_test.py`

## analysis/score_pieces.py
- **Input:** recordings with a MIDI file of the same name (default: the robot recordings in `../lol/Robot/` and scores in `../lol/Midi/` of this repo).
- **What it does:** runs `piece_reward.py` on every song: score per note, per song average, main issues, pitch per note, rhythm, and the timing error before a string change vs on the same string.
- **Output:** `outputs/pieces/pieces.json`
- **How to run:** `python analysis/score_pieces.py` (or `--audio_dir ... --midi_dir ...`, another model with `--model ... --model_dir ...`; about 0.5 s of compute per second of audio on CPU, mostly pitch detection).

## analysis/virtual_cello.py
- **Input:** bow force, bow speed, contact point (0.06–0.20 of the string length from the bridge) and pitch.
- **What it does:** a physical model of a bowed cello string (waveguide string + bow friction + body resonances). `play()` returns the audio and a physics check: how often the string moves in clean "Helmholtz motion" (one slip per cycle).
- **Output:** audio (numpy, 22.05 kHz) and the Helmholtz share (0–1).
- **How to run:** imported by `virtual_cello_test.py`; `from virtual_cello import play; y, helm = play(1.1, 0.1)`.

## analysis/virtual_cello_test.py
- **Input:** a model; for the body-colour calibration also `outputs/pieces/pieces.json` and the robot songs in `../lol/Robot/`.
- **What it does:** (1) matches the virtual cello's body colour to the real robot recordings, (2) plays a 20 × 12 grid of bow force × bow speed and compares the reward with the physics (clean vibration / too much / too little force), (3) the classic Schelleng diagram (contact point × force) on the G, D and A strings, (4) a tiny RL loop: a cross-entropy optimizer learns bow force, speed and contact point from 5 starting points using only the reward.
- **Output:** `outputs/virtual_cello/virtual_cello.json` (`virtual_cello_uncalibrated.json` = same test without the calibration); with `--audio_out <folder>` also before/after wav files of a short melody.
- **How to run:** `python analysis/virtual_cello_test.py` (about 5 minutes on CPU; `--calibrate no` skips the calibration; a candidate model: `--model fast_vc --model_dir candidates --out outputs/virtual_cello_candidate`).

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
- **What it does:** results of the shipped models: `evaluation/` (summary.csv, evaluation.json, and `all_models_original_run.md` = every model from the original runs incl. the old RL reward), `explain/` (feature and position analysis), `examples/` (explain / compare outputs for a good and a bad take), `pieces/` (note-by-note scores of 5 robot songs), `virtual_cello/` (reward vs bowing physics and the tiny RL loop).
- **Output:** –
- **How to run:** –
