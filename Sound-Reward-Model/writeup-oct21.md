# writeup-oct21

The sound reward model is now in the repo (`Sound-Reward-Model/`) with everything needed to use it or rebuild it:
data preparation, training, distillation, evaluation and the trained models. New this week: the reward can say
**why** a take is good or bad and **what to improve**, not just give a number.

---

## Part 1: What the reward is

- **Input:** audio only (no metadata). **Output:** a number in [0, 1] = the chance this take beats a typical good
  cello take (0.5 = as good as a normal good take).
- Trained only on "A is better than B" comparisons, never on absolute scores, because different people disagree on
  absolute scores much more than on comparisons. Pairs come from: synthetic defects at 3 severities, the public
  Good-sounds cello ratings and labels, our robot good/bad recordings, and human A-vs-B answers.
- Two versions: `ensemble12` (12 independently trained networks averaged, most accurate) and `fast` (one network
  distilled from the 12, for the RL loop).

## Part 2: How good it is

Measured on data no model trained on. "Agrees with human" uses 57 held-out A-vs-B answers (95% interval about ±12 points).
"Hack rate" = share of bad takes that beat the median good take after the best of 20 cheap audio tricks (EQ, reverb, added tone, limiter, ...).

| model | agrees with human | robot test AUC | human test AUC | hack rate |
|---|---|---|---|---|
| old RL reward (`MelQualityCNN`) | 60% | – | – | 100% |
| `ensemble12` | **77%** | 0.85 | 0.91 | **26%** |
| `fast` (distilled) | 72% | 0.84 | 0.86 | 29% |

The 77% vs 60% gap is close to but not past the usual significance bar with only 57 pairs (p = 0.06).
All numbers: `outputs/evaluation/summary.csv` and `outputs/evaluation/all_models_original_run.md`.

## Part 3: Why a take scores high

Correlation between the score and 12 acoustic features on all 327 of our real recordings (all p < 0.01):

- **Clean tone (HNR)** matters most (r = +0.50), especially the worst part of the note (r = +0.53).
- **Steady volume**: volume going up and down lowers the score (r = −0.47).
- **Ending holds up** instead of fading (r = +0.29); steady tone colour; less noise.
- Good-labelled takes really differ the same way: about 20 dB vs 13 dB HNR.
- **The whole note counts**: scoring only the start, only the middle or only the end ranks the takes almost the
  same as the full score (r = 0.92 / 0.92 / 0.88).

This is correlation, not proof of cause (see Part 6).

## Part 4: New: `explain()` and `compare()`

`reward/reward.py` now returns, on top of the reward:

- score of the **start / middle / end** of the note, and which part is weakest;
- **6 aspects** (tone cleanliness, worst-part cleanliness, volume steadiness, ending, tone colour steadiness,
  noisiness), each compared with 384 good reference takes, with status ok / weak / problem and **what to improve**;
- `compare(a, b)`: which take is better, how sure, and the aspects where the winner is better.

Check on all 327 real recordings: 82% of the takes humans labelled **bad** get at least one aspect flagged,
against 30% of the good ones.

Example (`outputs/examples/`), a bad take:
`reward 0.16; weakest part: start; main issue: tone cleanliness (HNR) -> cleaner tone: less scratch / noise in the sound`

## Part 5: How to plug it into RL

```python
import sys; sys.path.insert(0, "Sound-Reward-Model/reward")
from reward import Reward
R = Reward("fast")                  # about 25 ms per 2-second take on a laptop CPU
r = R.reward(audio, sr=22050)       # use as the sound term of the RL reward
info = R.explain(audio, sr=22050)   # for logging / debugging what the policy is doing
```

Keep the existing pitch / onset / dynamics terms next to it: the model has only heard cello and still can be
fooled sometimes.

## Part 6: Problems and next steps

Problems:
1. Almost all human comparisons are from one person, so the model mostly learned one person's taste.
2. Still foolable: 26% of bad takes can be pushed above a typical good take with cheap tricks.
3. Not yet tested inside robot training, so we have not seen what a policy learns from it.

Next:
1. More A-vs-B labels from the team (labeling page works; re-run `run_all.py` with the new answers).
2. One-feature tests: change only one thing in a good take (add noise / volume wobble / fade the ending) and check
   the score really drops. Turns Part 3 from correlation into cause; useful for the paper.
3. Put `Reward("fast")` into the RL loop, then send the robot's new takes back to the labeling page to catch reward
   hacking early.

---

Files: see `how_to_use_it.md` (input / what it does / output / how to run for every file).
Recordings are in the team Google Drive folder `bryan/data/local/`; slides in `bryan/sound_explain.pptx`.
