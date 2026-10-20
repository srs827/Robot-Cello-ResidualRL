# Overnight comparison of reward models

Same frozen human data for every run. Held-out human pairs: **57 decisive** (each touches a test clip; no model trained on them), 95% interval about +/-12 points.
`no-trick` = share of BAD takes already scored above the median good take with no audio trick (the model's ordinary error). `trick gain` = how many extra points the best of 20 cheap audio tricks adds on top (the part that is actual gaming).

## Every model

| model | agrees with human | 95% | tie-AUC | B3 robot | B4 human | B5 unseen defects | hack rate | no-trick | trick gain | hack on UNSEEN tricks |
|---|---|---|---|---|---|---|---|---|---|---|
| rl_reward | 59.6% | 47-71% | 0.62 | - | - | - | 100.0% | 56.9% | +43.1 pts | 100.0% |
| fuse_fast | 71.9% | 59-82% | 0.65 | - | - | - | 56.2% | 31.9% | +24.4 pts | 50.0% |
| m1_pairwise | 68.4% | 56-79% | 0.76 | 0.795 | 0.867 | 0.569 | 31.2% | 26.2% | +5.0 pts | 28.7% |
| pw_ctl_s1 | 64.9% | 52-76% | 0.71 | 0.803 | 0.879 | 0.599 | 32.5% | 19.4% | +13.1 pts | 27.5% |
| pw_ctl_s2 | 71.9% | 59-82% | 0.77 | 0.830 | 0.792 | 0.614 | 23.8% | 21.9% | +1.9 pts | 23.8% |
| pw_ctl_s3 | 70.2% | 57-80% | 0.75 | 0.826 | 0.818 | 0.601 | 26.9% | 23.1% | +3.7 pts | 25.6% |
| pw_v1_s0 | 63.2% | 50-74% | 0.70 | 0.749 | 0.769 | 0.519 | 43.8% | 30.6% | +13.1 pts | 36.9% |
| pw_v1_s1 | 73.7% | 61-83% | 0.71 | 0.754 | 0.905 | 0.619 | 40.0% | 21.2% | +18.8 pts | 31.9% |
| pw_v2_s2 | 75.4% | 63-85% | 0.74 | 0.772 | 0.830 | 0.608 | 28.1% | 20.6% | +7.5 pts | 25.6% |
| pw_v2_s3 | 75.4% | 63-85% | 0.69 | 0.843 | 0.822 | 0.586 | 40.6% | 18.8% | +21.9 pts | 25.6% |
| pw_ctl_s4 | 54.4% | 42-67% | 0.67 | 0.772 | 0.633 | 0.545 | 30.0% | 21.9% | +8.1 pts | 26.9% |
| pw_ctl_s5 | 71.9% | 59-82% | 0.64 | 0.831 | 0.739 | 0.607 | 48.1% | 25.0% | +23.1 pts | 43.8% |
| pw_v3_s4 | 70.2% | 57-80% | 0.72 | 0.759 | 0.686 | 0.555 | 45.0% | 21.9% | +23.1 pts | 35.6% |
| pw_v3_s5 | 77.2% | 65-86% | 0.68 | 0.796 | 0.780 | 0.589 | 35.6% | 21.2% | +14.4 pts | 23.8% |
| pw_ctl_s6 | 66.7% | 54-78% | 0.64 | 0.811 | 0.712 | 0.615 | 25.6% | 22.5% | +3.1 pts | 25.6% |
| pw_ctl_s7 | 64.9% | 52-76% | 0.69 | 0.778 | 0.924 | 0.596 | 31.9% | 22.5% | +9.4 pts | 28.7% |
| pw_v4_s6 | 66.7% | 54-78% | 0.72 | 0.792 | 0.788 | 0.610 | 32.5% | 17.5% | +15.0 pts | 24.4% |
| pw_v4_s7 | 70.2% | 57-80% | 0.75 | 0.750 | 0.784 | 0.617 | 31.2% | 16.9% | +14.4 pts | 24.4% |
| m1_pairwise+pw_ctl_s1 | 66.7% | 54-78% | 0.70 | 0.816 | 0.909 | 0.589 | 26.2% | 21.2% | +5.0 pts | 26.2% |
| pw_v1_s0+pw_v1_s1 | 75.4% | 63-85% | 0.75 | 0.768 | 0.920 | 0.575 | 36.2% | 24.4% | +11.9 pts | 30.6% |
| m1_pairwise+pw_ctl_s1+pw_ctl_s2+pw_ctl_s3 | 68.4% | 56-79% | 0.79 | 0.832 | 0.860 | 0.599 | 23.8% | 21.2% | +2.5 pts | 23.8% |
| pw_v1_s0+pw_v1_s1+pw_v2_s2+pw_v2_s3 | 75.4% | 63-85% | 0.77 | 0.811 | 0.890 | 0.606 | 28.7% | 18.1% | +10.6 pts | 23.1% |
| m1_pairwise+pw_ctl_s1+pw_ctl_s2+pw_ctl_s3+pw_v1_s0+pw_v1_s1+pw_v2_s2+pw_v2_s3 | 73.7% | 61-83% | 0.82 | 0.843 | 0.909 | 0.606 | 25.0% | 17.5% | +7.5 pts | 23.1% |
| distill8 | 66.7% | 54-78% | 0.77 | 0.835 | 0.845 | 0.578 | 28.7% | 21.9% | +6.9 pts | 25.6% |
| distill8b | 68.4% | 56-79% | 0.77 | 0.832 | 0.860 | 0.588 | 30.0% | 24.4% | +5.6 pts | 28.1% |
| m1_pairwise+pw_ctl_s1+pw_ctl_s2+pw_ctl_s3+pw_ctl_s4+pw_ctl_s5 | 68.4% | 56-79% | 0.75 | 0.841 | 0.841 | 0.607 | 29.4% | 21.2% | +8.1 pts | 28.1% |
| pw_v1_s0+pw_v1_s1+pw_v2_s2+pw_v2_s3+pw_v3_s4+pw_v3_s5 | 75.4% | 63-85% | 0.75 | 0.812 | 0.913 | 0.604 | 30.6% | 19.4% | +11.3 pts | 22.5% |
| m1_pairwise+pw_ctl_s1+pw_ctl_s2+pw_ctl_s3+pw_ctl_s4+pw_ctl_s5+pw_v1_s0+pw_v1_s1+pw_v2_s2+pw_v2_s3+pw_v3_s4+pw_v3_s5 | 77.2% | 65-86% | 0.78 | 0.849 | 0.905 | 0.611 | 25.6% | 21.9% | +3.7 pts | 24.4% |
| distill12 | 71.9% | 59-82% | 0.75 | 0.841 | 0.856 | 0.589 | 29.4% | 25.0% | +4.4 pts | 28.7% |
| m1_pairwise+pw_ctl_s1+pw_ctl_s2+pw_ctl_s3+pw_ctl_s4+pw_ctl_s5+pw_ctl_s6+pw_ctl_s7 | 70.2% | 57-80% | 0.75 | 0.835 | 0.833 | 0.609 | 25.6% | 21.2% | +4.4 pts | 25.6% |
| pw_v1_s0+pw_v1_s1+pw_v2_s2+pw_v2_s3+pw_v3_s4+pw_v3_s5+pw_v4_s6+pw_v4_s7 | 71.9% | 59-82% | 0.81 | 0.801 | 0.905 | 0.614 | 31.2% | 19.4% | +11.9 pts | 21.9% |
| m1_pairwise+pw_ctl_s1+pw_ctl_s2+pw_ctl_s3+pw_ctl_s4+pw_ctl_s5+pw_ctl_s6+pw_ctl_s7+pw_v1_s0+pw_v1_s1+pw_v2_s2+pw_v2_s3+pw_v3_s4+pw_v3_s5+pw_v4_s6+pw_v4_s7 | 75.4% | 63-85% | 0.80 | 0.842 | 0.879 | 0.613 | 24.4% | 21.9% | +2.5 pts | 23.8% |
| distill16 | 66.7% | 54-78% | 0.75 | 0.835 | 0.871 | 0.589 | 28.7% | 25.6% | +3.1 pts | 28.1% |
| ft_c_s0 | 77.2% | 65-86% | 0.75 | 0.819 | 0.845 | 0.593 | 28.1% | 21.9% | +6.2 pts | 25.6% |
| ft_c_s1 | 61.4% | 48-73% | 0.70 | 0.758 | 0.886 | 0.553 | 32.5% | 28.1% | +4.4 pts | 31.9% |
| ft_c_s2 | 66.7% | 54-78% | 0.72 | 0.799 | 0.875 | 0.565 | 30.0% | 23.8% | +6.2 pts | 28.7% |
| ft_c_s3 | 68.4% | 56-79% | 0.74 | 0.790 | 0.822 | 0.573 | 33.8% | 24.4% | +9.4 pts | 32.5% |
| ft_c_s4 | 63.2% | 50-74% | 0.72 | 0.786 | 0.833 | 0.575 | 30.6% | 23.1% | +7.5 pts | 28.7% |
| ft_c_s0+ft_c_s1+ft_c_s2 | 71.9% | 59-82% | 0.71 | 0.795 | 0.879 | 0.569 | 29.4% | 23.8% | +5.6 pts | 28.7% |
| ft_h_s0 | 68.4% | 56-79% | 0.73 | 0.807 | 0.898 | 0.588 | 31.2% | 19.4% | +11.9 pts | - |
| ft_h_s1 | 63.2% | 50-74% | 0.65 | 0.777 | 0.856 | 0.554 | 28.7% | 18.8% | +10.0 pts | - |
| ft_h_s2 | 68.4% | 56-79% | 0.71 | 0.774 | 0.883 | 0.576 | 38.8% | 23.8% | +15.0 pts | - |
| ft_h_s3 | 64.9% | 52-76% | 0.69 | 0.759 | 0.780 | 0.578 | 31.9% | 20.0% | +11.9 pts | 26.9% |
| ft_h_s4 | 68.4% | 56-79% | 0.71 | 0.783 | 0.886 | 0.586 | 30.0% | 20.6% | +9.4 pts | 25.6% |
| ft_h_s0+ft_h_s1+ft_h_s2 | 66.7% | 54-78% | 0.70 | 0.789 | 0.898 | 0.570 | 31.2% | 18.1% | +13.1 pts | - |
| ft_i2_s0 | 68.4% | 56-79% | 0.72 | 0.796 | 0.841 | 0.583 | 29.4% | 22.5% | +6.9 pts | 26.9% |
| ft_i2_s1 | 80.7% | 69-89% | 0.76 | 0.811 | 0.841 | 0.614 | 35.0% | 22.5% | +12.5 pts | 28.7% |

## Fine-tune experiments, averaged over seeds

Every fine-tune starts from the same `m1_pairwise` (400 steps, lr 5e-4). The **control** arm has no human labels; the other arms differ only in what is added. `vs control` = average paired difference on the same held-out pairs (mean +/- standard error over seeds).

| arm | seeds | agrees with human | vs control | B3 | B4 | trick gain | hack on UNSEEN tricks |
|---|---|---|---|---|---|---|---|
| control: fine-tune, NO human labels | 5 | 67.4% (seed spread 61%-77%) | - | 0.791 | 0.852 | +6.8 pts | 29.5% |
| + human labels (ties as soft labels, weight 2.5) | 5 | 66.7% (seed spread 63%-68%) | -0.7 +/- 2.5 pts (n=5) | 0.780 | 0.861 | +11.6 pts | 26.2% |
| robustness training x2 (no human labels) | 2 | 74.6% (seed spread 68%-81%) | +5.3 +/- 14.0 pts (n=2) | 0.803 | 0.841 | +9.7 pts | 27.8% |

## Paired comparisons on the identical held-out pairs (exact McNemar test)

`only A right` / `only B right` = pairs where exactly one of the two models matched the human; only those carry information.

| question | A | B | A minus B | only A right | only B right | p |
|---|---|---|---|---|---|---|
| ALL 8 from-scratch models averaged vs a single model (m1_pairwise) | m1_pairwise+.. 73.7% | m1_pairwise 68.4% | +5.3 pts | 6 | 3 | 0.51 |
| ALL 8 averaged vs the CURRENT RL reward | m1_pairwise+.. 73.7% | rl_reward 59.6% | +14.0 pts | 16 | 8 | 0.15 |
| 4 models WITH human labels vs 4 models WITHOUT (each averaged) | pw_v1_s0+.. 75.4% | m1_pairwise+.. 68.4% | +7.0 pts | 6 | 2 | 0.29 |
| 4 models WITH human labels vs the CURRENT RL reward | pw_v1_s0+.. 75.4% | rl_reward 59.6% | +15.8 pts | 16 | 7 | 0.09 |
| distilled single model vs the 8-model ensemble it imitates | distill8 66.7% | m1_pairwise+.. 73.7% | -7.0 pts | 1 | 5 | 0.22 |
| distilled single model vs the CURRENT RL reward | distill8 66.7% | rl_reward 59.6% | +7.0 pts | 17 | 13 | 0.58 |
| distilled v2 (real-window weighted) vs the 8-model ensemble it imitates | distill8b 68.4% | m1_pairwise+.. 73.7% | -5.3 pts | 0 | 3 | 0.25 |
| distilled v2 vs the CURRENT RL reward | distill8b 68.4% | rl_reward 59.6% | +8.8 pts | 16 | 11 | 0.44 |
| from scratch, WITH human labels vs the original m1_pairwise | pw_v1_s0 63.2% | m1_pairwise 68.4% | -5.3 pts | 5 | 8 | 0.58 |
| from scratch seed 1: WITH vs WITHOUT human labels | pw_v1_s1 73.7% | pw_ctl_s1 64.9% | +8.8 pts | 13 | 8 | 0.38 |
| seed noise alone: original m1_pairwise vs its seed-1 twin | m1_pairwise 68.4% | pw_ctl_s1 64.9% | +3.5 pts | 10 | 8 | 0.81 |
| from-scratch ensemble (human labels) vs from-scratch ensemble (none) | pw_v1_s0+.. 75.4% | m1_pairwise+.. 66.7% | +8.8 pts | 10 | 5 | 0.30 |
| from-scratch ensemble (human labels) vs the CURRENT RL reward | pw_v1_s0+.. 75.4% | rl_reward 59.6% | +15.8 pts | 15 | 6 | 0.08 |
| 3-seed fine-tune ensemble WITH human labels vs WITHOUT | ft_h_s0+.. 66.7% | ft_c_s0+.. 71.9% | -5.3 pts | 3 | 6 | 0.51 |
| original m1_pairwise vs the CURRENT RL reward | m1_pairwise 68.4% | rl_reward 59.6% | +8.8 pts | 18 | 13 | 0.47 |
| fine-tune control ensemble (3 seeds) vs the CURRENT RL reward | ft_c_s0+.. 71.9% | rl_reward 59.6% | +12.3 pts | 19 | 12 | 0.28 |
| 12-model ensemble vs the 8-model ensemble (grown with 4 more seeds) | m1_pairwise+.. 77.2% | m1_pairwise+.. 73.7% | +3.5 pts | 2 | 0 | 0.50 |
| 12-model ensemble vs the CURRENT RL reward | m1_pairwise+.. 77.2% | rl_reward 59.6% | +17.5 pts | 17 | 7 | 0.06 |
| 6 models WITH human labels vs 6 WITHOUT (12-model split) | pw_v1_s0+.. 75.4% | m1_pairwise+.. 68.4% | +7.0 pts | 6 | 2 | 0.29 |
| distilled v3 (12-model teacher) vs distilled v2 (8-model teacher) | distill12 71.9% | distill8b 68.4% | +3.5 pts | 2 | 0 | 0.50 |
| distilled v3 vs the 12-model ensemble it imitates | distill12 71.9% | m1_pairwise+.. 77.2% | -5.3 pts | 0 | 3 | 0.25 |
| distilled v3 vs the CURRENT RL reward | distill12 71.9% | rl_reward 59.6% | +12.3 pts | 16 | 9 | 0.23 |
| 16-model ensemble vs the 12-model ensemble (grown with 4 more seeds) | m1_pairwise+.. 75.4% | m1_pairwise+.. 77.2% | -1.8 pts | 0 | 1 | 1.00 |
| 16-model ensemble vs the CURRENT RL reward | m1_pairwise+.. 75.4% | rl_reward 59.6% | +15.8 pts | 17 | 8 | 0.11 |
| 8 models WITH human labels vs 8 WITHOUT (16-model split) | pw_v1_s0+.. 71.9% | m1_pairwise+.. 70.2% | +1.8 pts | 3 | 2 | 1.00 |
| distilled v4 (16-model teacher) vs distilled v3 (12-model teacher) | distill16 66.7% | distill12 71.9% | -5.3 pts | 0 | 3 | 0.25 |
| distilled v4 vs the 16-model ensemble it imitates | distill16 66.7% | m1_pairwise+.. 75.4% | -8.8 pts | 0 | 5 | 0.06 |

A p-value above 0.05 means the two models are not distinguishable on this many pairs. Do not read winners into gaps that are inside the noise.
