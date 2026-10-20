"""Run the whole pipeline: download data, build dataset, train 12 models, distil, score references, evaluate, explain."""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGES = ["download", "dataset", "train", "distill", "reference", "evaluate", "analysis"]
TRAIN = [("m1_pairwise", 0, False), ("pw_ctl_s1", 1, False), ("pw_ctl_s2", 2, False), ("pw_ctl_s3", 3, False),
         ("pw_ctl_s4", 4, False), ("pw_ctl_s5", 5, False), ("pw_v1_s0", 0, True), ("pw_v1_s1", 1, True),
         ("pw_v2_s2", 2, True), ("pw_v2_s3", 3, True), ("pw_v3_s4", 4, True), ("pw_v3_s5", 5, True)]
DISTILL = [("distill8", "ensemble8", "m1_pairwise", "mixed", 160, 6, 1e-3),
           ("distill8b", "ensemble8", "distill8", "real", 200, 4, 5e-4),
           ("distill12", "ensemble12", "distill8b", "real", 200, 4, 5e-4)]


def run(script, *args):
    """Run one script with the same Python; stop if it fails."""
    cmd = [sys.executable, os.path.join(ROOT, script), *map(str, args)]
    print(">>", " ".join(cmd), flush=True)
    if subprocess.call(cmd) != 0:
        sys.exit(f"failed: {script}")


def main():
    """Run the selected stages in order; finished models are skipped so it can be restarted."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES), help="comma list from: " + ",".join(STAGES))
    ap.add_argument("--model_dir", default=os.path.join(ROOT, "retrained"), help="where new models go (shipped ones stay in checkpoints/)")
    ap.add_argument("--steps", type=int, default=2000, help="training steps per model")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.environ["SOUND_MODEL_DIR"] = a.model_dir
    md = a.model_dir
    stages = a.stages.split(",")
    if "download" in stages:
        run("data/download_public.py")
    if "dataset" in stages:
        run("data/build_dataset.py")
    if "train" in stages:
        for name, seed, human in TRAIN:
            if not os.path.exists(os.path.join(md, f"{name}_seed{seed}.pt")):
                run("training/train_reward.py", "--name", name, "--seed", seed, "--steps", a.steps, "--out", md,
                    "--workers", a.workers, *([] if human else ["--human_pairs", "none"]))
    if "distill" in stages:
        for name, teachers, init, mode, nb, ep, lr in DISTILL:
            if not os.path.exists(os.path.join(md, f"{name}_seed0.pt")):
                run("training/train_distill.py", "--name", name, "--teachers", teachers, "--init", init, "--mode", mode,
                    "--n_batches", nb, "--epochs", ep, "--lr", lr, "--models", md, "--out", md, "--workers", a.workers)
    if "reference" in stages:
        run("reward/build_reference.py", "--model_dir", md)
    if "evaluate" in stages:
        run("evaluation/evaluate.py", "--model_dir", md, "--out", os.path.join(ROOT, "outputs", "evaluation_retrained"))
    if "analysis" in stages:
        out = os.path.join(ROOT, "outputs", "explain_retrained")
        run("analysis/explain_features.py", "--model_dir", md, "--out", out)
        run("analysis/explain_position.py", "--model_dir", md, "--out", out)
        run("analysis/example_outputs.py", "--model_dir", md, "--out", os.path.join(ROOT, "outputs", "examples_retrained"))


if __name__ == "__main__":
    main()
