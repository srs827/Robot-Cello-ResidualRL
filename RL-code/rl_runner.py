import time
import sys, importlib
sys.modules['numpy._core.numeric'] = importlib.import_module('numpy.core.numeric')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback       
import mujoco.viewer                                              

from rl_trajectory import UR5eCelloTrajectoryEnv
from parsemidi import parse_midi
import pandas as pd

# gets list of joint angles from csv file
def extract_joint_angles(csv_filename):
    df = pd.read_csv(csv_filename)
    # from rtde collection actual_q which is joint positions 
    cols = ['q_base','q_shoulder','q_elbow','q_wrist1','q_wrist2','q_wrist3']
    return df[cols].values.tolist()


# ─── Simple callback to refresh the viewer ─────────────────────────
class RenderCallback(BaseCallback):
    def __init__(self, every_n_steps: int = 1):
        super().__init__()
        self.every_n_steps = every_n_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps == 0:
            self.training_env.render()
        return True
# ───────────────────────────────────────────────────────────────────

def main():
    # we get the q_base, q_shoulder, q_elbow, q_wrist1, q_wrist2, q_wrist3
    trajectory = extract_joint_angles(
        '/Users/skamanski/Documents/GitHub/Robot-Cello/biglogs/minuet_no_2v2-log-detailed.csv'
    )
    # midi file for this trajectory
    note_sequence = parse_midi(
        '/Users/skamanski/Documents/GitHub/Robot-Cello/midi_robot_pipeline/midi_files/minuet_no_2v2.mid'
    )
    scene_path = (
        '/Users/skamanski/Documents/GitHub/Robot-Cello/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/scene.xml'
    )
    start_pos = trajectory[0]

    # create residual-RL env
    def make_env():
        return UR5eCelloTrajectoryEnv(
            model_path=scene_path,
            trajectory=trajectory,
            note_sequence=note_sequence,
            render_mode=None,          # 'human' for live rendering, None for no rendering
            action_scale=0.05,
            residual_penalty=0.02,
            contact_penalty=0.1,
            torque_penalty=0.001,
            kp=100.0, kd=2.0, ki=0.1,
            start_joint_positions=start_pos
        )
    
    
    #Test Code
    # ----------------
    model_path = '/Users/skamanski/Documents/GitHub/Robot-Cello/rl/ppo_residual_ur5e.zip'
    model = PPO.load(model_path)
    env = make_env()
    obs = env.reset()
    done = False
    total_reward = 0.0
    step_limit = len(trajectory) + 200       # safety cap

    while not done and step_limit > 0:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.01)                     # slow down for visibility
        step_limit -= 1

    env.close()
    print(f'Episode finished | reward = {total_reward:.2f}')
    # _______________________________

    # Training Code
    # ----------------------
    env = DummyVecEnv([make_env])

    # # PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_residual/")

    # train *with live rendering*
    model.learn(
        total_timesteps=200_000,
        callback=RenderCallback(every_n_steps=1)   # ← THE ONLY NEW ARGUMENT
    )
    model.save("ppo_residual_ur5e")


if __name__ == '__main__':
    main()
