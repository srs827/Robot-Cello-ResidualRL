import time
import pandas as pd
from parsemidi import parse_midi
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from rl_trajectory import UR5eCelloTrajectoryEnv


class RenderCallback(BaseCallback):
    def __init__(self, every_n_steps: int = 1):
        super().__init__()
        self.every_n_steps = every_n_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps == 0:
            self.training_env.render()
        return True

def generate_fake_trajectory(note_sequence, start_joint_positions):
    # For simulation only: create placeholder joint trajectories based on note count
    return [start_joint_positions for _ in range(len(note_sequence) + 1)]

def main():
    # Load and parse MIDI file
    note_sequence = parse_midi(
        '/Users/skamanski/Documents/GitHub/Robot-Cello/midi_robot_pipeline/midi_files/allegro.mid'
    )

    # Starting joint configuration (UR5e 6-DOF)
    start_pos = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]

    # Generate synthetic trajectory for testing
    trajectory = generate_fake_trajectory(note_sequence, start_pos)

    # Define MuJoCo scene XML
    scene_path = '/Users/skamanski/Documents/GitHub/Robot-Cello/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/scene.xml'
    

    def make_env():
        return UR5eCelloTrajectoryEnv(
            model_path=scene_path,
            trajectory=trajectory,
            note_sequence=note_sequence,
            render_mode='human',
            action_scale=0.05,
            residual_penalty=0.02,
            contact_penalty=0.1,
            torque_penalty=0.001,
            kp=100.0, kd=2.0, ki=0.1,
            start_joint_positions=start_pos
        )
    print("Trajectory len:", len(trajectory))
    print("Note sequence len:", len(note_sequence))
    env = make_env()
    obs = env.reset()
    done = False
    step_limit = len(trajectory) + 200

    while not done and step_limit > 0:
        action = env.action_space.sample()  # Replace with model.predict(obs) if you have a policy
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.01)
        step_limit -= 1

    env.close()
    print("Simulation complete.")

if __name__ == '__main__':
    main()
