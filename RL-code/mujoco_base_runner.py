import time
import pandas as pd
from parsemidi import parse_midi
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from rl_trajectory import UR5eCelloTrajectoryEnv

a_frog = [0.43487182,  0.39604015, -0.213195,   -1.37183167, -2.21080519,  1.27344916]
a_tip = [0.40931818,  0.69494985, -0.213195,   -1.37183167, -2.21080519,  1.27344916]
d_frog = [0.33455062,  0.31949893, -0.24925,    -1.5619303,  -1.97043579,  0.98332491]
d_tip = [0.34919938,  0.61914107, -0.24925,    -1.5619303,  -1.97043579,  0.98332491]
g_frog = [0.27658556,  0.26608705, -0.31557,    -1.51549779, -1.6723598,   0.7564576 ]
g_tip = [0.29907444,  0.56524295, -0.31557,    -1.51549779, -1.6723598,   0.7564576 ]
c_frog = [0.20489384,  0.23316687, -0.37714,    -1.55305625, -1.45776501,  0.41142248]
c_tip = [0.23001616,  0.53211313, -0.37714,    -1.55305625, -1.45776501,  0.41142248]




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


def generate_real_trajectory(note_sequence):
    for note in note_sequence:
        if note['note'] == 'a_bow':


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
