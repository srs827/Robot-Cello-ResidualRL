import time
import numpy as np
from parsemidi import parse_midi
from rl_trajectory import UR5eCelloTrajectoryEnv
from mujoco_base_runner import generate_trajectory, get_start_pos  # Make sure these are accessible or move them to a helper module

def main():
    # --- Load MIDI & parse note events ---
    midi_file = '/Users/skamanski/Documents/GitHub/Robot-Cello/midi_robot_pipeline/midi_files/allegro.mid'
    bowing_file = '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Pieces-Bowings/allegro_bowings.txt'
    note_events = parse_midi(midi_file, bowing_file)

    # --- Generate baseline trajectory (ideal TCP poses) ---
    trajectory = generate_trajectory(note_events)  # list of TCP poses (frog→tip motions and string crossings)

    # --- Set up environment ---
    scene_path = '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/UR5-Sim/universal_robots_ur5e/scene.xml'
    start_pos = get_start_pos(note_events)

    env = UR5eCelloTrajectoryEnv(
        model_path=scene_path,
        trajectory=trajectory,
        note_sequence=note_events,
        render_mode="human",          # Set to None to disable visualization
        action_scale=0.05,
        residual_penalty=0.02,
        contact_penalty=0.1,
        torque_penalty=0.001,
        kp=100.0, kd=2.0, ki=0.1,
        start_joint_positions=start_pos
    )

    # --- Step through the trajectory ---
    obs = env.reset()
    done = False
    step = 0
    step_limit = len(trajectory)

    while not done and step < step_limit:
        # Residual action is zero → just follow baseline
        action = np.zeros(env.action_space.shape)

        obs, reward, done, info = env.step(action)

        print(f"[{step}] reward={reward:.4f}")
        if 'tcp_pose' in info:
            print("TCP Pose:", info['tcp_pose'])

        env.render()
        time.sleep(0.05)
        step += 1

    env.close()
    print("Baseline trajectory execution finished.")

if __name__ == "__main__":
    main()