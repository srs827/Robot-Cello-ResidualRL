import time
import pandas as pd
import mido
from parsemidi import parse_midi
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from rl_trajectory import UR5eCelloTrajectoryEnv
from mido import MidiFile

import numpy as np
# Define bowing poses for frog and tip on each string
waypoints = {
 'A': {
        'frog': np.array([0.34398, 0.75769, -0.27971, 1.640, 2.582, -1.575]),
        'tip':  np.array([0.50021, 0.33330, -0.14668, 1.567, 2.584, -1.404])
    },
    'D': {
        'frog': np.array([0.32866, 0.70480, -0.26920, 2.109, 2.598, -1.301]),
        'tip':  np.array([0.35509, 0.23384, -0.22930, 2.037, 2.632, -1.309])
    },
    'G': {
        'frog': np.array([0.34391, 0.64827, -0.29245, 2.450, 2.761, -1.313]),
        'tip':  np.array([0.23175, 0.18306, -0.33869, 2.518, 2.722, -1.168])
    },
    'C': {
        'frog': np.array([0.29761, 0.59019, -0.31737, 3.016, 2.756, -0.651]),
        'tip':  np.array([0.13730, 0.17509, -0.43691, 2.871, 2.768, -0.905])
    }
}



def generate_trajectory(note_events):
    trajectory = []
    last_pose = None

    for note in note_events:
        if note['note'] == 'transition':
            str_from, str_to = note['string'].split('-')
            if str_from in waypoints and str_to in waypoints and last_pose is not None:
                # Calculate position on start and end bow
                bow_from = waypoints[str_from]
                bow_to = waypoints[str_to]

                start_vec = bow_from['tip'][:3] - bow_from['frog'][:3]
                end_vec = bow_to['tip'][:3] - bow_to['frog'][:3]
                start_len = np.linalg.norm(start_vec)
                end_len = np.linalg.norm(end_vec)

                start_dir = start_vec / start_len
                end_dir = end_vec / end_len

                rel_to_frog = last_pose[:3] - bow_from['frog'][:3]
                t = np.clip(np.dot(rel_to_frog, start_dir), 0.0, start_len)

                proj_start = bow_from['frog'][:3] + t * start_dir
                proj_end = bow_to['frog'][:3] + t * end_dir

                pose_start = np.concatenate([proj_start, bow_from['frog'][3:]])
                pose_end = np.concatenate([proj_end, bow_to['frog'][3:]])

                trajectory.append(pose_start)
                trajectory.append(pose_end)
                last_pose = pose_end
        else:
            direction = note['bowing']
            string = note['string']
            if string in waypoints:
                pose = waypoints[string]['tip' if direction == 'up' else 'frog']
                trajectory.append(pose)
                last_pose = pose

    return trajectory

def get_note_name(note_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note = note_names[note_number % 12]
    return f"{note}{octave}"

def read_bowing_file(bowing_file):
    bowing_dict = {}
    with open(bowing_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:
                index = int(parts[0])
                bowing_dict[index] = parts[1].strip("'")  
    return bowing_dict

def parse_midi(file_path, bowing_file, clef="bass"):
    midi = MidiFile(file_path)
    note_events = []
    #bowing_file = "/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Pieces-Bowings/allegro_bowings.txt"
    bowing_dict = read_bowing_file(bowing_file)

    def get_cello_string(note_number):
        if note_number >= 57:
            return 'A'
        elif note_number >= 50:
            return 'D'
        elif note_number >= 43:
            return 'G'
        else:
            return 'C'

    last_bow = "down"  
    index = 0  

    for track in midi.tracks:
        raw_notes = []
        active_notes = {}
        current_time = 0

        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = current_time
            elif msg.type in ('note_off', 'note_on') and msg.note in active_notes:
                start_time = active_notes.pop(msg.note)
                duration = (current_time - start_time) / midi.ticks_per_beat
                mapping_note = msg.note - 12 if clef == "tenor" else msg.note
                string = get_cello_string(mapping_note)

                if index in bowing_dict:
                    bowing = bowing_dict[index]
                    if "-s" in bowing:  
                        bowing = last_bow + "-s"
                    else:
                        last_bow = bowing  
                else:
                    bowing = "up" if last_bow == "down" else "down"
                    last_bow = bowing

                raw_notes.append({
                    'number': msg.note,
                    'note': get_note_name(msg.note),
                    'duration': duration,
                    'string': string,
                    'start_time': start_time,
                    'end_time': current_time,
                    'bowing': bowing
                })
                index += 1  
        #print(raw_notes)
        for i, note in enumerate(raw_notes):
            current_string = note['string']
            next_string = raw_notes[i + 1]['string'] if i + 1 < len(raw_notes) else None
            note_events.append(note)

            if next_string and next_string != current_string:
                note_events.append({
                    'number': 'transition',
                    'note': "transition",
                    'duration': 0.2,
                    'string': f"{current_string}-{next_string}",
                    'start_time': note['end_time'],
                    'end_time': note['end_time'] + 0.2,
                    'bowing': "transition"
                })

    return note_events

def get_start_pos(note_events):
    # Get the first note's string and return the corresponding start position
    if note_events:
        first_note = note_events[0]
        string = first_note['string']
        if string in waypoints:
            return waypoints[string]['frog']
    return np.zeros(6)  # Default start position if no notes are found


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
    note_events = parse_midi('/Users/skamanski/Documents/GitHub/Robot-Cello/midi_robot_pipeline/midi_files/allegro.mid', '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Pieces-Bowings/allegro_bowings.txt')
    print("Parsed note events:", note_events)
    trajectory = generate_trajectory(note_events)
    print(trajectory)
    # Define MuJoCo scene XML
    scene_path = '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/UR5-Sim/universal_robots_ur5e/scene.xml'
    

    def make_env():
        return UR5eCelloTrajectoryEnv(
            model_path=scene_path,
            trajectory=trajectory,
            note_sequence=note_events,
            render_mode='human',
            action_scale=0.05,
            residual_penalty=0.02,
            contact_penalty=0.1,
            torque_penalty=0.001,
            kp=100.0, kd=2.0, ki=0.1,
            start_joint_positions=get_start_pos(note_events)
        )
    print("Trajectory len:", len(trajectory))
    print("Note sequence len:", len(note_events))
    env = make_env()
    obs = env.reset()
    done = False
    step_limit = len(trajectory) + 200

    while not done and step_limit > 0:
        try:
            action = env.action_space.sample()  # Replace with model.predict(obs) if needed

            # Check for NaNs in action
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                print("[ERROR] Invalid action detected (NaN or Inf).")
                break

            # Step the environment
            obs, reward, done, info = env.step(action)

            # Check observation integrity
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print("[ERROR] Invalid observation (NaN or Inf).")
                break

            # Log some useful info
            print(f"Step {len(trajectory) + 200 - step_limit}: reward={reward:.4f}")
            if 'tcp_pose' in info:
                print(f"TCP: {info['tcp_pose']}")

            # Optional: check if joints are out of limits
            if hasattr(env, "sim"):
                joint_positions = env.sim.data.qpos[:6]
                joint_ranges = env.sim.model.jnt_range[:6]
                for i in range(6):
                    if joint_positions[i] < joint_ranges[i][0] or joint_positions[i] > joint_ranges[i][1]:
                        print(f"[WARNING] Joint {i} out of range: {joint_positions[i]:.4f} not in {joint_ranges[i]}")

            env.render()
            time.sleep(0.1)
            step_limit -= 1

        except Exception as e:
            print("[EXCEPTION]", e)
            break

    env.close()
    print("Simulation complete.")

if __name__ == '__main__':
    main()
