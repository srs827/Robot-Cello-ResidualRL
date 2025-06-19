import numpy as np
import pandas as pd
import mido
import time
import sys
import os 
import mujoco
from scipy.spatial.transform import Rotation as R
import torch
sys.path.append("/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Data-Files/Big-Logs")
from ik_model import IKNet

device = torch.device("cpu")
# Load trained model
ik_net = IKNet().to(device)
ik_net.load_state_dict(torch.load("ik_net.pth", map_location=device))
ik_net.eval()

# Add MujocoController to sys.path from full path
sys.path.append("/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/UR5_Sim/MuJoCo_RL_UR5/gym_grasper/controller/")
from MujocoController import MJ_Controller  # your custom MuJoCo controller with IK

# --- Load MujocoController ---
mj_ctrl = MJ_Controller("/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/UR5_Sim/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/ur5e3.xml")  # Assume this sets up model and sim internally
#mujoco.mj_step(mj_ctrl.model, mj_ctrl.data) 

# --- Define TCP poses for each bow direction on each string (from your URScript) ---
BOW_POSES = {
    "A": {"tip": np.array([.4731, .4131, .2563, -1.4605, -2.3101, 1.4458]),
           "frog": np.array([.3007, .7936, .0997, -1.5435, -2.3549, 1.3468])},
    "D": {"tip": np.array([.3404, .2802, .1763, -1.6146, -2.0448, 1.0423]),
           "frog": np.array([.3028, .7498, .1173, -1.6641, -2.0843, 1.0380])},
    "G": {"tip": np.array([.1620, .2013, .0594, -1.9298, -1.9313, .5551]),
           "frog": np.array([.2812, .6817, .1047, -1.8122, -1.9402, .4937])},
    "C": {"tip": np.array([.0798, .2852, -.0867, -1.8196, -1.6583, .1809]),
           "frog": np.array([.2567, .6101, .0626, -1.7432, -1.5245, .1638])}
}

# --- MIDI Parser ---
def parse_midi_notes(midi_path):
    mid = mido.MidiFile(midi_path)
    events = []
    time_elapsed = 0.0
    for msg in mid:
        time_elapsed += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            pitch = msg.note
            if pitch < 50:
                string = "C"
            elif pitch < 57:
                string = "G"
            elif pitch < 64:
                string = "D"
            else:
                string = "A"
            direction = "up" if (len(events) % 2 == 0) else "down"
            duration = 0.5  # default
            events.append({"time": time_elapsed, "string": string, "direction": direction, "duration": duration})
    return events

def generate_log(midi_path, output_csv):
    events = parse_midi_notes(midi_path)
    log_rows = []
    t0 = time.time()
    
    for e in events:
        t_event = e['time']
        bow_poses = BOW_POSES[e['string']]
        tcp_pose = bow_poses['tip'] if e['direction'] == 'up' else bow_poses['frog']

        # Convert rpy (assumed to be in radians) to quaternion [x, y, z, w]
        rotation = R.from_euler('xyz', tcp_pose[3:])  # assume tcp_pose[3:] is rpy
        quat = rotation.as_quat()  # returns [x, y, z, w]

        pose_target = list(tcp_pose[:3]) + list(quat)  # [x, y, z, qx, qy, qz, qw]
        joint_angles = mj_ctrl.ik_2(pose_target)
        print(f"IK for event at t={t_event}s: {joint_angles}")
        if joint_angles is None:
            print(f"⚠️ IK failed for event at t={t_event}s. Skipping...")
            continue

        log_rows.append([
            t_event,
            f"bow_{e['direction']}_{e['string']}",
            *tcp_pose,
            *joint_angles[:6],  # you may want to log all 6 or 7 joints
            0, 0, 0  # fx, fy, fz (placeholder)
        ])

    df = pd.DataFrame(log_rows, columns=[
        "timestamp", "event",
        "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz",
        "q0", "q1", "q2", "q3", "q4", "q5",
        "fx", "fy", "fz"
    ])
    df.to_csv(output_csv, index=False)
    print(f"✅ Simulated RTDE log saved to {output_csv}")



# --- Generate RTDE-style log ---
# def generate_log(midi_path, output_csv):
#     events = parse_midi_notes(midi_path)
#     log_rows = []
#     t0 = time.time()
#     for e in events:
#         t_event = e['time']
#         bow_poses = BOW_POSES[e['string']]
#         tcp_pose = bow_poses['tip'] if e['direction'] == 'up' else bow_poses['frog']

#         # TODO : use ik_2 and implement joint angles in ik_2, then change back to tcp_pose[:3] and tcp_pose[3:]
#         joint_angles = mj_ctrl.ik_2(tcp_pose[:3])  # xyz, rpy
#         log_rows.append([
#             t_event,
#             f"bow_{e['direction']}_{e['string']}",
#             *tcp_pose,
#             # will change this back to :6
#             *joint_angles[:3],
#             0, 0, 0  # fx, fy, fz
#         ])

#     df = pd.DataFrame(log_rows, columns=[
#         "timestamp", "event",
#         "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz",
#         "q0", "q1", "q2", "q3", "q4", "q5",
#         "fx", "fy", "fz"
#     ])
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Simulated RTDE log saved to {output_csv}")

if __name__ == "__main__":
    MIDI_PATH = '/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/MIDI-Files/minuet_no_2v2.mid'
    OUTPUT_CSV = "simulated_rtde_log.csv"
    generate_log(MIDI_PATH, OUTPUT_CSV)
