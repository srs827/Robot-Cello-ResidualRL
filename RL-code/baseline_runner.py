import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import torch
import sys, os
sys.path.append("/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Data-Files/Big-Logs")
from ik_nn import IKNet  # ensure this file is in the same directory or in PYTHONPATH
#from parsemidi import parse_midi
from robot_runner_detailed_logs import BOW_POSES
from robot_runner_detailed_logs import parse_midi
from rl_trajectory import UR5eCelloTrajectoryEnv
import mujoco


sys.path.append(os.path.dirname(os.path.abspath(
    "/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/UR5_Sim/MuJoCo_RL_UR5/gym_grasper/controller/MujocoController.py"
)))
from MujocoController import MJ_Controller

device = torch.device("cpu")
ik_net = IKNet().to(device)
ik_net.load_state_dict(torch.load("ik_net.pth", map_location=device))
ik_net.eval()

# def interpolate_pose(p1, p2, alpha):
#     pos = (1 - alpha) * np.array(p1[:3]) + alpha * np.array(p2[:3])
#     r1 = R.from_euler('xyz', p1[3:])
#     r2 = R.from_euler('xyz', p2[3:])
#     slerp = Slerp([0, 1], R.concatenate([r1, r2]))
#     rot = slerp([alpha])[0]
#     return list(pos) + list(rot.as_euler('xyz'))

def ease_scale(note_dur):
    return min(1.0, note_dur / 3.0)

def generate_trajectory(events, controller, sim_dt=0.01):
    trajectory_cartesian = []

    def get_direction_vector(p1, p2):
        return (np.array(p2[:3]) - np.array(p1[:3])) / np.linalg.norm(np.array(p2[:3]) - np.array(p1[:3]))

    def compute_string_crossing(start_bow_poses, end_bow_poses, last_pose):
        start_vec = get_direction_vector(start_bow_poses['frog'], start_bow_poses['tip'])
        end_vec = get_direction_vector(end_bow_poses['frog'], end_bow_poses['tip'])
        out_vector = np.cross(start_vec, end_vec)
        out_vector /= np.linalg.norm(out_vector)
        out_step = np.array(last_pose[:3]) + out_vector * 0.03
        dist_along_start = np.linalg.norm(np.array(last_pose[:3]) - np.array(start_bow_poses['frog'])[:3])
        bow_len_start = np.linalg.norm(np.array(start_bow_poses['tip'])[:3] - np.array(start_bow_poses['frog'])[:3])
        frac_from_frog = dist_along_start / bow_len_start
        dir_vec = get_direction_vector(end_bow_poses['frog'], end_bow_poses['tip'])
        target_xyz = np.array(end_bow_poses['frog'])[:3] + dir_vec * frac_from_frog * np.linalg.norm(np.array(end_bow_poses['tip'])[:3] - np.array(end_bow_poses['frog'])[:3])
        target_rot = np.array(end_bow_poses['frog'])[3:]
        step1 = list(out_step) + last_pose[3:]
        step2 = list(out_step) + list(target_rot)
        final  = list(target_xyz) + list(target_rot)
        return [step1, step2, final]

    last_pose = None

    for e in events:
        if e["is_transition"]:
            current_str, next_str = e["string"].split("-")
            start_bow_poses = BOW_POSES[current_str]
            end_bow_poses = BOW_POSES[next_str]
            if last_pose is None:
                continue
            crossing_poses = compute_string_crossing(start_bow_poses, end_bow_poses, last_pose)
            for pose in crossing_poses:
                trajectory_cartesian.append(pose)
                last_pose = pose
            continue

        duration = min(e['duration_sec'], 3.0)
        scale = ease_scale(duration)
        direction = e['bowing'] == "down"
        bow_poses = BOW_POSES[e['string']]
        tip, frog = bow_poses['tip'], bow_poses['frog']
        start_p = last_pose if last_pose is not None else (frog if direction else tip)
        end_p = tip if direction else frog
        dir_vec = get_direction_vector(frog, tip)
        t_dir = 1 if direction else -1
        bow_len = np.linalg.norm(np.array(tip[:3]) - np.array(frog[:3]))
        target_dist = scale * bow_len
        dist_to_end = np.linalg.norm(np.array(end_p[:3]) - np.array(start_p[:3]))
        poses_to_add = []
        if dist_to_end >= target_dist:
            final_pos = np.array(start_p[:3]) + t_dir * dir_vec * target_dist
            rot = np.array(start_p[3:])
            poses_to_add.append(list(final_pos) + list(rot))
        elif abs(dist_to_end - target_dist) <= 0.025:
            poses_to_add.append(end_p)
        else:
            d2 = target_dist - dist_to_end
            poses_to_add.append(end_p)
            extra_pos = np.array(end_p[:3]) + -1 * t_dir * dir_vec * d2
            extra_rot = np.array(end_p[3:])
            poses_to_add.append(list(extra_pos) + list(extra_rot))

        for pose in poses_to_add:
            steps = max(1, int(duration / sim_dt / len(poses_to_add)))
            for _ in range(steps):
                trajectory_cartesian.append(pose)
                last_pose = pose

    trajectory_joint_space = []

    for i, pose in enumerate(trajectory_cartesian):
        if len(pose) < 6:
            print(f"❌ Skipping invalid pose at step {i}: insufficient data {pose}")
            continue

        xyz = pose[:3]
        rpy = pose[3:]

        try:
            rotvec = R.from_euler('xyz', rpy).as_rotvec()
            input_data = xyz + rotvec.tolist()
            if len(input_data) != 6:
                raise ValueError(f"Expected 6 input values but got {len(input_data)}: {input_data}")
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        except Exception as e:
            print(f"❌ Skipping pose at step {i} due to error: {e}")
            continue

        print("input_tensor shape:", input_tensor.shape)

        with torch.no_grad():
            pred_q = ik_net(input_tensor).cpu().numpy().flatten().tolist()
        trajectory_joint_space.append(pred_q)

    return trajectory_joint_space

def simulate_trajectory(trajectory, model_path, sim_dt=0.01):
    if len(trajectory) == 0:
        print("❌ No valid joint-space trajectory was generated.")
        return
    env = UR5eCelloTrajectoryEnv(
        model_path=model_path,
        trajectory=trajectory,
        note_sequence=[],
        render_mode='human',
        action_scale=0.0,
        residual_penalty=0.0,
        contact_penalty=0.0,
        torque_penalty=0.0, 
        kp=0.0, kd=0.0, ki=0.0,
        start_joint_positions=trajectory[0]
    )

    for joint_angles in trajectory:
        env.data.qpos[:6] = joint_angles
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.render()
        time.sleep(sim_dt)

    env.close()
    print("✅ Simulation finished.")

def main():
    midi_file = "/Users/skamanski/Documents/GitHub/Robot-Cello/midi_robot_pipeline/midi_files/minuet_no_2v2.mid"
    model_path = "/Users/skamanski/Documents/GitHub/Robot-Cello/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/scene.xml"
    events = parse_midi(midi_file)
    print(events)
    controller = MJ_Controller(model_path, viewer=False)
    print("⏳ Precomputing trajectory...")
    trajectory = generate_trajectory(events, controller)
    print(f"✅ Trajectory generated with {len(trajectory)} poses.")
    simulate_trajectory(trajectory, model_path)

if __name__ == "__main__":
    main()