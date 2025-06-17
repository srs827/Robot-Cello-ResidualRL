import time
import pandas as pd
import mujoco
from rl_trajectory import UR5eCelloTrajectoryEnv
# it actually works kinda crazy 
def extract_joint_angles_and_timestamps(csv_filename):
    df = pd.read_csv(csv_filename)
    joint_cols = ['q_base','q_shoulder','q_elbow','q_wrist1','q_wrist2','q_wrist3']
    return df[joint_cols].values.tolist(), df["timestamp_robot"].values.tolist()

def main():
    trajectory, timestamps = extract_joint_angles_and_timestamps(
        '/Users/skamanski/Documents/GitHub/Robot-Cello/biglogs/minuet_no_2v2-log-detailed.csv'
    )

    scene_path = '/Users/skamanski/Documents/GitHub/Robot-Cello/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/scene.xml'

    env = UR5eCelloTrajectoryEnv(
        model_path=scene_path,
        trajectory=trajectory,
        note_sequence=[],             # not needed for replay
        render_mode='human',
        action_scale=0.0,
        residual_penalty=0.0,
        contact_penalty=0.0,
        torque_penalty=0.0,
        kp=0.0, kd=0.0, ki=0.0,
        start_joint_positions=trajectory[0]
    )

    obs = env.reset()
    start_wall = time.time()
    start_log = timestamps[0]

    for i, joint_angles in enumerate(trajectory):
        if i >= len(timestamps):
            break

        # Wait until correct real-time offset
        elapsed_target = timestamps[i] - start_log
        elapsed_actual = time.time() - start_wall
        wait_time = elapsed_target - elapsed_actual
        if wait_time > 0:
            time.sleep(wait_time)

        # Override simulation state
        env.data.qpos[:6] = joint_angles
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.render()

    env.close()

if __name__ == "__main__":
    main()