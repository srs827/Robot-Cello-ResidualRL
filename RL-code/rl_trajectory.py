import gym
from gym import spaces
import numpy as np
import mujoco
import contact
from parsemidi import parse_midi
import pandas as pd
import time
import sys
import os
import mujoco.viewer 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Baseline-Runners')))
import robot_runner  # baseline controller


class UR5eCelloTrajectoryEnv(gym.Env):
    """
    Residual-RL environment: learns to align the robot
    end-effector (TCP) to ideal linear bow paths rather
    than simply following a baseline trajectory.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        model_path: str,
        trajectory: list,
        note_sequence: list,
        render_mode=None,
        action_scale: float = 0.05,
        residual_penalty: float = 0.01,
        contact_penalty: float = 0.1,
        torque_penalty: float = 0.001,
        kp: float = 100.0,
        kd: float = 2.0,
        ki: float = 0.1,
        start_joint_positions=None,
    ):
        super().__init__()
        # --- MuJoCo setup ---
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = self.model.opt.timestep

        # --- RL hyperparameters ---
        self.action_scale = action_scale
        self.residual_penalty = residual_penalty
        self.contact_penalty = contact_penalty
        self.torque_penalty = torque_penalty

        # --- PID gains ---
        self.kp, self.kd, self.ki = kp, kd, ki
        self.total_pid_error = np.zeros(6)

        self.base_ctrl = robot_runner.CelloController(
            model_path,
            trajectory=trajectory,
            note_sequence=note_sequence,
            start_positions=start_joint_positions,
        )
        # Store provided trajectory purely as an initial demonstration
        self.demo_traj = np.array(trajectory)

        # --- Musical notes & string mapping ---
        self.note_sequence = note_sequence


        # --- Observation & Action spaces ---
        obs_dim = 6 + 6 + 4  # qpos, qvel, contact flags for strings
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        # --- Tracking ---
        self.prev_torque = np.zeros(6)
        self.current_time = 0.0
        self.current_idx = 0
        self.total_duration = len(self.demo_traj) * self.sim_dt
        self.start_positions = start_joint_positions

        # site IDs for frog/tip of each string
        self.string_sites = {}
        for s in ('A','D','G','C'):
            frog = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_frog')
            tip  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_tip')
            self.string_sites[s] = (frog, tip)
        print(f'String sites: {self.string_sites}')

        # TODO : from each string site, compute the ideal string line
        self.string_lines = {}
        for site in self.string_sites.values():
            frog, tip = site
            p1 = self.data.site_xpos[frog]
            p2 = self.data.site_xpos[tip]
            string_line = p2 - p1
            #string_line /= np.linalg.norm(string_line) if np.linalg.norm(string_line) > 1e-6 else 1.0
            norm = np.linalg.norm(string_line)
            if norm > 1e-6:
                string_line /= norm
            else:
                string_line[:] = 1.0  # or set to a default unit vector
            # allow for a vertical offset 

            self.string_lines[(frog, tip)] = string_line
        self.render_mode = render_mode
        self.viewer = None

        self.reset()

    def reset(self):
        # reset qpos/qvel
        if self.start_positions is not None:
            self.data.qpos[:6] = np.array(self.start_positions)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # reset trackers
        self.current_time = 0.0
        self.current_idx = 0
        self.total_pid_error = np.zeros(6)
        self.prev_torque = np.zeros(6)
        return self._get_obs()

    def step(self, action):
        # apply residual on demo trajectory (not used in reward)
        baseline_qpos = self.demo_traj[min(self.current_idx, len(self.demo_traj)-1)]
        target_qpos = baseline_qpos + action * self.action_scale
        # PID to compute torques
        pos_err = target_qpos - self.data.qpos[:6]
        vel_err = -self.data.qvel[:6]
        self.total_pid_error += pos_err * self.sim_dt
        torque = self.kp*pos_err + self.kd*vel_err + self.ki*self.total_pid_error
        self.data.ctrl[:6] = np.clip(
            torque, self.model.actuator_ctrlrange[:,0], self.model.actuator_ctrlrange[:,1]
        )
        # step
        mujoco.mj_step(self.model, self.data)
        # advance
        self.current_time += self.sim_dt
        self.current_idx = min(int(self.current_time/self.sim_dt), len(self.demo_traj)-1)
        # observe/reward/done
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self.current_time >= self.total_duration
        return obs, reward, done, {}

    def _get_tcp_pos(self):
        # assume body 'ee_link' is end-effector
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_link')
        return self.data.xpos[bid].copy()

    def _get_obs(self):
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()
        # string contacts
        contacts = contact.detect_bow_string_contact(self.model, self.data)
        contact_vec = [float(contacts[s]) for s in ('A','D','G','C')]
        return np.concatenate([qpos, qvel, contact_vec])

    def _compute_reward(self):
        idx = min(self.current_idx, len(self.note_sequence)-1)
        tcp = self._get_tcp_pos()
        raw_s = self.note_sequence[idx].get('string', '')

        if '-' in raw_s:
            target_str = raw_s.split('-')[1]
        else:
            target_str = raw_s

        fid, tid = self.string_sites.get(target_str, (None, None))
        if fid is None or tid is None:
            dist = 0.0
        else:
            p1 = self.data.site_xpos[fid]
            p2 = self.data.site_xpos[tid]
            line_vec = p2 - p1
            norm = np.linalg.norm(line_vec)
            if norm > 1e-6:
                dist = np.linalg.norm(np.cross(tcp - p1, tcp - p2)) / norm
            else:
                dist = 0.0

        r = -dist

        collision, _, _ = contact.detect_collision(self.model, self.data)
        if collision:
            r -= self.contact_penalty

        delta_tau = self.data.ctrl[:6] - self.prev_torque
        r -= self.torque_penalty * np.linalg.norm(delta_tau)
        self.prev_torque = self.data.ctrl[:6].copy()

        if fid is not None and tid is not None:
            p1 = self.data.site_xpos[fid]
            p2 = self.data.site_xpos[tid]
            string_vec = p2 - p1
            norm = np.linalg.norm(string_vec)
            if norm > 1e-8:
                string_vec /= norm
            else:
                string_vec[:] = 1.0  # fallback default unit vector

            tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_link')
            bow_vec = self.data.xmat[tcp_id].reshape((3, 3))[:, 0]
            bow_vec /= np.linalg.norm(bow_vec)

            dot = np.clip(np.dot(bow_vec, string_vec), -1.0, 1.0)
            angle_error = np.abs(np.pi / 2 - np.arccos(dot))
            r -= 0.5 * angle_error

        return r

    def render(self, mode: str = "human"):
        if mode != "human":
            return None      # Gym API requires this guard

        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model,
                                                            self.data)
            except Exception as e:
                print(f"Failed to launch MuJoCo viewer: {e}")
                self.viewer = None

        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# --- geminiRunv2.py ---
import time, sys, importlib
sys.modules['numpy._core.numeric'] = importlib.import_module('numpy.core.numeric')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_trajectory import UR5eCelloTrajectoryEnv
from parsemidi import parse_midi
import pandas as pd

def extract_joint_angles(csv_filename):
    df = pd.read_csv(csv_filename)
    cols = ['q_base','q_shoulder','q_elbow','q_wrist1','q_wrist2','q_wrist3']
    return df[cols].values.tolist()

# if __name__ == '__main__':
#     traj = extract_joint_angles('path/to/log.csv')
#     notes = parse_midi('path/to/file.mid')
#     scene = 'path/to/scene.xml'
#     start = traj[0]
#     def mk():
#         return UR5eCelloTrajectoryEnv(
#             model_path=scene,
#             trajectory=traj,
#             note_sequence=notes,
#             action_scale=0.05,
#             residual_penalty=0.02,
#             contact_penalty=0.1,
#             torque_penalty=0.001,
#             kp=100, kd=2, ki=0.1,
#             start_joint_positions=start
#         )
#     env = DummyVecEnv([mk])
#     model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./tb_residual/')
#     model.learn(total_timesteps=200000)
#     model.save('ppo_residual_ur5e')
