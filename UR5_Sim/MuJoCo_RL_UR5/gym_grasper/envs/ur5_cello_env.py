import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gym import spaces
import mujoco
from gym_grasper.controller.MujocoController import MJ_Controller as MujocoController
from gym.utils import seeding


class UR5eLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(UR5eLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x):
        out, _ = self.lstm(x)
        predicted_pos = self.fc(out[:, -1, :])
        return predicted_pos


class UR5eCelloEnv(gym.Env):
    """Custom Gym environment for UR5e cello bowing using MuJoCo with LSTM pretraining."""

    def __init__(self):
        super(UR5eCelloEnv, self).__init__()
        self.controller = MujocoController("/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/UR5_Sim/MuJoCo_RL_UR5/env_experiment/universal_robots_ur5e/ur5e3.xml")

        # self.lstm = UR5eLSTM(input_dim=33, hidden_dim=128, output_dim=24, num_layers=2)
        # # will need to replace the path
        # model_path = "/Users/samanthasudhoff/Desktop/midi_robot_pipeline/pretrained_ur5e_lstm3.pth"
        # if not torch.cuda.is_available():
        #     device = torch.device("cpu")
        # else:
        #     device = torch.device("cuda")

        # if not torch.cuda.is_available():
        #     self.lstm.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        # else:
        #     self.lstm.load_state_dict(torch.load(model_path))

        # self.lstm.to(device)
        # # set to inference mode 
        # self.lstm.eval()  


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)

        #self.lookback = 10
        self.current_pos = np.zeros(6, dtype=np.float32)
        self.previous_pos = np.zeros(6, dtype=np.float32)
        # set to initial TCP pos
        self.target_pos = np.array([0.3, 0.7, 0.1, -1.5, -2.2, 1.1])  

        # Maintain past states for LSTM input
        #self.state_history = np.zeros((self.lookback, 33), dtype=np.float32)  # 33 features per step

    def step(self, action):
        """Apply PPO action, update Mujoco, and return full 24-value observation."""
        
        action = np.array(action).flatten()
        if action.shape[0] == 1:  
            action = np.tile(action, (24,))  # action acts really quirky idk what the issue is
        if action.shape[0] != 24:
            raise ValueError(f"Expected action shape (24,), but got {action.shape}")

        predicted_tcp = action[:6] * 0.1  
        predicted_joints = action[6:12] * 0.05  
        predicted_speeds = action[12:18] * 0.02  
        predicted_forces = action[18:24] * 0.1  


        # apply actions in mujoco -- i don't know if this actually works
        self.controller.actuate_joint_group("Arm", predicted_joints.tolist())
        self.controller.data.qpos[:6] = predicted_tcp
        self.controller.data.qpos[6:12] = predicted_joints
        self.controller.data.qvel[:6] = predicted_speeds
        self.controller.data.qfrc_applied[:6] = predicted_forces

        # step simulation
        mujoco.mj_step(self.controller.model, self.controller.data)
        #mujoco.mj_forward(self.controller.model, self.controller.data)  # Ensure physics updates
        #mujoco.mjv_updateScene(self.controller.model, self.controller.data, mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera())
        # update positions
        self.previous_pos = self.current_pos.copy()
        self.current_pos = np.array(self.controller.data.qpos[:6])

        # compute reward 
        reward = self.compute_reward()

        observation = np.concatenate([
            self.controller.data.qpos[:6],  
            self.controller.data.qpos[6:12],  
            self.controller.data.qvel[:6],  
            self.controller.data.qfrc_applied[:6]  
        ])

        # done if at goal
        done = np.linalg.norm(self.current_pos - self.target_pos) < 0.01


        return observation, reward, done, False, {} #

    def compute_reward(self):
        """Reward for smooth bowing motion, positional accuracy, and minimal movement btwn string transitions."""
        
        distance_before = np.linalg.norm(self.previous_pos - self.target_pos)
        distance_after = np.linalg.norm(self.current_pos - self.target_pos)
        progress_reward = distance_before - distance_after  # positive if moving toward target

        # penalize large joint velocity changes to enforce smooth transitions
        velocity_penalty = np.linalg.norm(self.controller.data.qvel[:6])

        # transition reward: encourage small joint movement when switching strings
        # TODO 

       # add transition reward 
        reward = (progress_reward - 0.1 * velocity_penalty + 0.5 * np.exp(-np.linalg.norm(self.current_pos - self.target_pos)))

        # bonus for being close to target
        if distance_after < 0.05:
            reward += 10  
        #print(reward)
        return reward

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation."""
        if seed is not None:
            self.seed(seed)  
        mujoco.mj_resetData(self.controller.model, self.controller.data)
        self.state_history = np.zeros((self.lookback, 33), dtype=np.float32)

        # Use LSTM to initialize movement
        # lstm_input = torch.tensor(self.state_history, dtype=torch.float32).unsqueeze(0)  
        # lstm_prediction = self.lstm(lstm_input).detach().numpy().flatten()
        
        # we want to change this to use the baseline controller for initializing movement 

        # self.current_pos = lstm_prediction[:6]
        self.current_pos = 
        obs = np.concatenate([
            self.controller.data.qpos[:6],  
            self.controller.data.qpos[6:12],  
            self.controller.data.qvel[:6],  
            self.controller.data.qfrc_applied[:6]  
        ])
        return obs, {}

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
