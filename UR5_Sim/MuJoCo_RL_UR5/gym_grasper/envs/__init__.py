from gym_grasper.envs.GraspingEnv import GraspEnv
import gym
from gym.envs.registration import register
from gym_grasper.envs.ur5_cello_env import UR5eCelloEnv

# Register the UR5e Cello Bowing Environment
register(
    id="UR5e-Cello-v0",
    entry_point="gym_grasper.envs.ur5_cello_env:UR5eCelloEnv",
)
