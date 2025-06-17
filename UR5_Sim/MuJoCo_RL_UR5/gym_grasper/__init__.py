from gym.envs.registration import register
from gym_grasper.version import VERSION as __version__

# Ensure "Grasper-v0" is registered (it already appears in the list)
register(
    id="Grasper-v0",
    entry_point="gym_grasper.envs.GraspingEnv:GraspEnv",
)

# Register UR5e Cello Environment (This is missing!)
register(
    id="UR5e-Cello-v0",
    entry_point="gym_grasper.envs.ur5_cello_env:UR5eCelloEnv",
)
