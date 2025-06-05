from setuptools import setup, find_packages

setup(
    name="MuJoCo_RL_UR5",
    version="1.0",
    packages=find_packages(include=["gym_grasper", "gym_grasper.*"]),  # Explicitly install only `gym_grasper`
    install_requires=[
        "gym",
        "numpy",
        "mujoco",
        "stable-baselines3",
    ],
)

