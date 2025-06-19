# Robot Cello Residual Reinforcement Learning Repository

## TODO

- [ ] Replace deprecated `gym` with `gymnasium`
- [ ] Create a GUI frontend for managing `.env`

## Running Simulation Using `rl_runner.py`

1. Create a Python Virtual Environment
    ```sh
    mkdir robot-cello-residual-rl-venv && cd robot-cello-residual-rl-venv
    python3 -m venv .
    source ./bin/activate
    # `activate` is for a POSIX compliance shell (Bash, Zsh, etc.)
    # if you are using another shell, say Fish or Powershell, choose the appropriate activate file
    ```
    Your shell prompt should now have the blue `(robot-cello-residual-rl-venv)` in front of the prompt to indicate that you are in the virtual environment.
2. Download dependencies to your venv using `pip`
    - [MuJuCo](https://mujoco.org/): Robotics simulation engine
    - [Pandas](https://pandas.pydata.org/): Data analytics and calculation
    - [Gym](https://www.gymlibrary.dev/): Reinforcement learning environment
    - [Mido](https://mido.readthedocs.io/en/latest/): Parsing MIDI files and creating MIDI objects
    - [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/): Implementation of reinforcement learning
    - [python-dotenv](https://pypi.org/project/python-dotenv/): For parsing `.env` file
    ```sh
    python3 -m pip install mujoco pandas gym mido stable_baselines3 python-dotenv
    ```
3. Modify `.env` file
4. Run `rl_runner.py` with `mjpython` command
    ```sh
    mjpython ./RL-code/rl_runner.py
    ```

## File Structure

```
.
├── Baseline-Runners
│   ├── baseline_controller.py
│   ├── Physical-Data
│   │   └── ...
│   ├── robot_runner_detailed-logs.py
│   ├── robot_runner_simple.py
│   └── robot_runner.py
├── Data-Files
│   └── ...
├── MIDI-Files
│   └── ....mid
├── Pieces-Bowings
│   └── ..._bowings.txt
├── README.md
├── RL-code
│   ├── base_runner_2.py
│   ├── bowing_poses.csv
│   ├── bowing-info.txt
│   ├── calculate-bowing-traj.py
│   ├── contact.py
│   ├── logs_runner.py
│   ├── mujoco_base_runner.py
│   ├── parsemidi.py
│   ├── rl_runner.py
│   └── rl_trajectory.py
├── Robot-Programs
│   ├── ..._full_bow.script
│   └── programs
│       └── ... a lot of .script files
├── UR5_Sim
│   ├── GripperBow
│   │   └── material.mtl
│   ├── MuJoCo_RL_UR5
│   │   ├── ...
│   │   ├── gym_grasper
│   │   │   ├── ...
│   │   │   ├── envs
│   │   │   │   ├── __init__.py
│   │   │   │   ├── GraspingEnv.py
│   │   │   │   └── ur5_cello_env.py
│   │   └── UR5+gripper
│   │       └── ...
│   ├── universal_robots_ur5e
│   │   ├── assets
│   │   │   ├── Base Cello.stl
│   │   │   ├── base_0.obj
│   │   │   ├── wrist2_2.obj
│   │   │   └── ....obj
│   │   ├── Base
│   │   │   └── ...
│   │   ├── GripperBow_....obj
│   │   └── ...
│   ├── UR5+gripper
│   │   └── ...
│   └── ur5e
│       └── ...
└── URScripts
    └── ...

55 directories, ??? files
```

