import mujoco
import mujoco.viewer
import numpy as np
import time
import MujocoController as mj
import RobotController as rj
import pandas as pd
import time


controller = rj.RobotController()
# initial_angles = [0.3128551550760768, 0.6237390727874397, 0.1331473851509603, -1.6506585627816217, -2.0735279059562384, 1.0388212810915936]  # Adjust based on your robot's safe position
# controller = rj.RobotController(initial_joint_angles=initial_angles)
# print(controller.model.keyframe('home').qpos)
# controller.run_demo()

song = pd.read_csv('/Users/PV/Robot Cello/Robot-Cello/twinkle-lstm.csv')
joint_angles_list = []
for angle_str in song['joint_angles']:
    angles = [float(x) for x in angle_str.strip('[]').split(',')]
    joint_angles_list.append(angles)


# print(controller.model.geom_contype[:])
# print(controller.model.geom_conaffinity[:])

# print(mujoco.mj_name2id(controller.model,1,  "base1"))

# for i in range(10):
#    print(mujoco.mj_id2name(controller.model, 9, i))

# controller.model.geom_contype[:] = 0
# controller.model.geom_conaffinity[:] = 0

# controller.model.geom_contype[1] = 1
# controller.model.geom_conaffinity[1] = 1

try:
  while True:
    continue
except KeyboardInterrupt:
    pass

# exit()
# controller.simulate_with_joint_positions(joint_angles_list)

for i, angles in enumerate(joint_angles_list):
    # print(f"Position {i}/{len(joint_angles_list)}")
    controller.move_joints(angles)
    print(controller.detect_bow_string_contact())
    # touch_value = controller.data.sensor("base1_touch").data[0]
    # if touch_value > 0:
    #     print("Collision detected with base!")

controller.move_joints([-1.7, -0.5999738735011597, 1.3274701277362269, -1.5217138093760987, -1.3482697645770472, -2.635855499898092])
# touch_value = controller.data.sensor("base1_touch").data[0]
# print(touch_value)
# if touch_value > 0:
#     print("Collision detected with base!")

# controller.move_joints([0.31253460691426516, 0.6282445349567156, 0.13255107237145144, -1.6512125702884328, -2.0741156614679555, 1.039162720514099])

