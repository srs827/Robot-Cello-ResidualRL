import mujoco
import numpy as np
import time
import os
from pathlib import Path
from simple_pid import PID

debug = False
def PRINT(string):
    if debug:
        print(string)

class RobotController:
    def __init__(self, model_path=None):
        """Initialize the robot controller with a MuJoCo model."""
        # Load model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "universal_robots_ur5e/scene.xml")
            # model_path = os.path.join(os.path.dirname(__file__), "UR5+gripper/UR5gripper.xml")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo XML file not found: {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Set up viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True # collision visual!!!!!!!
        
        # Define joint names
        self.joint_names = [
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint", 
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ]
        
        # Get joint IDs
        self.joint_ids = []
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                PRINT(f"Warning: Could not find joint '{name}'")
            self.joint_ids.append(joint_id)
            
        # Create PID controllers
        self.setup_controllers()
        
        # Set target initial position
        initial_position = [-2.5310537973987024, -0.6143371623805542, 1.3550155798541468, -1.5331547309509297, -1.3534582296954554, -2.6431313196765345]
        
        # Move to initial position with collisions disabled
        self.move_to_position_no_collisions(initial_position)

        # Print(self.data.qpos)
        
        # Sync the viewer
        self.viewer.sync()
        
    def setup_controllers(self):
        """Set up PID controllers for each joint."""
        sample_time = 0.001
        p_gains = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0]  # Higher gains for stronger control
        i_gains = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        d_gains = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        output_limits = [(-50, 50), (-50, 50), (-50, 50), (-50, 50), (-50, 50), (-50, 50)]
        
        # Default target positions
        default_positions = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        
        # Create controllers
        self.controllers = []
        for i in range(6):
            controller = PID(
                Kp=p_gains[i],
                Ki=i_gains[i],
                Kd=d_gains[i],
                setpoint=default_positions[i],
                output_limits=output_limits[i],
                sample_time=sample_time
            )
            self.controllers.append(controller)
        
        # Store target positions
        self.target_positions = np.array(default_positions)
        
    def move_to_position_no_collisions(self, joint_positions):
        """Move the robot to initial position with collisions disabled."""
        PRINT("Moving to initial position with collisions disabled...")
        
        # Store original collision settings
        original_contype = self.model.geom_contype.copy()
        original_conaffinity = self.model.geom_conaffinity.copy()
        
        # Disable all collisions
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0

        self.move_joints(joint_positions)

        self.model.geom_contype[:] = original_contype
        self.model.geom_conaffinity[:] = original_conaffinity
        
        PRINT("Initial position reached with collisions disabled")
        
    def move_joints(self, joint_positions, max_steps=1000, tolerance=0.05):
        """Move robot joints to target positions."""
        # Set target positions
        for i, angle in enumerate(joint_positions):
            self.target_positions[i] = angle
            self.controllers[i].setpoint = angle
        
        # Execute movement
        steps = 0
        reached_target = False
        
        PRINT(f"Moving joints to: {joint_positions}")
        
        while not reached_target and steps < max_steps:
            # Get current positions
            current_positions = np.zeros(6)
            for i, joint_id in enumerate(self.joint_ids):
                if joint_id >= 0:
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    angle_deg = np.degrees(self.data.qpos[qpos_adr])
                    wrapped_angle_deg = (angle_deg + 360) % 720 - 360
                    current_positions[i] = np.radians(wrapped_angle_deg)

            
            # Compute new gotrol signals
            control_signals = np.zeros(self.model.nu)
            for i in range(6):
                if self.joint_ids[i] >= 0:
                    control_signals[i] = self.controllers[i](current_positions[i])
            
            # Apply control signals
            self.data.ctrl[:] = control_signals
            
            # Check if target reached
            deltas = np.abs(self.target_positions - current_positions)
            max_delta = np.max(deltas)
            
            if steps % 50 == 0:
                PRINT(f"Step {steps}: Max delta = {max_delta:.4f}")
            
            if max_delta < tolerance:
                reached_target = True
                PRINT(f"Target reached in {steps} steps!")
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            if steps % 5 == 0:  # Update viewer less frequently for speed
                self.viewer.sync()
            
            steps += 1
        
        if not reached_target:
            PRINT(f"Failed to reach target after {steps} steps")
        
        return reached_target
    
    def detect_bow_string_contact(self):
        """Detect contact between bow and strings by examining contact array"""
        bow_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bow_hair")
        g_string_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "G_string")
        c_string_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "C_string")
        d_string_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "D_string")
        a_string_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "A_string")
        
        contacts = {
            "G": False,
            "C": False,
            "D": False,
            "A": False
        }
        
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 == bow_geom_id:
                if con.geom2 == g_string_id:
                    contacts["G"] = True
                elif con.geom2 == c_string_id:
                    contacts["C"] = True
                elif con.geom2 == d_string_id:
                    contacts["D"] = True
                elif con.geom2 == a_string_id:
                    contacts["A"] = True
            elif con.geom2 == bow_geom_id:
                if con.geom1 == g_string_id:
                    contacts["G"] = True
                elif con.geom1 == c_string_id:
                    contacts["C"] = True
                elif con.geom1 == d_string_id:
                    contacts["D"] = True
                elif con.geom1 == a_string_id:
                    contacts["A"] = True
        
        return contacts
    
    def simulate_with_joint_positions(self, joint_angles_list, time_per_position=0.5):
        contact_history = []
        current_position_index = 0
        time_at_current_position = 0
        current_target = joint_angles_list[0]
        
        self.move_joints(current_target)
        
        while current_position_index < len(joint_angles_list):
            mujoco.mj_step(self.model, self.data)
            time_at_current_position += self.model.opt.timestep
            
            contacts = self.detect_bow_string_contact()
            
            contact_info = {
                "time": self.data.time,
                "position_index": current_position_index,
                "contacts": contacts.copy(),
                "joint_angles": self.data.qpos.copy()
            }
            contact_history.append(contact_info)
            
            if any(contacts.values()):
                if not contact_history[-2]["contacts"] if len(contact_history) > 1 else True:
                    print(f"Time {self.data.time:.2f}: Contact with {[k for k,v in contacts.items() if v]}")
            
            if time_at_current_position >= time_per_position:
                current_position_index += 1
                time_at_current_position = 0
                
                if current_position_index < len(joint_angles_list):
                    current_target = joint_angles_list[current_position_index]
                    self.move_joints(current_target)
                    print(f"Moving to position {current_position_index}/{len(joint_angles_list)}")
            
            self.viewer.sync()
        
        return contact_history
