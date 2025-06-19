import pandas as pd
import numpy as np
import torch
import sys, os
sys.path.append("/Users/skamanski/Documents/GitHub/Robot-Cello-ResidualRL/Data-Files/Big-Logs")
from ik_nn import IKNet  # Adjust path if necessary
from scipy.spatial.transform import Rotation as R

# Load your trained neural network
device = torch.device("cpu")
ik_net = IKNet().to(device)
ik_net.load_state_dict(torch.load("ik_net.pth", map_location=device))
ik_net.eval()

# Load log CSV
log_path = '/Users/skamanski/Documents/GitHub/Robot-Cello/biglogs/minuet_no_2v2-log-detailed.csv'  # TODO: update this
log_df = pd.read_csv(log_path)

log_df = log_df.rename(columns={
    'TCP_pose_x': 'x', 'TCP_pose_y': 'y', 'TCP_pose_z': 'z',
    'TCP_pose_rx': 'rx', 'TCP_pose_ry': 'ry', 'TCP_pose_rz': 'rz'
})

# Make predictions
required_fields = ['x', 'y', 'z', 'rx', 'ry', 'rz']
predicted_joint_angles = []
errors = []

device = torch.device("cpu")

for i, row in log_df.iterrows():
    try:
        if not all(field in row and pd.notna(row[field]) for field in required_fields):
            raise ValueError("Missing or NaN in input fields")

        input_pose = torch.tensor([
            row['x'], row['y'], row['z'],
            row['rx'], row['ry'], row['rz']
        ], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = ik_net(input_pose).cpu().numpy().flatten()

        predicted_joint_angles.append(pred)

        true_joint = np.array([
            row['q_base'], row['q_shoulder'], row['q_elbow'],
            row['q_wrist1'], row['q_wrist2'], row['q_wrist3']
        ])
        errors.append(np.linalg.norm(pred - true_joint))

    except Exception as e:
        print(f"Error processing row {i}: {e}")

# Add predictions back to DataFrame if successful
if predicted_joint_angles:
    pred_df = pd.DataFrame(
        predicted_joint_angles,
        columns=["pred_q_base", "pred_q_shoulder", "pred_q_elbow", "pred_q_wrist1", "pred_q_wrist2", "pred_q_wrist3"]
    )
    log_df = log_df.iloc[:len(pred_df)].copy()
    log_df[["pred_q_base", "pred_q_shoulder", "pred_q_elbow", "pred_q_wrist1", "pred_q_wrist2", "pred_q_wrist3"]] = pred_df

    print(f"✅ Average prediction error: {np.mean(errors):.4f} radians")
else:
    print("❌ No valid predictions made.")

# Optionally: save the results
# log_df.to_csv("logs/g_c_d-log-with-predictions.csv", index=False)
