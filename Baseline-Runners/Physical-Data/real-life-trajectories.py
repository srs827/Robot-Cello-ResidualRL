import numpy as np

# rx, ry, rz in rads

string_points = {}
string_points['A_string'] = {
    "nut-f": [-0.10341, 0.72787, 0.18586, 1.765, 2.697, -0.992],
    "nut-t": [0.04359, 0.27269, 0.24195, 1.768, 2.740, -1.010],
    "bridge-f": [0.39275, 0.75643, -0.33912, 1.649, 2.610, -1.547],
    "bridge-t": [0.57714, 0.33186, -0.21857, 1.675, 2.572, -1.693],
    "middle-f": [0.34398, 0.75769, -0.27971, 1.640, 2.582, -1.575],
    "middle-t": [0.50021, 0.33330, -0.14668, 1.567, 2.584, -1.404]
}
string_points['D_string'] = {
    "nut-f": [-0.08331, 0.71382, 0.18058, 2.175, 2.588, -1.170],
    "nut-t": [0.07991, 0.24027, 0.21263, 2.143, 2.585, -1.159],
    "bridge-f": [0.41601, 0.68951, -0.34736, 2.227, 2.681, -1.536],
    "bridge-t": [0.45321, 0.22193, -0.32321, 2.111, 2.720, -1.527],
    "middle-f": [0.32866, 0.70480, -0.26920, 2.109, 2.598, -1.301],
    "middle-t": [0.35509, 0.23384, -0.22930, 2.037, 2.632, -1.309]
}
string_points['G_string'] = {
    "nut-f": [-0.11029, 0.67776, 0.17291, 2.453, 2.594, -0.861],
    "nut-t": [-0.11991, 0.207, 0.14819, 2.425, 2.670, -1.166],
    "bridge-f": [0.37128, 0.63370, -0.35570, 2.635, 2.677, -0.978],
    "bridge-t": [0.30021, 0.18284, -0.38174, 2.435, 2.697, -1.180],
    "middle-f": [0.34391, 0.64827, -0.29245, 2.450, 2.761, -1.313],
    "middle-t": [0.23175, 0.18306, -0.33869, 2.518, 2.722, -1.168]
}
string_points['C_string'] = {
    "nut-f": [-0.08113, 0.64214, 0.14913, 2.777, 2.714, -0.989],
    "nut-t": [-0.24503, 0.20556, 0.04273, 2.774, 2.705, -0.893],
    "bridge-f": [0.34861, 0.59842, -0.36972, 2.971, 2.634, -0.786],
    "bridge-t": [0.014485, 0.19058, -0.50480, 2.899, 2.682, -0.752],
    "middle-f": [0.29761, 0.59019, -0.31737, 3.016, 2.756, -0.651],
    "middle-t": [0.13730, 0.17509, -0.43691, 2.871, 2.768, -0.905]
}

offset_string_points = {}
for string, points in string_points.items():
    offset_string_points[string] = {}
    for key, value in points.items():
        # Apply the offset to the first three coordinates
        offset_string_points[string][key] = [
            value[0] - 0.0,  # x offset
            value[1] + 0.0,  # y offset
            value[2] - 0.0   # z offset
        ] + value[3:]  # Keep the last three values unchanged

def unit_vector(vec):
    return vec / np.linalg.norm(vec)

def compute_bowing_vectors(offset_string_points):
    bowing_info = {}
    for string, points in offset_string_points.items():
        nut_f = np.array(points['nut-f'][:3])
        nut_t = np.array(points['nut-t'][:3])
        bridge_f = np.array(points['bridge-f'][:3])
        bridge_t = np.array(points['bridge-t'][:3])
        middle_f = np.array(points['middle-f'][:3])
        middle_t = np.array(points['middle-t'][:3])

        string_vec_f = unit_vector(bridge_f - nut_f)            # Along the string
        string_vec_t = unit_vector(bridge_t - nut_t)            # Along the string

        avg_string_vec = (string_vec_f + string_vec_t) / 2.0  # Average direction
        bow_vec = unit_vector(np.cross([0, 0, 1], avg_string_vec))  # Approx. horizontal perpendicular
        
        middle = (middle_f + middle_t) / 2.0  # Midpoint between frog and tip

        bowing_info[string] = {
            "bow_direction": bow_vec,     # unit vector perpendicular to the string
            "bow_start": middle - 0.15 * bow_vec,  # 15 cm toward frog
            "bow_end": middle + 0.15 * bow_vec,     # 15 cm toward tip
            "middle_pos": middle,  # Midpoint between frog and tip
            "string_vector": avg_string_vec
        }
    return bowing_info

bowing_info = compute_bowing_vectors(offset_string_points)
#print(bowing_info)

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def compute_axis_angles(bowing_info, offset_string_points):
    result = {}
    for string, info in bowing_info.items():
        # Extract middle rotations from both frog and tip ends
        middle_f_rot = np.array(offset_string_points[string]['middle-f'][3:6])
        middle_t_rot = np.array(offset_string_points[string]['middle-t'][3:6])

        # Create Rotation objects
        r_f = R.from_rotvec(middle_f_rot)
        r_t = R.from_rotvec(middle_t_rot)

        # Perform spherical linear interpolation
        slerp = Slerp([0, 1], R.concatenate([r_f, r_t]))
        r_avg = slerp(0.5)

        # Convert averaged rotation to axis-angle
        axis_angle = r_avg.as_rotvec()

        # Store full pose: frog and tip positions + shared orientation
        result[string] = {
            "frog_pose": np.concatenate([info["bow_start"], axis_angle]),
            "tip_pose":  np.concatenate([info["bow_end"], axis_angle]),
            "orientation_rpy": axis_angle
        }

    return result

# Recompute poses
poses = compute_axis_angles(bowing_info, offset_string_points)

# Convert to DataFrame for display
import pandas as pd
pose_df = pd.DataFrame.from_dict(poses, orient='index')
# print(pose_df['frog_pose'].apply(lambda x: x.tolist()).to_list())
# print(pose_df['tip_pose'].apply(lambda x: x.tolist()).to_list())
# print(pose_df['orientation_rpy'].apply(lambda x: x.tolist()).to_list())
# Save the poses to a CSV file
pose_df.to_csv('bowing_poses.csv', index_label='string')

def mujoco_string_sites(offset_string_points):
    sites = {}
    for string, points in offset_string_points.items():
        middle_f = np.array(points["middle-f"][:3])
        middle_t = np.array(points["middle-t"][:3])
        midpoint = (middle_f + middle_t) / 2
        sites[string] = midpoint.tolist()
    return sites

print(mujoco_string_sites(offset_string_points))
print(" ")
print(offset_string_points)