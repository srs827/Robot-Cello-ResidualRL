import numpy as np
import pprint
from scipy.spatial.transform import Rotation as R
# Your string_points dictionary goes here...

# rx, ry, rz in rads

string_points = {}
string_points_base['A_string'] = {
    "nut-f": [-0.10341, 0.72787, 0.58586, 1.765, 2.697, -0.992],
    "nut-t": [0.04359, 0.27269, 0.64195, 1.768, 2.740, -1.010],
    "bridge-f": [0.39275, 0.75643, 0.06088, 1.649, 2.610, -1.547],
    "bridge-t": [0.57714, 0.33186, 0.18143, 1.675, 2.572, -1.693],
    "middle-f": [0.34398, 0.75769, 0.12029, 1.640, 2.582, -1.575],
    "middle-t": [0.50021, 0.33330, 0.25332, 1.567, 2.584, -1.404]
}
string_points_base['D_string'] = {
    "nut-f": [-0.08331, 0.71382, 0.58058, 2.175, 2.588, -1.170],
    "nut-t": [0.07991, 0.24027, 0.61263, 2.143, 2.585, -1.159],
    "bridge-f": [0.41601, 0.68951, 0.05264, 2.227, 2.681, -1.536],
    "bridge-t": [0.45321, 0.22193, 0.07679, 2.111, 2.720, -1.527],
    "middle-f": [0.32866, 0.70480, 0.13080, 2.109, 2.598, -1.301],
    "middle-t": [0.35509, 0.23384, 0.17070, 2.037, 2.632, -1.309]
}
string_points_base['G_string'] = {
    "nut-f": [-0.11029, 0.67776, 0.57291, 2.453, 2.594, -0.861],
    "nut-t": [-0.11991, 0.207, 0.54819, 2.425, 2.670, -1.166],
    "bridge-f": [0.37128, 0.63370, 0.04430, 2.635, 2.677, -0.978],
    "bridge-t": [0.30021, 0.18284, 0.01826, 2.435, 2.697, -1.180],
    "middle-f": [0.34391, 0.64827, 0.10755, 2.450, 2.761, -1.313],
    "middle-t": [0.23175, 0.18306, 0.06131, 2.518, 2.722, -1.168]
}
string_points_base['C_string'] = {
    "nut-f": [-0.08113, 0.64214, 0.54913, 2.777, 2.714, -0.989],
    "nut-t": [-0.24503, 0.20556, 0.44273, 2.774, 2.705, -0.893],
    "bridge-f": [0.34861, 0.59842, 0.03028, 2.971, 2.634, -0.786],
    "bridge-t": [0.014485, 0.19058, -0.10480, 2.899, 2.682, -0.752],
    "middle-f": [0.29761, 0.59019, 0.08263, 3.016, 2.756, -0.651],
    "middle-t": [0.13730, 0.17509, -0.03691, 2.871, 2.768, -0.905]
}

string_lines = {}
# in terms of base feature
string_lines['A_string'] = {
    "point1": [-0.10338, 0.72786, 0.58588, 1.765, 2.697, -0.992],
    "point2": [0.39275, 0.75643, 0.06088, 1.649, 2.610, -1.547]
}

string_lines['D_string'] = {
    "point1": [-0.09468, 0.70845, 0.56944, 2.220, 2.541, -1.245],
    "point2": [0.41600, 0.68952, 0.05265, 2.227, 2.681, -1.536]
}

string_lines['G_string'] = {
    "point1": [-0.11028, 0.66782, 0.57292, 2.453, 2.594, -0.861],
    "point2": [0.37127, 0.63370, 0.04430, 2.635, 2.677, -0.978]
}

string_lines['C_string'] = {
    "point1": [-0.08713, 0.63129, 0.54729, 2.745, 2.742, -0.984],
    "point2": [0.34860, 0.59841, 0.03030, 2.971, 2.634, -0.786]
}

tcp_pos = [8.15, -7.88, 33.18] # in mm x,y,z
tcp_orientation = [-0.3253, 0.1259, -0.0058] # in rads rx,ry,rz

payload_mass = 0.240 # in kg
cog = [129, 47, 13] # in mm cx,cy, cz center of gravity from TCP

string_points_string = {}
string_points_string['A_string'] = {
    "nut-f": [],
    "nut-t": [],
    "bridge-f": [],
    "bridge-t": [],
    "middle-f": [],
    "middle-t": []
}
string_points_string['D_string'] = {
    "nut-f": [],
    "nut-t": [],
    "bridge-f": [],
    "bridge-t": [],
    "middle-f": [],
    "middle-t": []
}
string_points_string['G_string'] = {
    "nut-f": [],
    "nut-t": [],
    "bridge-f": [],
    "bridge-t": [],
    "middle-f": [],
    "middle-t": []
}
string_points_string['C_string'] = {
    "nut-f": [],
    "nut-t": [],
    "bridge-f": [],
    "bridge-t": [],
    "middle-f": [],
    "middle-t": []
}


def tcp_to_marker(tcp_pose, offset_tcp_frame):
    pos = np.array(tcp_pose[:3])
    rotvec = np.array(tcp_pose[3:6])
    rotmat = R.from_rotvec(rotvec).as_matrix()
    offset_world = rotmat @ np.array(offset_tcp_frame)
    return (pos + offset_world).tolist()

# Your provided TCP pose and distances
chosen_waypoint = [0.33293, 0.67443, 0.12055, 2.292, 2.658, -1.142]
bow_pos_from_tcp_f = 0.0889   # meters
bow_pos_from_tcp_t = 0.5715   # meters

# Offsets in +Y (bow extends right from TCP)
# for some reason +x not +y oop
frog_offset = [bow_pos_from_tcp_f, 0, 0]
tip_offset  = [bow_pos_from_tcp_t, 0, 0]

# Compute positions
bow_frog_pos = tcp_to_marker(chosen_waypoint, frog_offset)
bow_tip_pos  = tcp_to_marker(chosen_waypoint, tip_offset)



def compute_string_center_and_vector(points):
    """Compute the average string vector from nut and bridge at frog and tip."""
    nut_f = np.array(points['nut-f'][:3])
    nut_t = np.array(points['nut-t'][:3])
    bridge_f = np.array(points['bridge-f'][:3])
    bridge_t = np.array(points['bridge-t'][:3])

    vec_frog = bridge_f - nut_f
    vec_tip  = bridge_t - nut_t

    avg_vec = (vec_frog + vec_tip) / 2
    avg_vec = avg_vec / np.linalg.norm(avg_vec)

    return vec_frog, vec_tip

def compute_avg_rot(points):
    """Average of middle-f and middle-t rotations (rx, ry, rz)."""
    rot_f = np.array(points['middle-f'][3:])
    rot_t = np.array(points['middle-t'][3:])
    #return (rot_f + rot_t) / 2
    return rot_f, rot_t

def align_bow_to_string_perpendicular(bow_frog_pos, bow_tip_pos, string_vec_f, string_vec_t, rot_f, rot_t):
    """Align frog and tip TCPs so bow is perpendicular to string_vec, preserving spacing."""
    bow_frog_pos = np.array(bow_frog_pos)
    bow_tip_pos  = np.array(bow_tip_pos)

    bow_center = (bow_frog_pos + bow_tip_pos) / 2
    bow_length = np.linalg.norm(bow_tip_pos - bow_frog_pos)

    # Project string_vec_f into XY plane and compute perpendicular
    proj_string_vec_f = np.array([string_vec_f[0], string_vec_f[1], 0.0])
    if np.linalg.norm(proj_string_vec_f) < 1e-6:
        raise ValueError("Projected string vector is too small.")
    perp_vec_f = np.array([-proj_string_vec_f[1], proj_string_vec_f[0], 0.0])
    perp_vec_f = perp_vec_f / np.linalg.norm(perp_vec_f)
    
    # Project string_vec_t into XY plane and compute perpendicular
    proj_string_vec_t = np.array([string_vec_t[0], string_vec_t[1], 0.0])
    if np.linalg.norm(proj_string_vec_t) < 1e-6:
        raise ValueError("Projected string vector is too small.")
    perp_vec_t = np.array([-proj_string_vec_t[1], proj_string_vec_t[0], 0.0])
    perp_vec_t = perp_vec_t / np.linalg.norm(perp_vec_t)

    # Preserve Z height of original bow
    perp_vec_f[2] = 0.0
    perp_vec_t[2] = 0.0
    # Recalculate frog and tip positions
    new_frog_pos = bow_center - (bow_length / 2) * perp_vec_f
    new_tip_pos  = bow_center + (bow_length / 2) * perp_vec_t

    frog_tcp = np.concatenate([new_frog_pos, rot_f])
    tip_tcp  = np.concatenate([new_tip_pos, rot_t])

    return frog_tcp.tolist(), tip_tcp.tolist()

# --- Process each string ---

bowing_trajectories = {}

# Process each string and generate new frog/tip TCPs
for string_name, points in string_points.items():
    string_vec_f, string_vec_t = compute_string_center_and_vector(points)
    rot_f, rot_t = compute_avg_rot(points)

    frog_tcp, tip_tcp = align_bow_to_string_perpendicular(
        bow_frog_pos, bow_tip_pos, string_vec_f, string_vec_t, rot_f, rot_t
    )

    bowing_trajectories[string_name] = {
        'frog': frog_tcp,
        'tip': tip_tcp
    }

# Print the final result
print("\nGenerated perpendicular bowing trajectories:")
pprint.pprint(bowing_trajectories)
