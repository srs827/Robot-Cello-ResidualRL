
import numpy as np

def get_string_points(string_name):
    if string_name == "A_string":
        return [(0.7835, -0.234072, 0.584309), (0.785914, -0.293657, 0.495697)]
    elif string_name == "D_string":
        return [(0.769354, -0.239594, 0.584309), (0.769901, -0.29975, 0.496129)]
    elif string_name == "G_string":
        return [(0.754749, -0.239845, 0.584309), (0.754, -0.298963, 0.497968)]
    elif string_name == "C_string":
        return [(0.739636, -0.234574, 0.584309), (0.738086, -0.293657, 0.495697)]
    else:
        return "Invalid string name"
def get_string_center(string_name):
    if string_name == "A_string":
        return [(0.784707, -0.232115, 0.540003)]
    elif string_name == "D_string":
        return [(0.769628, -0.237922, 0.540219)]
    elif string_name == "G_string":
        return [(0.754375, -0.237654, 0.541139)]
    elif string_name == "C_string":
        return [(0.738861, -0.232366, 0.540003)]
    else:
        return "Invalid string name"
    
def string_dir_vector(string_name):
    p1, p2 = get_string_points(string_name)
    v = np.array(p2) - np.array(p1)
    return v / np.linalg.norm(v)

def perpendicular_bowing_vector(string_name):
    d = string_dir_vector(string_name)
    # Project to XY plane
    d_xy = np.array([d[0], d[1], 0.0])
    d_xy_norm = np.linalg.norm(d_xy)
    if d_xy_norm == 0:
        raise ValueError("String direction has no XY component, cannot compute perpendicular.")
    d_xy_unit = d_xy / d_xy_norm
    # Rotate 90 degrees CCW in XY plane: (x, y) → (−y, x)
    perp_xy = np.array([-d_xy_unit[1], d_xy_unit[0], 0.0])
    return perp_xy

def generate_frog_tip(string_name, center_point=None, length=0.25):
    """Generates frog and tip positions centered at center_point (or midpoint of string if None)
    along the perpendicular bowing direction, spaced by 'length' meters."""
    if center_point is None:
        p1, p2 = get_string_points(string_name)
        center_point = 0.5 * (np.array(p1) + np.array(p2))

    bow_dir = perpendicular_bowing_vector(string_name)
    offset = (length / 2.0) * bow_dir
    frog = center_point - offset
    tip = center_point + offset
    return frog.tolist(), tip.tolist()

a_frog, a_tip = generate_frog_tip("A_string")
d_frog, d_tip = generate_frog_tip("D_string")
g_frog, g_tip = generate_frog_tip("G_string")
c_frog, c_tip = generate_frog_tip("C_string")
print("A string frog:", a_frog, "tip:", a_tip)
print("D string frog:", d_frog, "tip:", d_tip)
print("G string frog:", g_frog, "tip:", g_tip)
print("C string frog:", c_frog, "tip:", c_tip)