import mujoco

def detect_bow_string_contact(model, data):
        """Detect contact between bow and strings by examining contact array"""
        bow_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bow_hair")
        g_string_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "G_string")
        c_string_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "C_string")
        d_string_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "D_string")
        a_string_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "A_string")
        
        contacts = {
            "G": False,
            "C": False,
            "D": False,
            "A": False
        }
        
        for i in range(data.ncon):
            con = data.contact[i]
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
    

def detect_collision(model, data):
    bow_hair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bow_hair")
    strings = [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "G_string"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "C_string"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "D_string"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "A_string") ]

    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 not in strings and con.geom1 != bow_hair:
            return True, con.geom1, con.geom2
        elif con.geom2 not in strings and con.geom2 != bow_hair:
            return True, con.geom1, con.geom2
        elif con.geom1 == bow_hair:
            if con.geom2 not in strings:
                return True, con.geom1, con.geom2
        elif con.geom2 == bow_hair:
            if con.geom1 not in strings:
                return True, con.geom1, con.geom2
    
    return False, None, None


def list_collision_mesh_geoms_with_bodies(model):
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"Geom ID: {i}, Name: {geom_name}, Body: {body_name}")
    
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"Body ID: {i}, Name: {name} ")

    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        print(f"Geom ID: {i}, Name: {name}     {model.geom_type[i]}")

