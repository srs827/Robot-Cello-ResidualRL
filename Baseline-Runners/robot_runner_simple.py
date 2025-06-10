import time
import pandas as pd
import socket
import mido
from mido import MidiFile
import sys
sys.path.append('/home/skamanski/Downloads/rtde-2.7.2-release/rtde-2.7.2')
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config


ROBOT_IP = "128.46.75.201"
UR_PORT = 30002  
RTDE_PORT = 30004  

CLEF = "bass"

config_filename = "/home/skamanski/Downloads/rtde-2.7.2-release/rtde-2.7.2/examples/control_loop_configuration.xml"
conf = rtde_config.ConfigFile(config_filename)
con = rtde.RTDE(ROBOT_IP, RTDE_PORT)
con.connect()
con.get_controller_version()

con.send_output_setup(["timestamp", "actual_TCP_pose", "actual_q", "actual_TCP_force", "actual_TCP_speed"])
con.send_start()

data_log = []

def get_note_name(note_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note = note_names[note_number % 12]
    return f"{note}{octave}"

def read_bowing_file(bowing_file):
    bowing_dict = {}
    with open(bowing_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:
                index = int(parts[0])
                bowing_dict[index] = parts[1].strip("'")  
    return bowing_dict

def parse_midi(file_path, bowing_file, clef="bass"):
    midi = MidiFile(file_path)
    note_events = []
    bowing_dict = read_bowing_file(bowing_file)

    def get_cello_string(note_number):
        if note_number >= 57:
            return 'A'
        elif note_number >= 50:
            return 'D'
        elif note_number >= 43:
            return 'G'
        else:
            return 'C'

    last_bow = "down"  
    index = 0  

    for track in midi.tracks:
        raw_notes = []
        active_notes = {}
        current_time = 0

        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = current_time
            elif msg.type in ('note_off', 'note_on') and msg.note in active_notes:
                start_time = active_notes.pop(msg.note)
                duration = (current_time - start_time) / midi.ticks_per_beat
                mapping_note = msg.note - 12 if clef == "tenor" else msg.note
                string = get_cello_string(mapping_note)

                if index in bowing_dict:
                    bowing = bowing_dict[index]
                    if "-s" in bowing:  
                        bowing = last_bow + "-s"
                    else:
                        last_bow = bowing  
                else:
                    bowing = "up" if last_bow == "down" else "down"
                    last_bow = bowing

                raw_notes.append({
                    'number': msg.note,
                    'note': get_note_name(msg.note),
                    'duration': duration,
                    'string': string,
                    'start_time': start_time,
                    'end_time': current_time,
                    'bowing': bowing
                })
                index += 1  
        print(raw_notes)
        for i, note in enumerate(raw_notes):
            current_string = note['string']
            next_string = raw_notes[i + 1]['string'] if i + 1 < len(raw_notes) else None
            note_events.append(note)

            if next_string and next_string != current_string:
                note_events.append({
                    'number': 'transition',
                    'note': "transition",
                    'duration': 0.2,
                    'string': f"{current_string}-{next_string}",
                    'start_time': note['end_time'],
                    'end_time': note['end_time'] + 0.2,
                    'bowing': "transition"
                })

    return note_events

script_funcs = {
    "A": "a_bow",
    "D": "d_bow",
    "G": "g_bow",
    "C": "c_bow",
    "A-D": "a_to_d",
    "D-A": "d_to_a",
    "D-G": "d_to_g",
    "G-D": "g_to_d",
    "A-G": "a_to_g",
    "G-A": "g_to_a",
    "D-C": "d_to_c",
    "C-D": "c_to_d",
    "G-C": "g_to_c",
    "C-G": "c_to_g"
}

def get_function_sequence(note_sequence):
    res = ""

    for note in note_sequence:
        function = script_funcs[note["string"]]
        note_duration = note["duration"] * 2
        
        if note["note"] == 'transition':
            res += f"{function}()\n  "
        else:
            bowing_value = "1" if "up" in note["bowing"] else "0"
            res += f"{function}({bowing_value}, {note_duration})\n  stay()\n  "
    return res

def get_function_sequence2(note_sequence):
    res = ""
    bow_direction = True

    for note in note_sequence:
        function = script_funcs[note["string"]]
        note_duration = note["duration"] * 2
        
        if note["note"] == 'transition':
            res += f"{function}()\n  "
        else:
            bow_direction = not bow_direction
            res += f"{function}({int(bow_direction)}, {note_duration})\n  stay()\n  "
    return res



rtde_running = True

def send_urscript(urscript, speed_scaling, note_sequence):
    global rtde_running
    try:
        if not urscript:
            print("No URScript!")
            return
        speed_command = f"set speed {speed_scaling}\n"
        urscript = speed_command + urscript

        init_time = -1
        
        with socket.create_connection((ROBOT_IP, UR_PORT), timeout=10) as sock:
            sock.sendall(urscript.encode('utf-8'))
            print("Sent URScript...")
            
            start_time = time.time()
            
            for note in note_sequence:
                if not rtde_running:  # Stop logging if interrupted
                    print("RTDE Not Running")
                    break

                # while True:
                #     state = con.receive()
                #     if state is None:
                #         continue

                    # if state.actual_TCP_speed[0] < 0.001:
                    #     break
                
                
                state = con.receive()
                if init_time == -1:
                    init_time = state.timestamp
                if state:
                    data_log.append({
                        "timestamp": state.timestamp - init_time,
                        "event": f"Start {note['note']} on {note['string']}",
                        "TCP_pose": state.actual_TCP_pose,
                        "joint_angles": state.actual_q,
                        "TCP_force": state.actual_TCP_force
                    })
                
                # time.sleep(note["duration"])  # Wait for the duration of the note

                
                if not rtde_running:  # Stop logging if interrupted
                    print("Test2")
                    break
                
                state = con.receive()
                if state:
                    data_log.append({
                        "timestamp": state.timestamp - init_time,
                        "event": f"End {note['note']} on {note['string']}",
                        "TCP_pose": state.actual_TCP_pose,
                        "joint_angles": state.actual_q,
                        "TCP_force": state.actual_TCP_force
                    })
                    
    except KeyboardInterrupt:  # Handle Ctrl+C interrupt
        print("Script interrupted. Saving collected data...")
        rtde_running = False
        save_data()
        con.disconnect()
        exit(0)
        
    except Exception as e:
        print(f"Error sending URScript: {e}")
    
    finally:
        save_data()

def save_data():
    df_log = pd.DataFrame(data_log)
    log_filename = "allegro-log-june9.csv"
    df_log.to_csv(log_filename, index=False)
    print(f"Saved note event log as '{log_filename}'.")

note_sequence = parse_midi("/home/skamanski/Robot-Cello-ResidualRL/MIDI-Files/allegro.mid", 
                           "/home/skamanski/Robot-Cello-ResidualRL/Pieces-Bowings/allegro_bowings.txt")

function_sequence = get_function_sequence(note_sequence)

with open("/home/skamanski/Robot-Cello-ResidualRL/URScripts/song.script", "r") as f:
    script = f.read()

script = script.replace("# $$$ CODE HERE $$$", f"""{function_sequence}""")
print(script)
with open('test.txt', "w") as test_file:
    test_file.write(script)

send_urscript(script, 1, note_sequence)

con.send_pause()
con.disconnect()