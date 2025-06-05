import time
import pandas as pd
import socket
import mido
from mido import MidiFile

# If robot doesn't connect, you may need to re-check the port by going to Settings -> Network on the teach pendant
ROBOT_IP = "128.46.75.219"
UR_PORT = 30002  # URScript Port
RTDE_PORT = 30004  # RTDE Port

CLEF = "bass"

data_log = []

def get_note_name(note_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note = note_names[note_number % 12]
    return f"{note}{octave}"

def parse_midi(file_path, clef="bass"):
    midi = MidiFile(file_path)
    note_events = []

    def get_cello_string(note_number):
        if note_number >= 57:
            return 'A'
        elif note_number >= 50:
            return 'D'
        elif note_number >= 43:
            return 'G'
        else:
            return 'C'

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

                raw_notes.append({
                    'number': msg.note,
                    'note': get_note_name(msg.note),
                    'duration': duration,
                    'string': string,
                    'start_time': start_time,
                    'end_time': current_time
                })

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
                    'end_time': note['end_time'] + 0.2
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
    bow_direction = True

    for note in note_sequence:
        function = script_funcs[note["string"]]
        note_duration = note["duration"]
        
        if note["note"] == 'transition':
            res += f"{function}()\n  "
        else:
            bow_direction = not bow_direction
            res += f"{function}({bow_direction}, {note_duration})\n  stay()\n  "
    return res

def send_urscript(urscript, speed_scaling):
    try:
        if not urscript:
            return
        speed_command = f"set speed {speed_scaling}\n"
        urscript = speed_command + urscript
        
        with socket.create_connection((ROBOT_IP, UR_PORT), timeout=10) as sock:
            sock.sendall(urscript.encode('utf-8'))
            print("Sent URScript")
            
    except Exception as e:
        print(f"Error sending URScript: {e}")

data_log = []

def get_function_sequence(note_sequence):
    # global data_log
    res = ""

    # for bow_direction, True is down bow (towards tip) and False is up bow (towards frog)
    bow_direction = True
    note_num = 0

    for i, note in enumerate(note_sequence):
        function = script_funcs[note["string"]]

        note_duration = note["duration"]
        
        if note["note"] == 'transition':
            res += function + "()\n  "
        else:
            # note_num += 1
            # if note_num % 4 == 0:
            #     bow_direction = not bow_direction
            bow_direction = not bow_direction

            # bow_direction = not bow_direction
            res += function + f"({bow_direction}, {note_duration})\n  stay()\n  "
    return res



# load_scripts()
note_sequence = parse_midi("/Users/skamanski/Documents/GitHub/Robot-Cello/midi_robot_pipeline/midi_files/minuet_no_2v2.mid")
print(note_sequence)
function_sequence = get_function_sequence(note_sequence)
# print(function_sequence)

f = open("/Users/skamanski/Documents/GitHub/Robot-Cello/song.script", "r")
script = f.read()

starting_pose = f'{note_sequence[0]["string"].lower()}_bow_poses.frog_p'

script = script.replace("# $$$ CODE HERE $$$", f"""
    movej(p[{starting_pose}[0], {starting_pose}[1], {starting_pose}[2], {starting_pose}[3], {starting_pose}[4], {starting_pose}[5]])
    {function_sequence}
    """)
# print(script)
test_file = open('test.txt', "w")
test_file.write(script)

# send_urscript(script, 0.8)

# load scripts before playing the note sequence
# load_scripts()
# can change to whichever MIDI file you want to test
# note_sequence = parse_midi("/Users/samanthasudhoff/Desktop/midi_robot_pipeline/midi_files/allegro.mid")
# # this actually plays the notes 
# play_sequence(note_sequence)

# df = pd.DataFrame(data_log)
# change the name as needed
# df.to_csv("ur5_rtde_data-allegro.csv", index=False)
# print("Saved as 'ur5_rtde_data-allegro.csv'.")

# con.send_pause()
# con.disconnect()
# --- in idk3.py, add at the end ---

# ────────────────────────────────────────────────────────────────
# at end of idk3.py, add:

class CelloController:
    def __init__(self, model_path, trajectory=None, note_sequence=None, start_positions=None, **kwargs):
        """
        Wrapper for your existing idk3 logic.
        :param model_path: (unused here—just passed through)
        :param trajectory: prerecorded joint data (unused here)
        :param note_sequence: list of note dicts from parse_midi()
        :param start_positions: initial joint positions (if needed)
        :param kwargs: absorbs any other args (like render_mode)
        """
        self.model_path = model_path
        self.trajectory = trajectory
        self.note_sequence = note_sequence
        self.start_positions = start_positions
    def get_baseline_qpos(self, idx):
        if self.trajectory is None:
            raise ValueError("No trajectory was passed to CelloController")
        # clamp idx
        i = min(max(idx, 0), len(self.trajectory)-1)
        return self.trajectory[i]
    def generate_joint_trajectory(self):
        """
        Return the full baseline trajectory as a list.
        """
        if self.trajectory is None:
            raise ValueError("No trajectory was passed to CelloController")
        return list(self.trajectory)
    def baseline(self, idx):
        """
        Return the URScript snippet for the single note at index idx.
        """
        if idx < 0 or idx >= len(self.note_sequence):
            return ""
        # use your existing function to build URScript for one-note slice
        return get_function_sequence([ self.note_sequence[idx] ])

    def play(self, urscript, speed=1.0):
        """
        Send URScript to the robot.
        """
        send_urscript(urscript, speed)
# ────────────────────────────────────────────────────────────────


