import time
import pandas as pd
import socket
import mido
from mido import MidiFile, tempo2bpm, bpm2tempo
import sys
sys.path.append('/home/skamanski/Downloads/rtde-2.7.2-release/rtde-2.7.2') # Adjust this path if needed
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import random
import math # For infinity

import numpy as np
def generate_random_sequence(num_notes: int = 64,
                             min_note_dur: float = 0.25,
                             max_note_dur: float = 2.0,
                             transition_dur: float = 0.20,
                             strings = ('A', 'D', 'G', 'C')
                            ):
    """
    Return a list of dicts identical to what `parse_midi` produces, but filled
    with pseudo-random notes and (optional) string-crossing transitions.

    • Durations are uniform in [min_note_dur, max_note_dur].  
    • First bowing is hardcoded as 'up'.  
    • Bowings alternate up/down after that.  
    • When the next note is on a *different* string, insert a transition
      event whose 'string' looks like 'A-D' so that `script_funcs` resolves.
    """
    events, t, last_string, last_bow = [], 0.0, None, None  # ← no default bowing

    for idx in range(num_notes):
        string = np.random.choice(strings)
        midi_number = {
            'A': np.random.randint(57, 60),  # ~A3
            'D': np.random.randint(50, 55),  # ~D3
            'G': np.random.randint(43, 48),  # ~G2
            'C': np.random.randint(36, 41)   # ~C2
        }[string]
        note_name = get_note_name(midi_number)
        dur = float(np.random.uniform(min_note_dur, max_note_dur))

        if last_string and string != last_string:
            events.append({
                'number'     : 'transition',
                'note'       : f'transition {last_string}->{string}',
                'duration_sec': transition_dur,
                'string'     : f'{last_string}-{string}',
                'start_time_sec': t,
                'end_time_sec'  : t + transition_dur,
                'bowing'     : 'transition',
                'is_transition': True,
                'event_index': -1
            })
            t += transition_dur

        # --- enforce first bowing is 'up' -------------------------------
        if last_bow is None:
            bowing = 'up'
        else:
            bowing = 'up' if last_bow == 'down' else 'down'

        events.append({
            'number'     : midi_number,
            'note'       : note_name,
            'duration_sec': dur,
            'string'     : string,
            'start_time_sec': t,
            'end_time_sec'  : t + dur,
            'bowing'     : bowing,
            'is_transition': False,
            'event_index': idx
        })

        t += dur
        last_string, last_bow = string, bowing

    print(f"Generated {len(events)} random events.")
    return events


# --- Configuration ---
ROBOT_IP = "128.46.75.201"
UR_PORT = 30001
RTDE_PORT = 30004
CLEF = "bass" # Or "tenor"
midi_name = "allegro"
CONFIG_FILENAME = "/home/skamanski/Downloads/rtde-2.7.2-release/rtde-2.7.2/examples/cello_configuration.xml" # Adjust path
MIDI_FILE_PATH = "/home/skamanski/Robot-Cello-ResidualRL/MIDI-Files/allegro.mid" # Adjust path
# MIDI_FILE_PATH = f"/home/skamanski/Downloads/Robot-Cello-main(2)/Robot-Cello-main/midi_robot_pipeline/midi_files/{midi_name}.mid" # Adjust path
BOWING_FILE = "/home/skamanski/Robot-Cello-ResidualRL/Pieces-Bowings/allegro_bowings.txt" # Or path to your bowing file
SONG_SCRIPT_TEMPLATE = "/home/skamanski/Robot-Cello-ResidualRL/URScripts/song.script" # Adjust path
OUTPUT_LOG_FILENAME = f"/home/skamanski/Robot-Cello-ResidualRL/generated_cello_script.txt"
DEFAULT_TEMPO_BPM = 120 # Used if no tempo message is found in MIDI

# --- RTDE Connection Setup ---
try:
    conf = rtde_config.ConfigFile(CONFIG_FILENAME)
    # Specify the recipe name from your XML file that contains the desired outputs
    # Make sure your XML recipe includes: "timestamp", "output_int_register_0", "actual_TCP_pose", "actual_q", "actual_TCP_force"
    output_names, output_types = conf.get_recipe("state") # Adjust recipe name if needed

    con = rtde.RTDE(ROBOT_IP, RTDE_PORT)
except FileNotFoundError:
    print(f"❌ Error: RTDE Configuration file not found at {CONFIG_FILENAME}")
    sys.exit(1)
except KeyError as e:
    print(f"❌ Error: Recipe 'state' (or your chosen name) not found or missing fields in {CONFIG_FILENAME}. Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error setting up RTDE configuration: {e}")
    sys.exit(1)


# --- Data Log ---
data_log = []
rtde_running = True
cntrl_c = False

# --- Helper Functions ---

def interpret_flag(flag):
    # (Keep your existing interpret_flag mapping)
    mapping = {
        1: "START a_bow", 2: "END a_bow", 3: "START d_bow", 4: "END d_bow",
        5: "START g_bow", 6: "END g_bow", 7: "START c_bow", 8: "END c_bow",
        101: "START a_to_d", 102: "END a_to_d", 103: "START d_to_a", 104: "END d_to_a",
        105: "START d_to_g", 106: "END d_to_g", 107: "START g_to_d", 108: "END g_to_d",
        109: "START a_to_g", 110: "END a_to_g", 111: "START g_to_a", 112: "END g_to_a",
        113: "START d_to_c", 114: "END d_to_c", 115: "START c_to_d", 116: "END c_to_d",
        117: "START g_to_c", 118: "END g_to_c", 119: "START c_to_g", 120: "END c_to_g",
        0: "IDLE/BETWEEN" # Assuming 0 is the default state
    }
    # Handle potential initial -1 state or other values if needed
    if flag == -1: return "INITIALIZING"
    return mapping.get(flag, f"Unknown ({flag})")


def get_note_name(note_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note = note_names[note_number % 12]
    return f"{note}{octave}"

def read_bowing_file(bowing_file):
    # (Keep your existing read_bowing_file function)
    bowing_dict = {}
    try:
        with open(bowing_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    try:
                        index = int(parts[0])
                        bowing_dict[index] = parts[1].strip().strip("'")
                    except ValueError:
                        print(f"Warning: Skipping invalid line in bowing file: {line.strip()}")
    except FileNotFoundError:
        print(f"Warning: Bowing file not found at {bowing_file}. Using default bowing.")
        return None
    return bowing_dict

def parse_midi(file_path, bowing_file="None", clef="bass"):
    try:
        midi = MidiFile(file_path)
    except FileNotFoundError:
        print(f"❌ Error: MIDI file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error parsing MIDI file {file_path}: {e}")
        sys.exit(1)

    note_events = []
    bowing_dict = read_bowing_file(bowing_file) if bowing_file != "None" else None

    # Cello string mapping based on MIDI note number
    def get_cello_string(note_number):
        if note_number >= 57: return 'A' # A3
        elif note_number >= 50: return 'D' # D3
        elif note_number >= 43: return 'G' # G2
        else: return 'C' # C2

    last_bow = "down"
    index = 0
    tempo = bpm2tempo(DEFAULT_TEMPO_BPM) # Default tempo in microseconds per beat
    ticks_per_beat = midi.ticks_per_beat if midi.ticks_per_beat > 0 else 480 # Common default

    print(f"MIDI Ticks Per Beat: {ticks_per_beat}")

    absolute_time_sec = 0.0 # Keep track of time in seconds

    for i, track in enumerate(midi.tracks):
        print(f"Processing Track {i}: {track.name}")
        raw_notes = []
        active_notes = {} # Store start time (in seconds) and original note number
        current_tick = 0

        for msg in track:
            # --- Calculate time delta in seconds ---
            delta_ticks = msg.time
            delta_seconds = mido.tick2second(delta_ticks, ticks_per_beat, tempo)
            absolute_time_sec += delta_seconds
            current_tick += delta_ticks

            # --- Update Tempo if message found ---
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
                print(f"  Tempo changed to {tempo2bpm(tempo):.2f} BPM at tick {current_tick} ({absolute_time_sec:.3f}s)")

            # --- Handle Note On ---
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {'start_sec': absolute_time_sec, 'start_tick': current_tick}

            # --- Handle Note Off (or Note On with velocity 0) ---
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_info = active_notes.pop(msg.note)
                    start_time_sec = start_info['start_sec']
                    start_time_tick = start_info['start_tick']
                    end_time_sec = absolute_time_sec
                    end_time_tick = current_tick

                    duration_sec = end_time_sec - start_time_sec
                    duration_ticks = end_time_tick - start_time_tick
                    # Use duration_sec if available, otherwise estimate from ticks
                    if duration_sec < 1e-6: # Avoid zero/negative duration if times are identical
                         duration_sec = mido.tick2second(duration_ticks, ticks_per_beat, tempo)


                    # Apply clef adjustment for string mapping
                    mapping_note = msg.note - 12 if clef == "tenor" else msg.note
                    string = get_cello_string(mapping_note)

                    # Determine bowing
                    if bowing_dict:
                        if index in bowing_dict:
                            bowing = bowing_dict[index]
                            if "-s" in bowing: # Slurred note
                                bowing = last_bow + "-s" # Inherit previous direction for slur
                            else:
                                last_bow = bowing # Update last explicit bow direction
                        else:
                            # Default alternation if index not in file
                            bowing = "up" if last_bow == "down" else "down"
                            last_bow = bowing
                    else:
                        # Default alternation if no file provided
                        bowing = "up" if last_bow == "down" else "down"
                        last_bow = bowing

                    raw_notes.append({
                        'number': msg.note,
                        'note': get_note_name(msg.note),
                        'duration_sec': duration_sec,
                        'string': string,
                        'start_time_sec': start_time_sec,
                        'end_time_sec': end_time_sec,
                        'bowing': bowing,
                        'is_transition': False, # Mark as a real note
                        'event_index': index # Store original index for reference
                    })
                    index += 1
                else:
                     # Note off received for a note not actively tracked (might happen)
                     # print(f"Warning: Received note_off for inactive note {msg.note} at {absolute_time_sec:.3f}s")
                     pass

        # Sort notes by start time (important if MIDI isn't strictly ordered)
        raw_notes.sort(key=lambda x: x['start_time_sec'])

        # Add transitions between notes on different strings
        processed_notes_with_transitions = []
        last_note_end_time = 0.0
        for i, note in enumerate(raw_notes):
             # Check for gap between notes, could represent silence or time for transition
             time_gap = note['start_time_sec'] - last_note_end_time

             # Add the note itself
             processed_notes_with_transitions.append(note)

             # Check if next note exists and requires a string transition
             if i + 1 < len(raw_notes):
                 next_note = raw_notes[i+1]
                 current_string = note['string']
                 next_string = next_note['string']

                 if next_string != current_string:
                     # Estimate transition start/end time - place it right after the current note ends
                     transition_start_time = note['end_time_sec']
                     # Estimate a fixed duration for transition, or use part of the gap if available
                     transition_duration = min(0.2, time_gap) if time_gap > 0.01 else 0.2 # Example: 200ms or use gap
                     transition_end_time = transition_start_time + transition_duration

                     processed_notes_with_transitions.append({
                         'number': 'transition',
                         'note': f"transition {current_string}->{next_string}",
                         'duration_sec': transition_duration,
                         'string': f"{current_string}-{next_string}",
                         'start_time_sec': transition_start_time,
                         'end_time_sec': transition_end_time,
                         'bowing': "transition",
                         'is_transition': True, # Mark as transition
                         'event_index': -1 # No specific note index
                     })
                     last_note_end_time = transition_end_time # Update end time after transition
                 else:
                    last_note_end_time = note['end_time_sec'] # Update end time after same-string note
             else:
                 last_note_end_time = note['end_time_sec'] # Update for the last note

        note_events.extend(processed_notes_with_transitions)

    # Final sort ensures transitions are correctly placed if tracks were processed out of order
    note_events.sort(key=lambda x: x['start_time_sec'])
    print(f"Parsed {len(note_events)} events (notes and transitions).")
    print(note_events)
    return note_events


# --- URScript Generation ---
script_funcs = {
    "A": "a_bow", "D": "d_bow", "G": "g_bow", "C": "c_bow",
    "A-D": "a_to_d", "D-A": "d_to_a", "D-G": "d_to_g", "G-D": "g_to_d",
    "A-G": "a_to_g", "G-A": "g_to_a", "D-C": "d_to_c", "C-D": "c_to_d",
    "G-C": "g_to_c", "C-G": "c_to_g"
}

def get_function_sequence(note_sequence):
    res = ""
    # Use bowing from MIDI parsing result
    for note in note_sequence:
        function = script_funcs.get(note["string"], None)
        if not function:
            print(f"Warning: No script function found for string '{note['string']}'")
            continue

        # Use duration in seconds from MIDI parsing
        # Note: URScript might interpret time differently; scaling might be needed.
        # The '2' multiplier here seems arbitrary - removing it to use MIDI duration directly.
        # Consider if your URScript functions expect a scaled duration.
        note_duration_sec = note["duration_sec"]

        if note['is_transition']:
            res += f"{function}()\n  "
        else:
            # URScript expects boolean for bowing? True for up, False for down? Adjust as needed.
            bowing_value = "True" if "up" in note["bowing"] else "False"
            # Make sure duration isn't negative or extremely small
            safe_duration = max(0.01, note_duration_sec)
            res += f"{function}({bowing_value}, {safe_duration:.4f})\n  stay()\n  "
    # starting_pose = f"a_bow_poses.frog_p"

    # bowings = [
    #     f"movej(p[{starting_pose}[0], {starting_pose}[1], {starting_pose}[2], {starting_pose}[3], {starting_pose}[4], {starting_pose}[5]])", "\ta_bow(True, 1)"]
    # strings = ['a', 'c', 'g', 'd']
    # prev_string = 'a'
    # timings = []

    # for i in range(112):
    #     if i % 9 == 0:
    #         next_string = random.choice(strings)
    #     else:
    #         next_string = prev_string

    #     if next_string != prev_string:
    #         bowings.append("\t" + prev_string + "_to_" + next_string + "()")
    #         time = 0.2
    #     else:
    #         bowings.append(f"\tstay()")
    #         time = max(random.random() * 2, 0.25)
    #         bowings.append(f"\t{next_string}_bow({i % 2 == 0}, {time})")


    #     prev_string = next_string
    
    # res = "\n".join(bowings)
    print(res)
    return res

# Alternative function (if bowing logic is purely alternating in script)
# def get_function_sequence2(note_sequence): ...


# --- Main Execution Logic ---

def send_urscript(urscript, speed_scaling, note_sequence_timed):
    global rtde_running, cntrl_c, data_log

    if not urscript:
        print("❌ Error: No URScript generated!")
        return

    # Prepend speed command
    speed_command = f"set_speed({speed_scaling})\n" # Use set_speed() URScript function
    full_urscript = speed_command + urscript

    if len(full_urscript.encode('utf-8')) > 15000: # Increased limit slightly, but be wary
        print("⚠️ Warning: URScript is large and might exceed robot buffer limits.")

    init_time = -1.0
    last_flag = -1
    last_event_label = "INITIALIZING"
    current_note_info = None # Holds the info of the note currently being played
    current_note_idx = -1

    print("--- URScript to be sent ---")
    print(full_urscript) # Optionally print the full script
    print("--- End URScript ---")
    # exit()
    print(f"Connecting to RTDE at {ROBOT_IP}:{RTDE_PORT}...")

    try:
        if not con.connect():
            print(f"❌ Failed to connect to RTDE on {ROBOT_IP}:{RTDE_PORT}.")
            return
        print(f"✅ RTDE Connected.")

        try:
            con.get_controller_version() # Verify connection
        except Exception as e:
            print(f"⚠️ Warning: Could not get controller version after connect: {e}")
            # Continue anyway, connection might still work for data

        # --- Setup RTDE Output Recipe ---
        print("Setting up RTDE output recipe...")
        if not con.send_output_setup(output_names, output_types):
             print("❌ Failed to setup RTDE output recipe.")
             print("   Check your config XML and ensure recipe name/fields match:")
             print(f"   Requested fields: {output_names}")
             con.disconnect()
             return
        print("✅ RTDE output recipe sent.")

        # --- Start RTDE Data Synchronization ---
        if not con.send_start():
            print("❌ Failed to start RTDE data synchronization.")
            con.disconnect()
            return
        print("✅ RTDE data synchronization started.")
        rtde_running = True

        # --- Send URScript via Primary Interface ---
        print(f"Connecting to UR Primary Interface at {ROBOT_IP}:{UR_PORT} to send script...")
        with socket.create_connection((ROBOT_IP, UR_PORT), timeout=10) as sock:
            print("✅ Connected to Primary Interface.")
            sock.sendall(full_urscript.encode('utf-8'))
            print("✅ Sent URScript.")

        # --- RTDE Data Logging Loop ---
        print("--- Starting RTDE Data Logging (Press Ctrl+C to stop early) ---")
        start_log_time = time.time()
        note = -1

        while rtde_running:
            try:
                # Receive data - this blocks until a packet arrives or timeout
                state = con.receive() # Use binary=True for efficiency if supported by library version

                if state is None:
                    # print("RTDE receive timed out or disconnected?") # Debug print
                    # Check if the script should have finished
                    elapsed_time = time.time() - start_log_time
                    # Estimate total script duration (sum of all note/transition durations)
                    total_estimated_duration = sum(n['duration_sec'] for n in note_sequence_timed) + 5 # Add buffer
                    if elapsed_time > total_estimated_duration:
                       print("Estimated script duration exceeded and no more RTDE data. Stopping.")
                       rtde_running = False
                    continue # Skip to next iteration if no data

                # --- Timestamp Handling ---
                if init_time < 0:
                    init_time = state.timestamp
                    print(f"Initial timestamp received: {init_time}")

                current_rtde_time_sec = state.timestamp - init_time

                # --- Flag Change Detection (Event Logging) ---
                current_flag = state.output_int_register_0
                bow_dir = state.output_int_register_1
                if current_flag != last_event_label:
                    note += 1
                
                event_label = None
                if True:
                    event_label = interpret_flag(current_flag)
                    print(f"[{current_rtde_time_sec:.3f}s] Event: {event_label} (Flag: {current_flag})")
                    last_flag = current_flag
                    last_event_label = event_label # Store the latest event label

                    # --- Simplified Logic to find note associated with START event ---
                    # This is an approximation - assumes flags change near note starts
                    if event_label and "START" in event_label:
                       # Find the next note/transition in sequence after the current time
                       # This is not perfect as the flag might relate to a script function
                       # that slightly precedes the exact MIDI start time.
                       found_note_for_event = False
                       for idx, note_info in enumerate(note_sequence_timed):
                           # Look for a note starting shortly after the current time
                           if note_info['start_time_sec'] >= current_rtde_time_sec - 0.1: # Small tolerance
                               current_note_info = note_info
                               current_note_idx = idx
                               found_note_for_event = True
                               # print(f"    -> Associated with event index: {note_info.get('event_index', 'N/A')}, Type: {'Note' if not note_info['is_transition'] else 'Transition'}")
                               break
                       # if not found_note_for_event:
                           # print("    -> Could not associate START event with a specific upcoming note.")
                           # current_note_info = None # Reset if no clear association

                # --- Find Current Note/Transition based on TIME ---
                # More robust way to find what *should* be playing based on time
                # Start searching from the last known index for efficiency
                temp_current_note = None
                temp_current_idx = -1
                search_start_idx = max(0, current_note_idx) # Start search around the last known note

                for idx in range(search_start_idx, len(note_sequence_timed)):
                     note_info = note_sequence_timed[idx]
                     # Check if the current RTDE time falls within this event's duration
                     if note_info['start_time_sec'] <= current_rtde_time_sec < note_info['end_time_sec']:
                         temp_current_note = note_info
                         temp_current_idx = idx
                         break # Found the note/transition active at this timestamp

                # Update if found, otherwise keep the one associated with the last START event
                if temp_current_note:
                    current_note_info = temp_current_note
                    current_note_idx = temp_current_idx
                # Else: current_note_info might still hold the note from the last START event, or None


                # --- Calculate Remaining Duration ---
                remaining_duration_sec = 0.0
                current_event_type = "None"
                if current_note_info:
                    remaining_duration_sec = max(0.0, current_note_info['end_time_sec'] - current_rtde_time_sec)
                    current_event_type = "Transition" if current_note_info['is_transition'] else "Note"

                # --- Log Data Point ---
                data_log.append({
                    "timestamp_robot": state.timestamp, # Raw robot timestamp
                    "time_elapsed_sec": current_rtde_time_sec,
                    # "event_flag": current_flag,
                    "event_label": last_event_label, # Log the last *detected* event label
                    "current_event_type": current_event_type, # What MIDI event is active by time
                    # "current_note_number": current_note_info.get('number', None) if current_note_info else None,
                    # "current_note_name": current_note_info.get('note', None) if current_note_info else None,
                    "current_string": current_note_info.get('string', None) if current_note_info else None,
                    "remaining_duration_sec": remaining_duration_sec,
                    "bow_direction": bow_dir,   
                    "TCP_pose_x": state.actual_TCP_pose[0],
                    "TCP_pose_y": state.actual_TCP_pose[1],
                    "TCP_pose_z": state.actual_TCP_pose[2],
                    "TCP_pose_rx": state.actual_TCP_pose[3],
                    "TCP_pose_ry": state.actual_TCP_pose[4],
                    "TCP_pose_rz": state.actual_TCP_pose[5],
                    "q_base": state.actual_q[0],
                    "q_shoulder": state.actual_q[1],
                    "q_elbow": state.actual_q[2],
                    "q_wrist1": state.actual_q[3],
                    "q_wrist2": state.actual_q[4],
                    "q_wrist3": state.actual_q[5],
                    "qd_base": state.actual_qd[0],
                    "qd_shoulder": state.actual_qd[1],
                    "qd_elbow": state.actual_qd[2],
                    "qd_wrist1": state.actual_qd[3],
                    "qd_wrist2": state.actual_qd[4],
                    "qd_wrist3": state.actual_qd[5],
                    # "TCP_force_x": state.actual_TCP_force[0],
                    # "TCP_force_y": state.actual_TCP_force[1],
                    # "TCP_force_z": state.actual_TCP_force[2],
                    # "TCP_force_rx": state.actual_TCP_force[3],
                    # "TCP_force_ry": state.actual_TCP_force[4],
                    # "TCP_force_rz": state.actual_TCP_force[5],
                })

                # Simplified End Condition: Check for a known final flag if you have one
                FINAL_EXPECTED_FLAG = 120 # Example: END c_to_g
                # Add a time-based condition as well in case the flag isn't reached
                if current_flag == FINAL_EXPECTED_FLAG and current_rtde_time_sec > 1.0: # Avoid immediate stop on initial flags
                     print(f"Final expected flag ({FINAL_EXPECTED_FLAG}) reached. Stopping RTDE logging soon.")
                     # Optional: add a small delay here before stopping if needed
                     # time.sleep(0.5)
                     # rtde_running = False # Defer stopping to timeout or Ctrl+C for now


            except socket.timeout:
                 print("RTDE receive socket timeout.")
                 continue # Continue trying to receive
            except ConnectionAbortedError:
                 print("RTDE Connection Aborted.")
                 rtde_running = False
                 break
            except Exception as e:
                 print(f"❌ Error in RTDE loop: {e}")
                 # Decide if error is fatal
                 rtde_running = False
                 break

    except socket.timeout:
        print("❌ Socket timeout during connection or sending URScript.")
    except ConnectionRefusedError:
         print(f"❌ Connection refused. Is the robot on ({ROBOT_IP}) and URCap running?")
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt detected. Stopping script and saving data...")
        cntrl_c = True
        rtde_running = False # Signal loop to stop
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("--- Execution/Logging Finished ---")
        if con and con.is_connected():
            print("Pausing RTDE stream...")
            try:
                con.send_pause()
            except Exception as e:
                print(f"⚠️ Warning: Exception during send_pause: {e}")
            print("Disconnecting RTDE...")
            try:
                con.disconnect()
            except Exception as e:
                print(f"⚠️ Warning: Exception during disconnect: {e}")
        else:
             print("RTDE connection was not active or already closed.")
        # Save data regardless of how the loop ended
        save_data(data_log, OUTPUT_LOG_FILENAME)


def save_data(log_data, filename):
    if not log_data:
        print("No data collected to save.")
        return

    print(f"Saving {len(log_data)} data points to {filename}...")
    try:
        # Create DataFrame with specific column order if desired
        # (Ensure columns match keys in the logged dictionary)
        columns = [
            "timestamp_robot", 
            "time_elapsed_sec", 
            # "event_flag", 
            "event_label",
            "current_event_type", 
            # "current_note_number", 
            # "current_note_name",
            "bow_direction",
            "current_string", 
            "remaining_duration_sec",
            "TCP_pose_x", 
            "TCP_pose_y", 
            "TCP_pose_z", 
            "TCP_pose_rx", 
            "TCP_pose_ry", 
            "TCP_pose_rz",
            "q_base", 
            "q_shoulder", 
            "q_elbow", 
            "q_wrist1", 
            "q_wrist2", 
            "q_wrist3",
            "qd_base", 
            "qd_shoulder", 
            "qd_elbow", 
            "qd_wrist1", 
            "qd_wrist2", 
            "qd_wrist3",
            # "TCP_force_x", 
            # "TCP_force_y", 
            # "TCP_force_z", 
            # "TCP_force_rx", 
            # "TCP_force_ry", 
            # "TCP_force_rz"
        ]
        # Filter out any keys in data_log not in columns to avoid errors
        filtered_log_data = [{k: d.get(k, None) for k in columns} for d in log_data]

        df_log = pd.DataFrame(filtered_log_data, columns=columns)
        df_log.to_csv(filename, index=False, float_format='%.6f')
        print(f"✅ Successfully saved log data to '{filename}'.")
    except Exception as e:
        print(f"❌ Error saving data log: {e}")


# --- Script Execution ---
if __name__ == "__main__":
    # print("Parsing MIDI file...")
    use_random_sequence = False         # ← toggle here

    if use_random_sequence:
        note_sequence_timed = generate_random_sequence()
    else:
        note_sequence_timed = parse_midi(MIDI_FILE_PATH, BOWING_FILE, CLEF)

    if not note_sequence_timed:
        print("❌ No note events parsed from MIDI. Exiting.")
        sys.exit(1)

    print("\n--- Parsed Note Sequence (with seconds) ---")
    for note in note_sequence_timed[:10]: # Print first few notes
        print(f"  {note['start_time_sec']:.3f}s - {note['end_time_sec']:.3f}s ({note['duration_sec']:.3f}s): "
              f"{note['note']} ({note['string']}) Bow: {note['bowing']} Transition: {note['is_transition']}")
    print("...\n")


    print("Generating URScript function sequence...")
    function_sequence = get_function_sequence(note_sequence_timed)

    print("Loading URScript template...")
    try:
        with open(SONG_SCRIPT_TEMPLATE, "r") as f:
            script_template = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Song script template not found at {SONG_SCRIPT_TEMPLATE}")
        sys.exit(1)

    # Inject the generated function calls into the template
    starting_pose = f"{note_sequence_timed[0]['string'].lower()}_bow_poses.frog_p"
    final_script = script_template.replace("# $$$ CODE HERE $$$", f"""
        movej(p[{starting_pose}[0], {starting_pose}[1], {starting_pose}[2], {starting_pose}[3], {starting_pose}[4], {starting_pose}[5]])
        {function_sequence}
        """)

    # print("\n--- Generated URScript (snippet) ---")
    # print(function_sequence[:500] + "...") # Print beginning of generated part
    # print("---")

    # Save the generated script for inspection (optional)
    try:
        with open('generated_cello_script2.txt', "w") as test_file:
            test_file.write(final_script)
        print("✅ Saved full generated URScript to 'generated_cello_script.txt'")
    except Exception as e:
        print(f"⚠️ Warning: Could not save generated script file: {e}")


    print("Starting robot execution and data logging...")
    
    send_urscript(final_script, 0.25, []) # Adjust speed scaling (0.1 to 1.0)

    print("--- Script Finished ---")