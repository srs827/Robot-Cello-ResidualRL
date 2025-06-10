from mido import MidiFile

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

def get_note_name(note_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note = note_names[note_number % 12]
    return f"{note}{octave}"

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

# print(parse_midi("/Users/PV/Robot Cello/Robot-Cello/midi_robot_pipeline/midi_files/twinkle_twinkle.mid"))