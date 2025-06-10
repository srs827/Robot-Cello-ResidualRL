import socket

ROBOT_IP = "128.46.75.201"
UR_PORT = 30002

# Minimal function body injected into song.script
def generate_test_function_sequence():
    return """a_bow(0, 3.0)
  stay()
  d_bow(0, 3.0)
  stay()
  g_bow(0, 3.0)
  stay()
  c_bow(0, 3.0)
  stay()"""

def send_test_script():
    # Load song.script from file
    with open("/home/skamanski/Robot-Cello-ResidualRL/URScripts/song.script", "r") as f:
        song_script = f.read()

    # Inject bowing commands
    new_body = generate_test_function_sequence()
    song_script = song_script.replace("# $$$ CODE HERE $$$", new_body)

    # Save to text file (for debugging)
    with open("string_test_script.txt", "w") as out:
        out.write(song_script)

    # Send URScript to robot
    with socket.create_connection((ROBOT_IP, UR_PORT), timeout=10) as sock:
        sock.sendall(song_script.encode("utf-8"))
        print("✅ Test script sent to robot.")

if __name__ == "__main__":
    send_test_script()