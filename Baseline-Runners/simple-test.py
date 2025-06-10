import socket

ROBOT_IP = "128.46.75.201"
UR_PORT = 30002

# Minimal function body injected into song.script with debug popups
def generate_test_function_sequence():
    return """popup("Starting A string")
  a_bow(0, 3.0)
  stay()
  popup("Starting D string")
  d_bow(0, 3.0)
  stay()
  popup("Starting G string")
  g_bow(0, 3.0)
  stay()
  popup("Starting C string")
  c_bow(0, 3.0)
  stay()
  popup("Finished all strings")"""

def send_test_script():
    try:
        # Load song.script from file
        with open("/home/skamanski/Robot-Cello-ResidualRL/URScripts/song.script", "r") as f:
            song_script = f.read()

        # Inject debug bowing sequence
        new_body = generate_test_function_sequence()
        song_script = song_script.replace("# $$$ CODE HERE $$$", new_body)

        # Save full script for manual inspection
        with open("string_test_script.txt", "w") as out:
            out.write(song_script)

        # Send URScript to robot
        with socket.create_connection((ROBOT_IP, UR_PORT), timeout=10) as sock:
            sock.sendall(song_script.encode("utf-8"))
            print("✅ Test script sent to robot.")

    except Exception as e:
        print(f"❌ Error sending script: {e}")

if __name__ == "__main__":
    send_test_script()
