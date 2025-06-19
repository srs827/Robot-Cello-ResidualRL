from dotenv import load_dotenv
import os


def load_rl_env_var():
    """
    Parse `.env` file in the project root directory and create a dictionary to
    access them.

    :return: dictionary of parsed environment variable
    :raises ValueError: if any of the environment variable is empty
    """

    try:
        load_dotenv(verbose=True)

        env_vars = {
            "TRAJ_LOG_CSV_PATH": os.getenv("TRAJ_LOG_CSV_PATH"),
            "NOTE_SEQ_MIDI_PATH": os.getenv("NOTE_SEQ_MIDI_PATH"),
            "SCENE_XML_PATH": os.getenv("SCENE_XML_PATH"),
            "MODEL_ZIP_PATH": os.getenv("MODEL_ZIP_PATH")
        }

        if not all(env_vars.items()):
            raise ValueError("Missing one of the environment variables")

        return env_vars

    except Exception as e:
        print(f"Error loading environment variables: {e}")
        return None
