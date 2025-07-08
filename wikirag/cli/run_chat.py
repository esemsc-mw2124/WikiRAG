import os
import subprocess
from pathlib import Path

os.chdir(Path(__file__).parent)  # Ensure we're at project root
subprocess.run(["streamlit", "run", "app.py"])
