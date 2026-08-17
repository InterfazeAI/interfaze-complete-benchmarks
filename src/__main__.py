# Some provider SDKs (google-genai) leave non-daemon background threads that keep
# the process alive after the run finishes and every result is already fsync'd.

import os
import sys

from src.cli import main

main()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
