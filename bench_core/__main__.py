import os
import sys

from bench_core.cli import main

main()
# Some provider SDKs (google-genai) leave non-daemon background threads that keep
# the process alive after the run finishes and every result is already fsync'd.
# This is a one-shot CLI with nothing left to do, so exit hard rather than hang a
# CI job until its timeout. A SystemExit from an error propagates before this and
# keeps its non-zero code.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
