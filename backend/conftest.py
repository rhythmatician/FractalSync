# Ensure backend `src` is importable when running pytest from repository root
import os
import sys

# Insert the backend directory at front of sys.path so tests can import `src.*`
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
