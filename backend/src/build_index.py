import os
import sys

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from index_manager import rebuild_index

def build_index():
    try:
        rebuild_index()
    except Exception as error:
        print(f"Error rebuilding index: {error}")
        sys.exit(1)

if __name__ == "__main__":
    build_index()
