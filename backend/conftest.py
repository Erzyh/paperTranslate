# Ensure the backend directory is importable so `import app` works
# regardless of the pytest invocation directory.
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
