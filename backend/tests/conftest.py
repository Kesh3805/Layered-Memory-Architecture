"""Pytest conftest — ensure backend/ is importable for flat module imports."""

import sys
from pathlib import Path

# Add backend/ to sys.path so `import policy`, `from llm.classifier import ...` etc. work
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
