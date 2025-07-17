"""
This module provides a cleanup function to reset the FORAGER environment by deleting generated files and directories.
"""

import shutil
from pathlib import Path

# Paths to delete
paths_to_delete = [
    "FORAGER_corpus/heterogenous_integration/chunks",
    "FORAGER_corpus/heterogenous_integration/html_text",
    "FORAGER_corpus/heterogenous_integration/pdf_text",
    "FORAGER_corpus/heterogenous_integration/json",
    "../FORAGER/vector_database"
]

for path_str in paths_to_delete:
    path = Path(path_str)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
            print(f"✅ Deleted directory: {path}")
        else:
            path.unlink()
            print(f"✅ Deleted file: {path}")
    else:
        print(f"⚠️ Path does not exist: {path}")

print("✅ Cleanup complete.")