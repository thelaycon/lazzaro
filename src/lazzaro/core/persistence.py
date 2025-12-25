import os
import pickle
import shutil
import time
from typing import Any, Optional


class PersistenceManager:
    """
    Handles atomic persistence of the memory system state to disk.

    PersistenceManager ensures that the graph, profile, and system metrics are
    saved safely using a write-rename-backup strategy to prevent data corruption.

    Args:
        db_dir: Directory where the persistence files will be stored.
        filename: Name of the primary pickle file.

    Example:
        ```python
        pm = PersistenceManager(db_dir="my_memories")
        pm.save(my_data)
        loaded_data = pm.load()
        ```
    """

    def __init__(self, db_dir: str = "db", filename: str = "lazzaro.pkl"):
        self.db_dir = db_dir
        self.filename = filename
        self.filepath = os.path.join(db_dir, filename)

        # Ensure db directory exists
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)

    def save(self, data: Any) -> bool:
        """
        Saves data to disk using Pickle with atomic write (write to temp then rename).
        Also maintains a '.bak' file of the previously successfully saved state.
        """
        try:
            temp_path = self.filepath + ".tmp"
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)

            # Atomic rename
            shutil.move(temp_path, self.filepath)

            # Create a backup of the current valid state
            backup_path = self.filepath + ".bak"
            if os.path.exists(self.filepath):
                shutil.copy2(self.filepath, backup_path)

            return True
        except Exception as e:
            print(f"⚠ Error saving persistence: {e}")
            return False

    def load(self) -> Optional[Any]:
        """
        Loads data from disk. Falls back to the '.bak' file if the primary file fails.
        Returns None if no persistence files are found.
        """
        if not os.path.exists(self.filepath):
            return None

        try:
            with open(self.filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠ Error loading persistence: {e}")
            # Try backup
            backup_path = self.filepath + ".bak"
            if os.path.exists(backup_path):
                print("🔄 Attempting to load from backup...")
                try:
                    with open(backup_path, "rb") as f:
                        return pickle.load(f)
                except Exception as e2:
                    print(f"⚠ Error loading backup: {e2}")
            return None

    def exists(self) -> bool:
        """Checks if the primary persistence file exists."""
        return os.path.exists(self.filepath)
