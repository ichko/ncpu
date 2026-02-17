from datetime import datetime


from pathlib import Path
from typing import Optional
from collections import deque

CHECKPOINT_DIR = "./checkpoints"


class CheckpointTracker:

    def __init__(
        self,
        checkpoint_dir: str = CHECKPOINT_DIR,
        garbage_limit: Optional[
            int
        ] = None,  # garbage collector for checkpoints, if -1 then does nothing
    ):
        self.garbage_limit = garbage_limit
        self.checkpoint_dir = checkpoint_dir
        Path(self.checkpoint_dir).mkdir(exist_ok=True)

        self.file_indexes = deque()

    def make(self, custom_string, timestamp = True):
        if timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{self.checkpoint_dir}/ncpu_{custom_string}_{timestamp}.pth"
        else:
            path = f"{self.checkpoint_dir}/ncpu_{custom_string}.pth"

        if self.garbage_limit:
            self.file_indexes.append(path)

            if len(self.file_indexes) > self.garbage_limit:
                old_path = self.file_indexes.popleft()
                Path(old_path).unlink()

        return path

    def get(self, name : str) -> str:
        return f"{self.checkpoint_dir}/{name}.pth"