
import uuid
import time
import random
import shutil
import unittest
import tempfile
from pathlib import Path
from typing import Optional, List

from src.ncpu.trainer import CheckpointTracker

class TestCheckpointer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.test_directory = "test_directory" 
        for file in Path(self.test_directory).rglob("*"):
            if file.is_file():
                file.unlink()
        
    def tearDown(self):
        for file in Path(self.test_directory).rglob("*"):
            if file.is_file():
                file.unlink()
        Path(self.test_directory).rmdir()

    def test_create_files_and_remove_files_keep_them_under_10_any_given_moment(self):
        garbage_limit = 10
        ct = CheckpointTracker(
            checkpoint_dir = self.test_directory,
            garbage_limit = garbage_limit, # garbage collector for checkpoints, if -1 then does nothing
        )

        for n in range(100):
            path = ct.make(f"_{n}_")
            Path(path).touch()
            files = [p for p in Path(self.test_directory).iterdir() if p.is_file()]
            self.assertLessEqual(len(files), garbage_limit)
        self.assertLessEqual(len(files), garbage_limit)


