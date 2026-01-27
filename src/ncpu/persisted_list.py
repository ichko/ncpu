import os
import pickle
import uuid


class PersistedList:
    FILE_DIGITS = 6

    def __init__(self, base_path):
        self.dirpath = os.path.abspath(base_path)
        os.makedirs(self.dirpath, exist_ok=True)
        self._ids = []

        for fname in sorted(os.listdir(self.dirpath)):
            if fname.endswith(".pkl"):
                name = os.path.splitext(fname)[0]
                try:
                    num = int(name)
                    self._ids.append(num)
                except ValueError:
                    continue
        self._ids.sort()
        self._next_id = max(self._ids, default=-1) + 1

    @classmethod
    def new_list(cls, base_path):
        base_path = os.path.abspath(base_path)
        new_id = str(uuid.uuid4())
        uuid_dir = os.path.join(base_path, new_id)
        return cls(uuid_dir)

    def append(self, item):
        el_id = self._next_id
        fname = f"{el_id:0{self.FILE_DIGITS}d}.pkl"
        path = os.path.join(self.dirpath, fname)
        with open(path, "wb") as f:
            pickle.dump(item, f)
        self._ids.append(el_id)
        self._next_id += 1
        return el_id

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx: int):
        el_id = self._ids[idx]
        fname = f"{el_id:0{self.FILE_DIGITS}d}.pkl"  # use leading zeros for filename
        path = os.path.join(self.dirpath, fname)
        if not os.path.exists(path):
            raise KeyError(f"Element with id {el_id} does not exist.")
        with open(path, "rb") as f:
            return pickle.load(f)


def _sanity_check():
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp()
    try:
        plist = PersistedList.new_list(tmp_dir)
        values = ["a", "b", 123, {"x": 1}]
        for v in values:
            plist.append(v)

        for idx, v in enumerate(values):
            assert plist[idx] == v
        print("✅ Sanity check passed successfully")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    _sanity_check()
