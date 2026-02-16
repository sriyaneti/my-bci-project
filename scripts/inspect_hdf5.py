"""
Inspect the HDF5 dataset layout:
- lists sessions under data/hdf5_data_final
- lists trial keys in one session's data_train.hdf5
- prints datasets inside a chosen trial group
"""
import argparse
import os
import h5py

def pick_session(root: str, session: str | None) -> str:
    sessions = sorted([d for d in os.listdir(root) if d.startswith("t15")])
    if not sessions:
        raise FileNotFoundError(f"No sessions found under {root}")
    if session:
        if session not in sessions:
            raise ValueError(f"Session '{session}' not found. Available: {sessions[:10]} ...")
        return session
    return sessions[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hdf5_data_final", help="Path to hdf5_data_final/")
    ap.add_argument("--session", default=None, help="Session folder name, e.g., t15.2023.08.13")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--trial", default=None, help="Trial key to inspect, e.g., trial_0000")
    args = ap.parse_args()

    root = args.root
    session = pick_session(root, args.session)
    file_path = os.path.join(root, session, f"data_{args.split}.hdf5")

    print("Root:", root)
    print("Session:", session)
    print("File:", file_path)

    with h5py.File(file_path, "r") as f:
        trial_keys = sorted(list(f.keys()))
        print(f"\n# Trials in file: {len(trial_keys)}")
        print("First 10 trial keys:", trial_keys[:10])

        trial = args.trial or trial_keys[0]
        if trial not in f:
            raise KeyError(f"Trial '{trial}' not found in file.")
        print(f"\n# Contents of {trial}:")
        for k in f[trial].keys():
            obj = f[trial][k]
            shape = getattr(obj, "shape", None)
            print(f" - {k}  shape={shape}  dtype={getattr(obj, 'dtype', None)}")

if __name__ == "__main__":
    main()
