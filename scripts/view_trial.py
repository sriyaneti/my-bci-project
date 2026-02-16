"""
View one trial:
- prints input_features shape and basic stats
- decodes the 'transcription' char-code array (handles padding zeros)
"""
import argparse
import os
import h5py
import numpy as np

def pick_session(root: str, session: str | None) -> str:
    sessions = sorted([d for d in os.listdir(root) if d.startswith("t15")])
    if not sessions:
        raise FileNotFoundError(f"No sessions found under {root}")
    if session:
        if session not in sessions:
            raise ValueError(f"Session '{session}' not found.")
        return session
    return sessions[0]

def decode_transcription(arr: np.ndarray) -> str:
    # arr is typically uint8 or int codes with 0 padding
    arr = np.asarray(arr).astype(int)
    arr = arr[arr != 0]
    return "".join(chr(c) for c in arr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hdf5_data_final")
    ap.add_argument("--session", default=None)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--trial", default=None, help="e.g., trial_0000")
    ap.add_argument("--ms_per_step", type=int, default=20)
    args = ap.parse_args()

    session = pick_session(args.root, args.session)
    file_path = os.path.join(args.root, session, f"data_{args.split}.hdf5")

    with h5py.File(file_path, "r") as f:
        trials = sorted(list(f.keys()))
        trial_key = args.trial or trials[0]
        if trial_key not in f:
            raise KeyError(f"Trial '{trial_key}' not found. Example: {trials[0]}")
        g = f[trial_key]

        X = g["input_features"][:]
        print("Session:", session)
        print("Split:", args.split)
        print("Trial:", trial_key)
        print("input_features shape:", X.shape, f"(duration≈{X.shape[0]*args.ms_per_step} ms)")
        print("stats: min/mean/max =", float(X.min()), float(X.mean()), float(X.max()))

        if "transcription" in g:
            t = g["transcription"][:]
            print("transcription:", decode_transcription(t))
        else:
            print("transcription: (not present in this split)")

        if "seq_class_ids" in g:
            ids = g["seq_class_ids"][:]
            print("seq_class_ids shape:", ids.shape, "dtype:", ids.dtype)

if __name__ == "__main__":
    main()
