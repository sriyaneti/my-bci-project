"""
Print shapes and approximate durations for multiple trials.
Assumes each trial group contains 'input_features' (time_steps x 512).
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
            raise ValueError(f"Session '{session}' not found.")
        return session
    return sessions[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hdf5_data_final")
    ap.add_argument("--session", default=None)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=10, help="Number of trials to inspect")
    ap.add_argument("--ms_per_step", type=int, default=20, help="Milliseconds per time step")
    args = ap.parse_args()

    session = pick_session(args.root, args.session)
    file_path = os.path.join(args.root, session, f"data_{args.split}.hdf5")

    with h5py.File(file_path, "r") as f:
        trials = sorted(list(f.keys()))
        print("Session:", session)
        print("Split:", args.split)
        print("Trials in file:", len(trials))
        print()

        for i, trial_key in enumerate(trials[:args.n]):
            X = f[trial_key]["input_features"]
            t_steps = X.shape[0]
            print(f"{i:>3}  {trial_key}  shape={tuple(X.shape)}  duration≈{t_steps*args.ms_per_step} ms")

if __name__ == "__main__":
    main()
