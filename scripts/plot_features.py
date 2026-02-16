"""
Plot selected feature indices over time for one trial (terminal-only).
"""
import argparse
import os
import h5py
import matplotlib.pyplot as plt

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
    ap.add_argument("--trial", default=None, help="e.g., trial_0000")
    ap.add_argument("--features", nargs="+", type=int, default=[0, 1, 2, 256], help="Feature indices to plot")
    ap.add_argument("--ms_per_step", type=int, default=20)
    args = ap.parse_args()

    session = pick_session(args.root, args.session)
    file_path = os.path.join(args.root, session, f"data_{args.split}.hdf5")

    with h5py.File(file_path, "r") as f:
        trials = sorted(list(f.keys()))
        trial_key = args.trial or trials[0]
        X = f[trial_key]["input_features"][:]

    for idx in args.features:
        plt.plot(X[:, idx], label=f"feature {idx}")

    plt.xlabel(f"Time steps ({args.ms_per_step} ms each)")
    plt.ylabel("Normalized activity")
    plt.title(f"{session} {args.split} {trial_key}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
