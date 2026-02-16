# Scripts

These scripts help you inspect the NEJM Brain-to-Text T15 dataset locally.

They assume you have:
- data/hdf5_data_final/... (unzipped neural dataset)
- Python 3.11 with: h5py, numpy, matplotlib

Run examples:
- python3.11 scripts/inspect_hdf5.py
- python3.11 scripts/inspect_trials.py --n 10
- python3.11 scripts/view_trial.py --trial trial_0000
- python3.11 scripts/plot_features.py --trial trial_0000 --features 0 1 2 256
