# CVAT review environment snapshot

This snapshot records the local `.venvs/cvat273` environment used during review.
Python: 3.11.15. CVAT SDK: 2.73.0. The installed package versions are recorded in
`requirements-cvat-review.txt` at the repository root, including packaging tools.
This is an observed package snapshot, not a hash-locked or tested portable environment.
It does not provide the Stage 4/Stage 5 PyTorch, OpenCV, H5 or CUDA environment.

From the repository root, using an available Python 3.11 interpreter:

```bash
python3.11 -m venv .venvs/cvat273
.venvs/cvat273/bin/python -m pip install -r requirements-cvat-review.txt
```

These commands are for recreating a missing environment. Do not recreate an existing
environment as part of committing the repository. Installation and CVAT connectivity
must be verified on the target system; no installation was run during patch preparation.
The `.venvs/` directory remains local and is excluded from version control.
