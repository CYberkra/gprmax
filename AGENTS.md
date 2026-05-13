# AGENTS.md

This file provides guidance for agentic coding agents working in the gprMax repository.

## Local UavGPR Mission

This checkout is currently used as the gprMax-based simulation data producer
for UavGPR. The priority is to generate correct, feasible, reproducible, and
eventually richer synthetic UavGPR datasets.

The downstream data processing project is `D:\CDUT-UavGPR-Controller\MyGPR`.
Do not move filtering, gprpy processing, migration experiments, detection
metrics, or controller-side algorithm work into this repository unless the user
explicitly asks for a local adapter. This project should output gprMax `.out`
files, optional `_merged.out` B-scans, preview images, metadata JSON, and
manifest JSON sidecars.

Read `CONTEXT.md` before making non-trivial GUI, output, or dataset-generation
changes. Read `docs/adr/0001-uavgpr-simulation-output-boundary.md` before
adding processing-related features.

## Agent Workflow For This Project

- Treat `gprmax_gui_pyside6.py` and `start_gui.bat` as the active UavGPR
  simulation GUI path.
- Keep `.out` / `_merged.out` as the primary deliverables. Sidecars should make
  those outputs easier for MyGPR to consume; they should not replace them.
- When borrowing ideas from external GPR processing skills, only fold in the
  gprMax-output parts that improve generation correctness, manifests,
  reproducibility, or validation. Do not add gprpy as a GUI dependency here.
- Check grid quantization explicitly. User-entered distances such as scan step
  must match gprMax's integer-cell behavior before claiming geometry and B-scan
  outputs correspond.
- Use behavior-level tests for output contracts. Prefer tests that create or
  inspect real HDF5 files over tests that only check implementation details.
- For bug reports, first build a fast feedback loop that reproduces the issue:
  generated `.in`, smoke-test command, HDF5 attribute check, or preview/output
  comparison.
- For generation-path changes, run at least:

```bash
python -m unittest tests.test_input_cmd_funcs tests.test_outputfiles_merge
python -m compileall -q gprmax_gui_pyside6.py gprMax tools tests
python gprmax_gui_pyside6.py --smoke-test --output-root %TEMP% --traces 2
```

On PowerShell, use `.\.venv\Scripts\python.exe` for the Python executable.

## Project Overview

gprMax is an electromagnetic wave propagation simulator using the Finite-Difference Time-Domain (FDTD) method. It solves Maxwell's equations in 3D using the Yee cell discretization. Originally designed for Ground Penetrating Radar (GPR) but applicable to many EM problems. Written in Python 3 with Cython (CPU/OpenMP) and CUDA (GPU).

## Build & Setup

```bash
conda env create -f conda_env.yml
conda activate gprMax

# Build (compiles .pyx Cython extensions in-place)
python setup.py build

# Install
python setup.py install

# Clean compiled artifacts (.c, .pyd, .so, build dirs)
python setup.py cleanall

# Rebuild from scratch after changes
python setup.py cleanall && python setup.py build && python setup.py install

# Build without re-Cythonizing (compile existing .c files only)
python setup.py build --no-cython
```

## Running Simulations

```bash
# Basic usage
python -m gprMax path_to/input_file.in

# B-scan with 60 traces
python -m gprMax input_file.in -n 60

# Restart from trace 45
python -m gprMax input_file.in -n 15 -restart 45

# GPU (device IDs 0 and 1)
python -m gprMax input_file.in -gpu 0 1

# MPI task farm (60 traces, 61 MPI tasks)
python -m gprMax input_file.in -n 60 -mpi 61

# Geometry check only (no simulation)
python -m gprMax input_file.in --geometry-only

# Fixed geometry (only src/rx positions change between runs)
python -m gprMax input_file.in -n 60 --geometry-fixed
```

## Input File Format (`.in`)

ASCII text with `#command: params` syntax. SI units throughout (metres, seconds, Hertz). Commands are case-sensitive. Essential commands:

```
#domain: x_size y_size z_size       # model size in metres
#dx_dy_dz: dx dy dz                 # spatial discretization
#time_window: seconds               # or use iteration count
#waveform: type amplitude freq id   # e.g. gaussiandotnorm 1 1.5e9 my_pulse
#hertzian_dipole: z x y z my_pulse  # source
#rx: x y z                          # receiver
```

Python scripting blocks: `#python:` ... `#end_python:`. Access constants `c`, `e0`, `m0`, `z0` and variables `current_model_run`, `inputfile`, `number_model_runs`. Use `from gprMax.input_cmd_funcs import *` for functional API.

## Output Format

HDF5 files (`.out`) with structure:
```
/  (root attrs: gprMax version, Title, Iterations, nx_ny_nz, dx_dy_dz, dt)
/rxs/rx1/  (Name, Position, Ex, Ey, Ez, Hx, Hy, Hz)
/srcs/src1/  (Type, Position)
```

Geometry views use VTK format (`.vti`/`.vtp`), viewable in ParaView.

## Tools (Post-Processing)

```bash
# Plot A-scan (all field components at a receiver)
python -m tools.plot_Ascan outputfile.out

# Plot A-scan with FFT
python -m tools.plot_Ascan outputfile.out --outputs Ez -fft

# Merge A-scans into B-scan
python -m tools.outputfiles_merge basename

# Plot B-scan
python -m tools.plot_Bscan merged_output.out Ez

# Plot antenna parameters (impedance, s11, s21)
python -m tools.plot_antenna_params outputfile.out

# Plot source waveform
python -m tools.plot_source_wave ricker 1 1.5e9 3e-9 1.926e-12 -fft

# Convert old input files to new syntax
python -m tools.inputfile_old2new old_file.in
```

## Testing

### Unit tests (unittest)
```bash
python -m tests.test_input_cmd_funcs          # all tests
python -m unittest tests.test_input_cmd_funcs.My_input_cmd_funcs_test.test_rx  # single test
```

### Model comparison tests
```bash
python -m tests.test_models
```
Edit `test_models.py` to set `basepath` (`models_basic`, `models_advanced`, `models_pmls`) and `testmodels` list.

### Experimental tests
```bash
python -m tests.test_experimental modelfile realfile output
```

No pytest. Tests run via `python -m` or `python -m unittest`.

## Code Style

### File header
Every `.py` and `.pyx` file must start with the standard GPLv3 copyright header (17 lines). Copy from any existing source file.

### Imports
- Order: standard library, third-party, local project
- Individual `from X import Y` style for project-internal imports
- `colorama.init()` called at module level immediately after import

### Naming
- **Classes:** PascalCase — `FDTDGrid`, `Material`, `PML`
- **Functions/methods:** snake_case — `run_std_sim`, `calculate_update_coeffsH`
- **Constants/types:** lowercase — `floattype` (`np.float32`), `complextype` (`np.complex64`)
- **Module-level variables:** snake_case

### Docstrings
Google-style (sphinx.ext.napoleon compatible):
```python
def func(arg1, arg2):
    """Short description.

    Args:
        arg1 (type): Description.
        arg2 (type): Description.

    Returns:
        result (type): Description.
    """
```

### Strings, Classes, Comments
- Prefer `.format()` over f-strings
- Old-style `class Grid(object):` inheritance exists
- Section separators: blocks of `#` characters
- Do not add comments/docstrings unless requested

### Error handling
- `GeneralError` for runtime errors, `CmdInputError` for invalid user commands
- Both subclass `ValueError`, print in red via colorama
- Example: `raise GeneralError('message')`

### No linting tools configured
No flake8, black, ruff, mypy, or pylint. Match surrounding code style. Indent with 4 spaces.

## Cython & CUDA Notes

- `.pyx` — Cython extensions compiled to C (CPU/OpenMP)
- `.pxd` — Cython declaration headers (like C `.h`)
- `*_gpu.py` — CUDA kernel launchers using Numba CUDA
- After modifying `.pyx`, rebuild with `python setup.py build`
- `--no-cython` compiles existing `.c` files without re-Cythonizing

## Parallelization

### OpenMP (Shared Memory)
- Used for FDTD solver loops
- Default: use all physical CPU cores
- Override with `#num_threads: N` in input file or `OMP_NUM_THREADS` env var

### MPI (Distributed Task Farm)
- Used to farm out independent models (e.g., B-scan traces)
- Mixed-mode OpenMP/MPI: MPI distributes models, OpenMP threads each model
- Example: `python -m gprMax model.in -n 60 -mpi 61` (60 workers + 1 master)
- Alternative (no spawn): `python -m gprMax model.in -n 60 --mpi-no-spawn`
- Requires `mpi4py`: `pip install mpi4py`

### GPU (CUDA)
- Requires NVIDIA CUDA-Enabled GPU
- Install CUDA Toolkit and `pycuda`: `pip install pycuda`
- Usage: `python -m gprMax model.in -gpu` or `-gpu 0 1 2 3` for specific devices
- Combine with MPI: `python -m gprMax model.in -n 60 -mpi 5 -gpu 0 1 2 3`

## FDTD Modelling Rules

- **Spatial discretization:** step should be <= 1/10 of smallest wavelength: `Δl = λ/10`
- **PML distance:** keep sources and targets >= 15 cells from PML boundaries
- **Air gap:** include >= 15-20 cells of free space above sources
- **2D mode:** set one domain dimension equal to its spatial step (TMz mode)
- **Dielectric smoothing:** on by default, disable per-object with `n` flag

## User Libraries

Pre-defined antenna models in `user_libs/antennas/`:
- `GSSI.py` — GSSI 1.5GHz (Model 5100), GSSI 400MHz
- `MALA.py` — MALA 1.2GHz

Usage:
```python
#python:
from user_libs.antennas.GSSI import antenna_like_GSSI_1500
antenna_like_GSSI_1500(0.125, 0.094, 0.100, resolution=0.002)
#end_python:
```

## Architecture

| Directory | Purpose |
|-----------|---------|
| `gprMax/` | Main package — FDTD solver, input processing, materials, fields, PML, sources, receivers |
| `gprMax/pml_updates/` | PML boundary update equations (Cython + CUDA) |
| `tests/` | Test scripts and reference model output files |
| `tools/` | Post-processing utilities (plot A/B-scans, merge output files) |
| `user_libs/` | User-contributed libraries (antennas, materials, optimization) |
| `user_models/` | Example `.in` input files and reference `.out` files |
| `docs/` | Sphinx documentation source |

Key files:
- `gprMax/gprMax.py` — entry point (`main()`, `api()`, `run_bscan_sim()`, `run_std_sim()`)
- `gprMax/model_build_run.py` — model building and execution
- `gprMax/grid.py` — `Grid` and `FDTDGrid` classes
- `gprMax/materials.py` — `Material` class
- `gprMax/constants.py` — physical constants, NumPy/CUDA dtypes
- `gprMax/exceptions.py` — custom exception classes
- `gprMax/input_cmd_funcs.py` — functional API for input files
- `gprMax/input_cmds.py` — input command parsing and processing
- `gprMax/fields.py` — field component updates
- `gprMax/sources.py` — source definitions (Hertzian dipole, voltage source, transmission line)

## Platform Notes

- Linux: Matplotlib backend switched to `'agg'` (headless) in tests
- GPU: requires NVIDIA CUDA; detected via `utilities.detect_check_gpus()`
- MPI: available for HPC clusters (`-mpi` flag)
- Windows: compiled extensions are `.pyd`; Linux/Mac: `.so`
- macOS: requires gcc (Homebrew) for OpenMP support
- Input/output: uses HDF5 for output, VTK for geometry views
