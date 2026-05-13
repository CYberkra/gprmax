# ADR 0001: Keep This Repository Focused on UavGPR Simulation Outputs

## Status

Accepted

## Context

The work in this repository is currently driven by UavGPR dataset generation.
The goal is not to build a complete GPR processing suite here. The downstream
processing project is `D:\CDUT-UavGPR-Controller\MyGPR`.

The GUI launched by `start_gui.bat` builds gprMax input files, runs gprMax, and
produces `.out` HDF5 simulation outputs. A separate GPR processing skill
describes useful gprMax HDF5 reading advice, but also includes gprpy workflows
that belong downstream.

## Decision

This repository owns simulation data generation only:

- Generate `.in` files.
- Run gprMax.
- Preserve numbered A-scan `.out` files when useful.
- Produce `_merged.out` as the primary B-scan output for multi-trace scans.
- Write metadata and manifest JSON sidecars that make the output consumable by
  MyGPR without guesswork.
- Produce preview images for human inspection.

This repository does not own:

- gprpy processing pipelines.
- dewow, filtering, gain, migration, detection, or interpretation algorithms.
- MyGPR controller integration.
- Algorithm evaluation metrics.

## Consequences

- GUI changes should improve correctness, feasibility, reproducibility, and
  manifest quality of simulated UavGPR datasets.
- `.out` and `_merged.out` remain the main deliverables.
- Any gprMax-specific insight from external skills should be converted into
  generation-time metadata, validation, or documentation, not into a local
  processing pipeline.
- When a feature request crosses into downstream processing, document the output
  contract here and implement the algorithm in MyGPR.

## Verification Expectations

Before declaring a generation-path change complete, run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_input_cmd_funcs tests.test_outputfiles_merge
.\.venv\Scripts\python.exe -m compileall -q gprmax_gui_pyside6.py gprMax tools tests
.\.venv\Scripts\python.exe gprmax_gui_pyside6.py --smoke-test --output-root $env:TEMP --traces 2
```

For changes involving scan geometry, also inspect the generated HDF5 positions
and confirm that GUI preview geometry, `.in` commands, `rxsteps/srcsteps`, and
manifest effective step agree.
