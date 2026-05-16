# UavGPR Baseline Simulation Scenarios

This file defines the first baseline scenarios for generating UavGPR simulation
datasets in this gprMax checkout. The goal is to create correct and explainable
`.out` datasets first, then expand realism after the output contract is stable.

## Baseline Set

| Scenario | GUI preset | Target | Expected use |
| --- | --- | --- | --- |
| Metal cylinder | `official_cylinder_bscan` or `realistic_pipe_bscan` | PEC cylinder | Fast sanity case and pipe-like hyperbola case |
| UAV raw workflow | `uav_pipe_gain_workflow_bscan` | PEC cylinder below one clear ground surface | Default MyGPR auto-parameter validation baseline |
| Air void | `air_void_halfspace` | Free-space circular void | Low-permittivity anomaly baseline |
| Water-filled void | `water_void_halfspace` | Water-filled circular void | High-permittivity anomaly baseline |
| Horizontal crack | `air_crack_halfspace` | Thin horizontal air crack | Thin-layer response baseline |

`official_cylinder_bscan` remains the smallest quick-check model. Use
`realistic_pipe_bscan` when the dataset should look closer to a UavGPR pipe
survey, with a longer time window, larger domain, and nonzero antenna offset.
Use `uav_pipe_gain_workflow_bscan` as the default MyGPR handoff dataset: one
ground surface, uniform dry soil, UAV lift-off, and one simple anomaly.

## Generation Contract

For every accepted baseline run, keep these files together in one output
directory:

- generated `.in`
- raw gprMax `.out` files
- `_merged.out` for multi-trace B-scans
- geometry preview PNG
- B-scan preview PNG when simulation was run
- `*_metadata.json`
- `*_manifest.json`

The downstream primary output is always the manifest field
`primary_out_file`. For one trace it points to `name.out`; for multi-trace
surveys it points to `name_merged.out`.

## Acceptance Gate

Run the validator before handing a folder to MyGPR:

```bash
python scripts/validate_uavgpr_dataset.py path\to\output_folder
```

The first baseline dataset batch should not be treated as usable until the
validator reports no errors.
