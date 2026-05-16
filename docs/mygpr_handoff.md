# MyGPR Minimum Read Contract

This repository produces gprMax simulation datasets for UavGPR. MyGPR should
read the generated files rather than infer geometry from GUI screenshots.

## Required Read Order

1. Read `*_manifest.json`.
2. Read `primary_out_file` from the manifest.
3. Open the HDF5 `.out` file.
4. Read `/rxs/rx1/Ez`.
5. Read root attribute `dt` and build the time axis in seconds.
6. Read `/rxs/rx1/Positions` when present and use receiver x positions as the
   spatial axis. If `Positions` is absent for a single trace, use the receiver
   `Position` attribute.
7. Read `metadata_file` for scenario truth: target shape, material, position,
   scan geometry, host permittivity, and effective scan step.

## Minimal Python Example

```python
import glob
import json

import h5py
import numpy as np


output_dir = r"path\to\uavgpr_output"
manifest_path = glob.glob(output_dir + r"\*_manifest.json")[0]

with open(manifest_path, "r", encoding="utf-8") as fobj:
    manifest = json.load(fobj)

with h5py.File(manifest["primary_out_file"], "r") as fobj:
    ez = np.asarray(fobj["/rxs/rx1/Ez"][:], dtype=np.float32)
    if ez.ndim == 1:
        ez = ez[:, np.newaxis]

    dt = float(fobj.attrs["dt"])
    time_axis_s = np.arange(ez.shape[0], dtype=np.float64) * dt

    rx_group = fobj["/rxs/rx1"]
    if "Positions" in rx_group:
        positions = np.asarray(rx_group["Positions"][:], dtype=np.float64)
        x_axis_m = positions[:, 0]
    else:
        x_axis_m = np.asarray([rx_group.attrs["Position"][0]], dtype=np.float64)

with open(manifest["metadata_file"], "r", encoding="utf-8") as fobj:
    metadata = json.load(fobj)

scenario_truth = metadata["config"]
```

`Ez` uses `samples x traces` layout after the single-trace reshape above.
`time_axis_s` is in seconds. `x_axis_m` is in meters.

## Manifest Contract

MyGPR should treat these manifest fields as the first-class read contract:

- `schema`: manifest schema version, expected to be `uavgpr_manifest_v1`.
- `primary_out_file`: raw gprMax `.out` or `_merged.out` to process.
- `component`: receiver dataset component, currently `Ez`.
- `simple_targets`: scenario truth for simple anomaly targets.
- `scan_geometry`: source/receiver positions, effective scan step, trace count,
  ground surface, and UAV lift-off.
- `medium`: host material name, relative permittivity, conductivity, and
  recommended velocity.
- `raw_is_unchanged`: confirms the primary HDF5 output was not processed in this
  repository.
- `preview_processing_only`: confirms background removal and gain are only PNG
  previews/reports.
- `primary_out_summary`: selected HDF5 attributes and receiver dataset shapes.

Processing choices such as filtering, gain, migration, detection metrics, and
controller integration belong in `D:\CDUT-UavGPR-Controller\MyGPR`, not in this
simulation producer.

## Before Handoff To MyGPR

Run the local GUI gate before handing a generated dataset to MyGPR:

```powershell
.\scripts\run_uavgpr_gui_checks.ps1
```

For a specific generated output folder, run the validator directly:

```powershell
.\.venv\Scripts\python.exe scripts\validate_uavgpr_dataset.py <generated-output-folder>
```

The handoff is ready only when the script passes, the manifest validates, and
`primary_out_file` points to the raw `.out` or `_merged.out` file.
