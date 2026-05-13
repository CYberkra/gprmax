# UavGPR Simulation Context

This repository is the gprMax-based simulation data producer for UavGPR work.
Its job is to generate correct, feasible, and progressively more complete
synthetic UavGPR datasets. Downstream data processing and algorithm evaluation
belong in `D:\CDUT-UavGPR-Controller\MyGPR`.

## Project Boundary

- This project owns model construction, gprMax execution, simulation output
  files, geometry previews, simulation metadata, and dataset manifests.
- This project does not own post-processing algorithms, interpretation
  workflows, gprpy pipelines, filtering, migration experiments, detection
  metrics, or controller-side integration. Those belong in MyGPR unless the
  user explicitly asks for a local export adapter.
- GUI output should include gprMax `.out` files. For multi-trace scans, the
  preferred primary output is `_merged.out`; individual numbered `.out` files
  may remain for trace-level audit.

## Domain Terms

- **UavGPR simulated dataset**: A set of generated `.in`, `.out`, optional
  `_merged.out`, preview images, metadata JSON, and manifest JSON describing one
  synthetic survey scenario.
- **A-scan**: One receiver time series from one model run. In gprMax it is a
  one-dimensional receiver dataset such as `/rxs/rx1/Ez`.
- **B-scan**: A matrix assembled from multiple A-scans. In this project it uses
  the layout `samples x traces`.
- **Trace**: One A-scan column in a B-scan. Trace numbering follows gprMax model
  run numbering starting at 1.
- **Requested scan step**: The meter value entered in the GUI.
- **Effective scan step**: The actual step that gprMax will execute after
  converting the requested scan step to an integer number of grid cells.
- **Grid cell step**: The integer source/receiver step stored by gprMax in
  `srcsteps` and `rxsteps`.
- **Primary output**: The `.out` file downstream consumers should read first.
  For one trace it is `name.out`; for multi-trace scans it is
  `name_merged.out`.
- **Manifest**: JSON sidecar that records primary output paths, raw output
  files, important HDF5 attributes, receiver positions, effective scan step, and
  simulation-processing guidance.
- **Ground truth**: The known target geometry, material, position, scan
  geometry, and medium parameters encoded in the generated `.in` and metadata.

## Output Contract

The stable consumer-facing data contract is:

- Primary HDF5 output: `.out`
- Preferred B-scan dataset: `/rxs/rx1/Ez`
- B-scan layout: `samples x traces`
- Time step source: root attribute `dt` in seconds
- Trace positions: `/rxs/rx1/Positions` in merged outputs when available
- Spatial units: meters
- Time units in gprMax attributes: seconds
- Recommended host velocity: derive from host relative permittivity as
  `c / sqrt(eps_r)`

Do not silently treat GUI preview geometry as ground truth. Always prefer the
generated `.in`, manifest, and HDF5 attributes when checking whether a B-scan
matches the model.

## Simulation Quality Rules

- Grid-dependent quantities must be checked against gprMax's integer-cell
  behavior. Scan steps that are not integer multiples of `dx` are quantized.
- Sources, receivers, and targets must avoid the default PML region unless the
  scenario intentionally tests boundary behavior.
- Time windows must cover the expected two-way travel time to the nearest target
  with a margin.
- Synthetic data should remain raw by default. Do not apply dewow, mean-trace
  removal, AGC, migration, or interpretation steps inside this project.
- When adding realism, model or record it explicitly: UAV altitude, nonuniform
  trace spacing, terrain, pose/position error, noise, material variation, and
  antenna approximation.

## Downstream Handoff

MyGPR should be able to consume this project's output without guessing:

- Read `*_manifest.json` first when present.
- Read `primary_out_file` from the manifest.
- Use `/rxs/rx1/Ez` for default B-scan data.
- Use `Positions` or `rxsteps * dx` to build the spatial axis.
- Use `dt` and `Iterations` for the time axis.
- Use metadata and generated `.in` as scenario truth.
