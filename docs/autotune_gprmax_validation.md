# AutoTune gprMax Validation Ground Truth Metadata

This document defines the minimal ground truth metadata format used to validate
AutoTune behavior on gprMax-generated UavGPR simulations. The file records what
the simulation contains and which regions should be used for target/background
metrics. It does not define or implement machine learning.

The metadata is a sidecar file stored next to a generated gprMax dataset, for
example:

```text
examples/gprmax/sim_pipe_001/ground_truth.yaml
```

The schema is versioned separately from the GUI manifest:

```text
docs/schemas/gprmax_ground_truth.schema.json
```

## Scope

This format is for AutoTune paper validation and benchmark reporting. It should
be read by downstream validation code when calculating metrics such as CNR,
background energy, target localization error, and target preservation.

It does not replace the gprMax `.out` or `_merged.out` file. It also does not
replace the UavGPR manifest. The manifest describes generated files and scan
metadata; this ground truth file describes evaluation regions and known target
truth.

## Required Fields

- `schema`: schema identifier, currently `gprmax_ground_truth_v1`.
- `dataset_id`: stable dataset identifier used in reports and tables.
- `model_file`: relative or absolute path to the gprMax `.in` model file.
- `output_file`: relative or absolute path to the primary `.out` or
  `_merged.out` file used by AutoTune validation.
- `target_roi.trace_range`: inclusive trace index range covering the target
  response.
- `target_roi.sample_range`: inclusive sample index range covering the target
  response.
- `background_roi.trace_range`: inclusive trace index range used for background
  comparison.
- `background_roi.sample_range`: inclusive sample index range used for
  background comparison.
- `target.type`: target class, such as `pipe`, `void`, `crack`, or `box`.
- `target.depth_m`: target depth below the ground surface in meters.
- `target.material`: target material name, for example `pec`.
- `metrics`: metric contract for AutoTune evaluation.

## ROI Convention

Trace and sample ranges are zero-based and inclusive:

```yaml
target_roi:
  trace_range: [42, 48]
  sample_range: [720, 860]
```

The first value must be less than or equal to the second value. Validation code
should clip or reject ranges outside the loaded B-scan shape rather than
silently wrapping indices.

## Metric Contract

The `metrics` object records which validation metrics should be calculated and,
when useful, stores expected or reference values from a fixed baseline run.
Values may be `null` when a metric is required but not yet computed.

Recommended metric keys:

- `cnr_db`: contrast-to-noise ratio between `target_roi` and `background_roi`.
- `background_energy`: mean squared amplitude in `background_roi`.
- `target_energy`: mean squared amplitude in `target_roi`.
- `localization_error_trace`: absolute trace-index error between expected and
  detected target center.
- `localization_error_sample`: absolute sample-index error between expected and
  detected target center.

AutoTune validation should treat this file as ground truth metadata, not as a
processing recipe. Processing method selection, parameter search, and scoring
logic remain in MyGPR.
