# UavGPR GUI Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current gprMax PySide6 GUI into a focused UavGPR simulation-data workbench that reliably produces raw `.out/_merged.out`, manifests, validation evidence, previews, and MyGPR handoff reports.

**Architecture:** Keep gprMax responsible for simulation generation, metadata, validation, and preview/report orchestration only. Keep processing algorithms and auto-parameter selection in `D:\MyGPR`; gprMax should call stable MyGPR APIs and render outputs, not duplicate algorithm logic. First fix correctness gaps in geometry/audit/manifest behavior, then improve UI workflow, then split the current 3902-line GUI file into smaller modules.

**Tech Stack:** Python 3.10, PySide6, matplotlib Figure/FigureCanvas, h5py, numpy, gprMax CLI, `tools.outputfiles_merge`, MyGPR `core.processing_engine` and later `core.auto_tune_pipeline`.

---

## Current Version Review

Repository state at review start:

- gprMax: `master...origin/master [ahead 2]`, latest commit `2a3f5cf Simplify UAV GUI default workflow`.
- MyGPR: `main...origin/main [ahead 25]`, latest commit `10eb962 Add time power gain method`.
- No staged changes in either repo.
- Preserved uncommitted gprMax files: `CONTEXT.md`, `docs/mygpr_handoff.md`, `docs/uavgpr_baseline_scenarios.md`, `scripts/validate_uavgpr_dataset.py`, `tests/test_validate_uavgpr_dataset.py`.
- Preserved uncommitted MyGPR file: `docs/mygpr_uavgpr_final_workflow_obsidian.md`.

Important current code facts:

- `E:\gprMax\gprMax-v.3.1.7\gprmax_gui_pyside6.py` is 3902 lines and currently owns presets, dataclasses, physics audit, input generation, runner, previews, Qt UI, report threading, and smoke CLI.
- `E:\gprMax\gprMax-v.3.1.7\uavgpr_processing_report.py` is 546 lines and directly builds local background/gain candidate combinations around MyGPR `run_processing_method`.
- Current tests cover handoff, GUI default state, processing report bridge, and UAV presets, but do not yet cover multi-target audit, validator GUI integration, cancellation, report quality gates, or manifest schema version migration.
- Current UI defaults to the UAV baseline and hides advanced settings, which is the right direction, but the left-side form is still a stack of low-level groups rather than a workflow surface.

High-priority risks found:

- `PhysicsAuditor` computes bounds, PML clearance, coverage, and closest two-way time from the primary target only; `extra_targets` are generated into `.in` and manifest, but not fully audited.
- `_closest_two_way_time_ns()` uses `config.scan_step` instead of `config.effective_scan_step`, so time-window estimates can diverge from gprMax integer-cell behavior when the requested step is quantized.
- `use_gpu_check` defaults to checked while hidden under advanced settings; users may unknowingly request GPU and hit fallback messages on ordinary CPU runs.
- `docs/uavgpr_baseline_scenarios.md` still names `air_angled_crack_halfspace`, but that preset was intentionally removed from GUI changes.
- The uncommitted validator and handoff docs are referenced by `CONTEXT.md` but are not committed, so the project contract is not reproducible from `HEAD`.
- `uavgpr_processing_report.py` duplicates candidate selection/scoring logic that MyGPR already has more developed auto-tune modules for. This should be converted to a bridge or clearly marked as a lightweight preview comparator.
- The processing report treats 2-trace smoke output like a meaningful processing candidate ranking. It should emit an insufficiency warning and avoid over-interpreting such runs.
- GUI result state is path-string based. It lacks a structured "dataset readiness" object that knows whether `.in`, `.out`, `_merged.out`, manifest, validator result, previews, and report exist.
- The current preview B-scan x-axis is trace number only; MyGPR handoff needs receiver positions and effective scan step to be visible in preview/report metadata.
- Thread cancellation is absent. A long gprMax run or MyGPR report cannot be cancelled cleanly from the GUI.

## File Map

Files to modify or create across the plan:

- Modify `E:\gprMax\gprMax-v.3.1.7\gprmax_gui_pyside6.py`: short-term fixes and UI wiring; later reduce it to UI orchestration.
- Create `E:\gprMax\gprMax-v.3.1.7\uavgpr_config.py`: dataclasses, presets, small geometry/math helpers after tests are green.
- Create `E:\gprMax\gprMax-v.3.1.7\uavgpr_physics_audit.py`: `PhysicsAuditor`, `AuditReport`, target aggregation.
- Create `E:\gprMax\gprMax-v.3.1.7\uavgpr_input_builder.py`: `.in`, metadata, geometry preview generation.
- Create `E:\gprMax\gprMax-v.3.1.7\uavgpr_runner.py`: gprMax subprocess, merge, HDF5 summary, run metadata.
- Create `E:\gprMax\gprMax-v.3.1.7\uavgpr_manifest.py`: manifest schema constants, path normalization, JSON write/read helpers.
- Modify `E:\gprMax\gprMax-v.3.1.7\uavgpr_processing_report.py`: use MyGPR APIs more cleanly, add trace-count warnings, multi-target ROI support, stable summary fields.
- Add or modify `E:\gprMax\gprMax-v.3.1.7\scripts\validate_uavgpr_dataset.py`: commit the existing validator after reviewing stale assumptions.
- Add or modify `E:\gprMax\gprMax-v.3.1.7\tests\test_validate_uavgpr_dataset.py`: commit validator coverage and extend for new manifest schema.
- Modify `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_realistic_pipe_preset.py`: add multi-target audit and effective-step regressions.
- Modify `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_handoff_contract.py`: assert manifest schema, relative/absolute paths, preview flags, dataset readiness.
- Modify `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_gui_ui_state.py`: assert workflow-oriented default UI and validation/report controls.
- Modify `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_processing_report.py`: assert trace insufficiency warning and multi-target ROI behavior.
- Modify `E:\gprMax\gprMax-v.3.1.7\docs\mygpr_handoff.md`: align with latest manifest fields and MyGPR read order.
- Modify `E:\gprMax\gprMax-v.3.1.7\docs\uavgpr_baseline_scenarios.md`: remove stale preset references and document current accepted baselines.
- Optionally modify `D:\MyGPR\core\auto_tune_pipeline.py` or create `D:\MyGPR\core\gprmax_report_bridge.py` later, but only if gprMax needs a stable MyGPR-side report API.

---

### Task 1: Commit The Existing Contract/Validator Baseline

**Files:**
- Modify: `E:\gprMax\gprMax-v.3.1.7\docs\mygpr_handoff.md`
- Modify: `E:\gprMax\gprMax-v.3.1.7\docs\uavgpr_baseline_scenarios.md`
- Modify: `E:\gprMax\gprMax-v.3.1.7\scripts\validate_uavgpr_dataset.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_validate_uavgpr_dataset.py`
- Do not stage: `E:\gprMax\gprMax-v.3.1.7\scripts\__pycache__\validate_uavgpr_dataset.cpython-310.pyc`

- [ ] **Step 1: Remove stale scenario references from baseline docs**

Update `docs\uavgpr_baseline_scenarios.md` so the accepted baseline table contains only presets currently present in `PRESETS`: `official_cylinder_bscan`, `realistic_pipe_bscan`, `uav_pipe_gain_workflow_bscan`, `air_void_halfspace`, `water_void_halfspace`, and `air_crack_halfspace`.

- [ ] **Step 2: Align handoff docs with current manifest**

Update `docs\mygpr_handoff.md` to list these manifest fields as first-class contract: `schema`, `primary_out_file`, `component`, `simple_targets`, `scan_geometry`, `medium`, `raw_is_unchanged`, `preview_processing_only`, and `primary_out_summary`.

- [ ] **Step 3: Run validator tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_validate_uavgpr_dataset
```

Expected: validator tests pass. If the test module currently fails because it was written against old manifest fields, update only the test expectations and validator schema checks.

- [ ] **Step 4: Run contract tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_handoff_contract tests.test_validate_uavgpr_dataset
```

Expected: both test modules pass.

- [ ] **Step 5: Commit**

Stage only docs, validator script, and validator test:

```powershell
git add -- docs/mygpr_handoff.md docs/uavgpr_baseline_scenarios.md scripts/validate_uavgpr_dataset.py tests/test_validate_uavgpr_dataset.py
git commit -m "Add UavGPR dataset validation contract"
```

---

### Task 2: Fix Multi-Target And Effective-Step Physics Audit

**Files:**
- Modify: `E:\gprMax\gprMax-v.3.1.7\gprmax_gui_pyside6.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_realistic_pipe_preset.py`

- [ ] **Step 1: Add failing tests for extra target audit**

Add tests that create `SimulationConfig(extra_targets=[SimpleTargetSpec(...)])` where the second target violates PML or ground constraints. Assert `PhysicsAuditor().build_report(config).has_errors()` is true and the message mentions the second anomaly.

- [ ] **Step 2: Add failing test for effective scan step in two-way time**

Create a config where `scan_step` is not an integer multiple of `dx`. Assert `report.derived["effective_scan_step"] == config.effective_scan_step` and that `closest_twt_ns` is computed with `config.effective_scan_step`, not the raw requested step.

- [ ] **Step 3: Implement target aggregation**

In `PhysicsAuditor`, replace single-target helper usage with helpers that return all active targets:

```python
def _all_target_bounds(self, config):
    bounds = [self._primary_target_bounds(config)]
    for target in config.extra_targets:
        if target.enabled:
            bounds.append(target.bounds())
    return bounds
```

Then audit each target independently for domain, host half-space, PML clearance, minimum dimension, and scan midpoint coverage. Keep the existing aggregate values in `report.derived`, but add per-target entries such as `targets[0].center_x_m`.

- [ ] **Step 4: Use effective scan step everywhere gprMax will quantize**

In `_closest_two_way_time_ns()`, compute trace positions using `config.effective_scan_step`.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_realistic_pipe_preset
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -- gprmax_gui_pyside6.py tests/test_uavgpr_realistic_pipe_preset.py
git commit -m "Audit all UAV simple targets"
```

---

### Task 3: Version The Manifest And Add Dataset Readiness

**Files:**
- Modify: `E:\gprMax\gprMax-v.3.1.7\gprmax_gui_pyside6.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_handoff_contract.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_validate_uavgpr_dataset.py`

- [ ] **Step 1: Add schema assertions**

In handoff tests, assert:

```python
self.assertEqual(manifest["schema"], "uavgpr_manifest_v1")
self.assertTrue(manifest["raw_is_unchanged"])
self.assertTrue(manifest["preview_processing_only"])
self.assertIn("scan_geometry", manifest)
self.assertIn("medium", manifest)
self.assertIn("simple_targets", manifest)
```

- [ ] **Step 2: Add a dataset readiness object to manifest**

Add `dataset_readiness` to `_write_manifest()`:

```python
"dataset_readiness": {
    "input_file": bool(artifacts.input_path and os.path.exists(artifacts.input_path)),
    "primary_out_file": bool(artifacts.primary_out_path and os.path.exists(artifacts.primary_out_path)),
    "merged_out_file": bool(artifacts.merged_out_path and os.path.exists(artifacts.merged_out_path)),
    "metadata_file": bool(artifacts.metadata_path and os.path.exists(artifacts.metadata_path)),
    "bscan_preview_file": bool(artifacts.bscan_png_path and os.path.exists(artifacts.bscan_png_path)),
    "background_preview_files": bool(artifacts.background_removed_png_path and artifacts.background_removed_gain_png_path),
}
```

- [ ] **Step 3: Include relative paths for portability**

Add `paths_relative_to_output_dir` alongside absolute paths, using `os.path.relpath(path, artifacts.output_dir)` for files inside the output directory.

- [ ] **Step 4: Update validator**

Make `scripts\validate_uavgpr_dataset.py` require `schema == "uavgpr_manifest_v1"` and treat missing `dataset_readiness.primary_out_file` as an error.

- [ ] **Step 5: Run tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_handoff_contract tests.test_validate_uavgpr_dataset
```

- [ ] **Step 6: Commit**

```powershell
git add -- gprmax_gui_pyside6.py tests/test_uavgpr_handoff_contract.py tests/test_validate_uavgpr_dataset.py scripts/validate_uavgpr_dataset.py
git commit -m "Version UavGPR manifest readiness"
```

---

### Task 4: Integrate Dataset Validator Into The GUI

**Files:**
- Modify: `E:\gprMax\gprMax-v.3.1.7\gprmax_gui_pyside6.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_gui_ui_state.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_validate_uavgpr_dataset.py`

- [ ] **Step 1: Add GUI state test for validation controls**

Assert MainWindow exposes `validate_dataset_button`, `validation_status_label`, and that the button is disabled before a run and enabled after a fake successful `BuildArtifacts` with a manifest path.

- [ ] **Step 2: Add `ValidationThread`**

Add a QThread that imports `scripts.validate_uavgpr_dataset.validate_dataset` or its equivalent function and emits structured success/failure messages.

- [ ] **Step 3: Add validation UI**

In the MyGPR/report group or a new "结果检查" group, add:

- `运行数据集检查`
- one-line validation status
- open manifest button
- open report button

- [ ] **Step 4: Run validator automatically after simulation**

In `on_success`, if `artifacts.manifest_path` exists, run validation asynchronously and show pass/warning/error state without blocking the GUI.

- [ ] **Step 5: Run tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_gui_ui_state tests.test_validate_uavgpr_dataset
```

- [ ] **Step 6: Commit**

```powershell
git add -- gprmax_gui_pyside6.py tests/test_uavgpr_gui_ui_state.py tests/test_validate_uavgpr_dataset.py
git commit -m "Add GUI dataset validation action"
```

---

### Task 5: Make Runtime Defaults Safer And Add Cancel Support

**Files:**
- Modify: `E:\gprMax\gprMax-v.3.1.7\gprmax_gui_pyside6.py`
- Add or modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_runner_process.py`

- [ ] **Step 1: Add runner command tests**

Add tests for `_run_process` command construction through a fake spawn function. Assert CPU is default, `--geometry-fixed` is included for multi-trace, `-gpu` is only included when explicitly selected and available.

- [ ] **Step 2: Make CPU the default GUI setting**

Change `self.use_gpu_check.setChecked(True)` to unchecked. Rename the checkbox label to `尝试使用 GPU（可用时）`.

- [ ] **Step 3: Add cancel button**

Add `self.cancel_button` beside progress. Disable it by default. Enable while `RunnerThread` or `ProcessingReportThread` is running.

- [ ] **Step 4: Track subprocess handle**

In `GprMaxRunner`, store the active `subprocess.Popen` handle and implement:

```python
def cancel(self):
    if self.active_process is not None and self.active_process.poll() is None:
        self.active_process.terminate()
```

Thread cancellation should emit a controlled failure message such as `用户取消运行`.

- [ ] **Step 5: Bound process log memory**

Replace unbounded `output_lines.append(text)` with a bounded tail list, for example the last 2000 lines, while still streaming all lines to the GUI log.

- [ ] **Step 6: Run tests and smoke**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_runner_process tests.test_uavgpr_gui_ui_state
.\.venv\Scripts\python.exe gprmax_gui_pyside6.py --smoke-test --smoke-preset uav_pipe_gain_workflow_bscan --output-root $env:TEMP --traces 2
```

- [ ] **Step 7: Commit**

```powershell
git add -- gprmax_gui_pyside6.py tests/test_uavgpr_runner_process.py tests/test_uavgpr_gui_ui_state.py
git commit -m "Make UAV GUI runs safer"
```

---

### Task 6: Improve B-Scan Preview Axes And Result Inspection

**Files:**
- Modify: `E:\gprMax\gprMax-v.3.1.7\gprmax_gui_pyside6.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_handoff_contract.py`
- Add or modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_preview_axes.py`

- [ ] **Step 1: Add preview axis tests**

Create a tiny HDF5 merged file with `/rxs/rx1/Ez` and `/rxs/rx1/Positions`. Assert a new loader returns `data`, `dt`, and `x_axis_m`.

- [ ] **Step 2: Change B-scan figure signature**

Change `create_bscan_figure(data, dt, title=...)` to:

```python
def create_bscan_figure(data, dt, title="原始 Ez B-scan", x_axis_m=None):
    ...
```

If `x_axis_m` is present, use `[x_axis_m[0], x_axis_m[-1], time_ns, 0]` and label x as `距离 x [m]`; otherwise keep trace number fallback.

- [ ] **Step 3: Load x-axis from HDF5**

Add a runner helper that reads `Positions[:, 0]` when available. Use this in GUI B-scan and preview PNG generation.

- [ ] **Step 4: Add preview tabs**

In the B-scan tab, show raw, background removed, and mild gain as selectable sub-tabs after a simulation. Keep raw as default.

- [ ] **Step 5: Run tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_preview_axes tests.test_uavgpr_handoff_contract
```

- [ ] **Step 6: Commit**

```powershell
git add -- gprmax_gui_pyside6.py tests/test_uavgpr_preview_axes.py tests/test_uavgpr_handoff_contract.py
git commit -m "Use spatial axes in UAV B-scan previews"
```

---

### Task 7: Make The Processing Report A MyGPR Bridge Instead Of Local Scoring

**Files:**
- Modify: `E:\gprMax\gprMax-v.3.1.7\uavgpr_processing_report.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_processing_report.py`
- Possibly create: `D:\MyGPR\core\gprmax_report_bridge.py`
- Possibly add: `D:\MyGPR\tests\test_gprmax_report_bridge.py`

- [ ] **Step 1: Add trace-count risk test**

In `tests\test_uavgpr_processing_report.py`, assert that a 2-trace report summary includes:

```python
self.assertIn("warnings", summary)
self.assertIn("insufficient_trace_count_for_ranking", [item["code"] for item in summary["warnings"]])
```

- [ ] **Step 2: Add multi-target ROI test**

Build a manifest with two `simple_targets` and assert summary includes `target_rois` with length 2. The recommendation can still use aggregate scoring, but the report must show that both targets were considered.

- [ ] **Step 3: Add a MyGPR bridge probe**

Check whether `D:\MyGPR\core\auto_tune_pipeline.py` can run a pipeline such as `["subtracting_average_2D", "energy_decay_gain"]` using manifest-derived header context. If it can, replace local ranking with that API. If it cannot, create `D:\MyGPR\core\gprmax_report_bridge.py` as the stable API and move candidate scoring there.

- [ ] **Step 4: Keep gprMax bridge thin**

In gprMax, `uavgpr_processing_report.py` should do only:

- load manifest and HDF5
- build MyGPR context
- call MyGPR report/auto-tune API
- save PNG/HTML artifacts
- record `raw_is_unchanged=True`

- [ ] **Step 5: Run tests in both repos**

```powershell
cd /d D:\MyGPR
python -m pytest tests/test_gprmax_report_bridge.py tests/test_auto_tune_pipeline.py -q

cd /d E:\gprMax\gprMax-v.3.1.7
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_processing_report
```

- [ ] **Step 6: Commit separately**

MyGPR:

```powershell
git add -- core/gprmax_report_bridge.py tests/test_gprmax_report_bridge.py
git commit -m "Add gprMax report bridge API"
```

gprMax:

```powershell
git add -- uavgpr_processing_report.py tests/test_uavgpr_processing_report.py
git commit -m "Use MyGPR report bridge for UAV previews"
```

---

### Task 8: Redesign The Front-End Workflow Surface

**Files:**
- Modify: `E:\gprMax\gprMax-v.3.1.7\gprmax_gui_pyside6.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\tests\test_uavgpr_gui_ui_state.py`

- [ ] **Step 1: Define expected top-level UI groups in test**

Update GUI test to assert top-level visible groups in default mode are exactly:

```python
[
    "输出",
    "场景",
    "测线",
    "异常体",
    "运行与结果",
]
```

- [ ] **Step 2: Merge material/domain controls into a scene group**

Rename `计算域 / 宿主介质` to `场景`. In default mode show only:

- ground surface y
- lift-off
- host preset
- host eps_r
- host sigma
- center frequency

Keep domain size, dx, dy, and time window under advanced.

- [ ] **Step 3: Rename target group**

Rename `目标体` to `异常体`. Default fields:

- target shape
- target preset
- center x
- center y
- radius or width/height depending shape
- enable second anomaly

Keep target internal material name and custom eps/sigma under advanced unless target preset is `custom`.

- [ ] **Step 4: Merge runtime/report/validation into one result group**

Replace separate `运行设置` and `MyGPR 处理报告` default surface with `运行与结果`. Default fields:

- CPU/GPU mode label or checkbox
- generated input status
- simulation output status
- dataset validation status
- MyGPR report status
- buttons: generate input, run simulation, validate, generate report, open result folder

- [ ] **Step 5: Keep advanced options accessible**

The existing `高级参数` checkbox should reveal:

- source waveform group
- Python executable
- geometry-only
- write geometry view
- domain size
- dx/dy/time window
- internal material names

- [ ] **Step 6: Run Qt tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_gui_ui_state
```

- [ ] **Step 7: Commit**

```powershell
git add -- gprmax_gui_pyside6.py tests/test_uavgpr_gui_ui_state.py
git commit -m "Reshape UAV GUI workflow surface"
```

---

### Task 9: Refactor The Monolithic GUI File After Behavior Is Stable

**Files:**
- Create: `E:\gprMax\gprMax-v.3.1.7\uavgpr_config.py`
- Create: `E:\gprMax\gprMax-v.3.1.7\uavgpr_physics_audit.py`
- Create: `E:\gprMax\gprMax-v.3.1.7\uavgpr_input_builder.py`
- Create: `E:\gprMax\gprMax-v.3.1.7\uavgpr_runner.py`
- Create: `E:\gprMax\gprMax-v.3.1.7\uavgpr_manifest.py`
- Modify: `E:\gprMax\gprMax-v.3.1.7\gprmax_gui_pyside6.py`
- Modify imports in UavGPR tests.

- [ ] **Step 1: Move pure config/presets first**

Move `HOST_PRESETS`, `TARGET_PRESETS`, `PRESETS`, helper functions, `MaterialSpec`, `CrackSpec`, `SimulationConfig`, `SimpleTargetSpec`, `BuildArtifacts` into `uavgpr_config.py`.

- [ ] **Step 2: Move physics audit**

Move `AuditMessage`, `AuditReport`, `PhysicsAuditor` into `uavgpr_physics_audit.py`.

- [ ] **Step 3: Move builder and preview-free manifest helpers**

Move `ScenarioBuilder` into `uavgpr_input_builder.py`, importing config and audit classes.

- [ ] **Step 4: Move runner**

Move `GprMaxRunner`, `remove_horizontal_background`, `apply_mild_time_gain`, and B-scan HDF5 loading into `uavgpr_runner.py`.

- [ ] **Step 5: Leave only Qt UI and CLI entrypoint in GUI file**

`gprmax_gui_pyside6.py` should contain:

- `PlotCanvas`
- `RunnerThread`
- `ProcessingReportThread`
- `ValidationThread`
- `MainWindow`
- `build_smoke_config`
- `run_smoke_test`
- `main`

- [ ] **Step 6: Run full UavGPR tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_handoff_contract tests.test_uavgpr_realistic_pipe_preset tests.test_uavgpr_processing_report tests.test_uavgpr_gui_ui_state tests.test_validate_uavgpr_dataset
.\.venv\Scripts\python.exe -m compileall -q gprmax_gui_pyside6.py uavgpr_config.py uavgpr_physics_audit.py uavgpr_input_builder.py uavgpr_runner.py uavgpr_manifest.py uavgpr_processing_report.py gprMax tools tests
```

- [ ] **Step 7: Run smoke**

```powershell
.\.venv\Scripts\python.exe gprmax_gui_pyside6.py --smoke-test --smoke-preset uav_pipe_gain_workflow_bscan --output-root $env:TEMP --traces 2
```

- [ ] **Step 8: Commit**

```powershell
git add -- gprmax_gui_pyside6.py uavgpr_config.py uavgpr_physics_audit.py uavgpr_input_builder.py uavgpr_runner.py uavgpr_manifest.py tests
git commit -m "Split UavGPR GUI implementation modules"
```

---

### Task 10: Add A Full Local Verification Gate

**Files:**
- Create or modify: `E:\gprMax\gprMax-v.3.1.7\scripts\run_uavgpr_gui_checks.ps1`
- Modify: `E:\gprMax\gprMax-v.3.1.7\docs\mygpr_handoff.md`

- [ ] **Step 1: Add one-command local check script**

Create a PowerShell script that runs:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_input_cmd_funcs tests.test_outputfiles_merge tests.test_uavgpr_handoff_contract tests.test_uavgpr_realistic_pipe_preset tests.test_uavgpr_processing_report tests.test_uavgpr_gui_ui_state tests.test_validate_uavgpr_dataset
.\.venv\Scripts\python.exe -m compileall -q gprmax_gui_pyside6.py uavgpr_processing_report.py gprMax tools tests
.\.venv\Scripts\python.exe gprmax_gui_pyside6.py --smoke-test --smoke-preset uav_pipe_gain_workflow_bscan --output-root $env:TEMP --traces 2
```

- [ ] **Step 2: Document the gate**

Add a short "Before handoff to MyGPR" section in `docs\mygpr_handoff.md` telling future agents to run the script and then validate the generated smoke folder with `scripts\validate_uavgpr_dataset.py`.

- [ ] **Step 3: Run the script**

```powershell
.\scripts\run_uavgpr_gui_checks.ps1
```

- [ ] **Step 4: Commit**

```powershell
git add -- scripts/run_uavgpr_gui_checks.ps1 docs/mygpr_handoff.md
git commit -m "Add UavGPR GUI verification gate"
```

---

## Recommended Execution Order

1. Push or otherwise back up current gprMax and MyGPR local commits before large GUI edits.
2. Task 1: commit existing contract/validator baseline.
3. Task 2: fix physics audit correctness for extra targets and effective scan step.
4. Task 3: version manifest and dataset readiness.
5. Task 4: integrate validator into GUI.
6. Task 5: safer runtime defaults and cancel support.
7. Task 6: B-scan preview axes and result inspection.
8. Task 7: thin MyGPR processing report bridge.
9. Task 8: UI workflow surface redesign.
10. Task 9: module split only after behavior tests pass.
11. Task 10: one-command verification gate.

## Verification Matrix

Run this after every generation-path change:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_input_cmd_funcs tests.test_outputfiles_merge
.\.venv\Scripts\python.exe -m unittest tests.test_uavgpr_handoff_contract tests.test_uavgpr_realistic_pipe_preset tests.test_uavgpr_processing_report tests.test_uavgpr_gui_ui_state tests.test_validate_uavgpr_dataset
.\.venv\Scripts\python.exe -m compileall -q gprmax_gui_pyside6.py uavgpr_processing_report.py gprMax tools tests
.\.venv\Scripts\python.exe gprmax_gui_pyside6.py --smoke-test --smoke-preset uav_pipe_gain_workflow_bscan --output-root $env:TEMP --traces 2
```

Run this after MyGPR bridge changes:

```powershell
cd /d D:\MyGPR
python -m pytest tests/test_round2_processing_kernels.py tests/test_runtime_warnings.py tests/test_auto_tune_pipeline.py -q
```

Run this before handing a real generated dataset to MyGPR:

```powershell
.\.venv\Scripts\python.exe scripts\validate_uavgpr_dataset.py <generated-output-folder>
```

## Self-Review

- Spec coverage: code, UI, runtime logic, manifest/handoff, report bridge, tests, validation, and repo hygiene are covered.
- Boundary check: gprMax remains the simulation producer. MyGPR remains the algorithm owner. The only cross-repo plan item is a stable MyGPR bridge API if current MyGPR auto-tune APIs cannot directly serve report generation.
- Risk ordering: correctness and validation come before visual UI redesign and module splitting.
- Placeholder scan: no unspecified implementation placeholders remain; every task has files, concrete actions, and verification commands.
