$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Set-Location $RepoRoot

& $Python -m unittest `
    tests.test_input_cmd_funcs `
    tests.test_outputfiles_merge `
    tests.test_uavgpr_handoff_contract `
    tests.test_uavgpr_realistic_pipe_preset `
    tests.test_uavgpr_processing_report `
    tests.test_uavgpr_gui_ui_state `
    tests.test_validate_uavgpr_dataset `
    tests.test_uavgpr_runner_process `
    tests.test_uavgpr_preview_axes

& $Python -m compileall -q `
    gprmax_gui_pyside6.py `
    uavgpr_config.py `
    uavgpr_physics_audit.py `
    uavgpr_input_builder.py `
    uavgpr_runner.py `
    uavgpr_manifest.py `
    uavgpr_processing_report.py `
    gprMax `
    tools `
    scripts `
    tests

& $Python gprmax_gui_pyside6.py `
    --smoke-test `
    --smoke-preset uav_pipe_gain_workflow_bscan `
    --output-root $env:TEMP `
    --traces 2
