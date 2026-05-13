$ErrorActionPreference = 'Stop'

Write-Host '=== GPRMax GUI 环境一键修复 ===' -ForegroundColor Cyan
$py = 'D:\Miniconda3\python.exe'
if (-not (Test-Path $py)) {
    $py = (Get-Command python -ErrorAction Stop).Source
}
Write-Host "使用 Python: $py" -ForegroundColor Yellow

& $py -m pip install --upgrade pip
& $py -m pip install numpy matplotlib pillow h5py

Write-Host '--- 依赖检查 ---' -ForegroundColor Cyan
& $py - <<'PY'
import sys
mods = ['numpy','matplotlib','PIL','h5py','tkinter']
print('Python:', sys.executable)
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, '__version__', 'built-in')
        print(f'[OK] {m}: {ver}')
    except Exception as e:
        print(f'[FAIL] {m}: {e}')
        raise
print('环境检查通过。')
PY

Write-Host ''
Write-Host '现在可运行：' -ForegroundColor Green
Write-Host "& '$py' 'E:\gprMax\gprMax-v.3.1.7\gprmax_gui_standalone_v3.py'"
