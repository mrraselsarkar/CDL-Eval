# All runs of the benchmark (bench.py), in order of priority. Toto runs in its own environment.
# Set $env:PY and $env:PY_TOTO to the Python interpreters of the two environments, for example
#   $env:PY = ".venv\Scripts\python.exe"; $env:PY_TOTO = ".venv-toto\Scripts\python.exe"; .\run_benchmark.ps1
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Set-Location $PSScriptRoot
$py = if ($env:PY) { $env:PY } else { "python" }
$pt = if ($env:PY_TOTO) { $env:PY_TOTO } else { "python" }
function B($exe, [string[]]$a) { & $exe bench.py @a }

B $py @('--data','heldout','--horizon','24','--models','naive','bolt','bolt_base','chronos2','timesfm','sundial','aurora')
B $py @('--data','aemo','--horizon','24','--models','naive','bolt','bolt_base','chronos2','timesfm','sundial','aurora')
B $pt @('--data','heldout','--horizon','24','--models','toto','--tag','toto')
B $pt @('--data','aemo','--horizon','24','--models','toto','--tag','toto')
foreach ($h in @('192', '720')) {
  B $py @('--data','heldout','--horizon',$h,'--n_test','150','--seeds','0','--models','naive','bolt','chronos2','timesfm','sundial','aurora')
  B $pt @('--data','heldout','--horizon',$h,'--n_test','150','--seeds','0','--models','toto','--tag','toto')
}
B $py @('--data','heldout','--horizon','24','--models','chronos')
B $py @('--data','aemo','--horizon','24','--seeds','0','--models','chronos')
B $py @('--data','heldout','--horizon','24','--seeds','0','--models','chronos_base','--tag','base')
B $py @('--data','heldout','--horizon','192','--n_test','150','--seeds','0','--models','chronos')
"benchmark done"
