param(
  [string]$EnvName = "wellness-copilot-rag"
)

$ErrorActionPreference = "SilentlyContinue"
Write-Host "[1/3] Removing conda environment (if exists): $EnvName"
if (Get-Command conda) {
  $envList = conda env list | Out-String
  if ($envList -match $EnvName) {
    conda env remove -n $EnvName -y
    Write-Host "Removed conda env: $EnvName"
  } else {
    Write-Host "Conda env not found: $EnvName"
  }
} else {
  Write-Host "Conda command not found, skip env removal."
}

Write-Host "[2/3] Cleaning project artifacts and caches"
$targets = @(
  "reports",
  "profile_store.json",
  "observability.db",
  "checkpoints.db",
  ".pytest_cache",
  ".mypy_cache",
  ".ruff_cache",
  "__pycache__"
)

foreach ($t in $targets) {
  if (Test-Path $t) {
    Remove-Item $t -Recurse -Force
    Write-Host "Removed: $t"
  }
}

Get-ChildItem -Path "knowledge_base" -Directory -Filter ".index_cache" -Recurse | ForEach-Object {
  Remove-Item $_.FullName -Recurse -Force
  Write-Host "Removed: $($_.FullName)"
}

Write-Host "[3/3] Done. Project is ready for a fresh rerun."
