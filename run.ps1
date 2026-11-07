$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache\uv'

# Use a system Node installation, or the project-local portable installation.
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    $nodeDirectory = Get-ChildItem -LiteralPath '.tools' -Directory -Filter 'node-v22.*-win-x64' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $nodeDirectory) {
        & (Join-Path $PSScriptRoot 'scripts\setup-node.ps1')
        $nodeDirectory = Get-ChildItem -LiteralPath '.tools' -Directory -Filter 'node-v22.*-win-x64' | Select-Object -First 1
    }
    $env:PATH = "$($nodeDirectory.FullName);$env:PATH"
}

uv sync --locked
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
Push-Location -LiteralPath 'frontend'
try {
    npm.cmd ci --cache '..\.cache\npm' --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally {
    Pop-Location
}
Write-Output 'Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.'
uv run --locked python main.py
