$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolsDirectory = Join-Path $projectRoot '.tools'
New-Item -ItemType Directory -Force -Path $toolsDirectory | Out-Null

# Download the official portable Node 22 release and verify its SHA-256 hash.
$releaseUrl = 'https://nodejs.org/dist/latest-v22.x'
$checksums = (Invoke-WebRequest -Uri "$releaseUrl/SHASUMS256.txt" -UseBasicParsing).Content
$match = [regex]::Match($checksums, '(?m)^([a-f0-9]{64})\s+(node-v22\.[0-9]+\.[0-9]+-win-x64\.zip)\r?$')
if (-not $match.Success) { throw 'Could not identify the official Node 22 Windows download.' }
$fileName = $match.Groups[2].Value
$archivePath = Join-Path $toolsDirectory $fileName
Invoke-WebRequest -Uri "$releaseUrl/$fileName" -OutFile $archivePath -UseBasicParsing
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLower() -ne $match.Groups[1].Value) {
    throw 'Node download checksum did not match. The archive was not extracted.'
}
Expand-Archive -LiteralPath $archivePath -DestinationPath $toolsDirectory -Force
Write-Output "Portable Node is ready in $(Join-Path $toolsDirectory $fileName.Replace('.zip', ''))"
