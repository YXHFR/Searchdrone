# Checks the Windows side of the Searchdrone simulator setup. Changes nothing.
# Run in PowerShell (as administrator for the firewall check):
#   powershell -ExecutionPolicy Bypass -File tools\setup\windows-check.ps1
#   powershell -ExecutionPolicy Bypass -File tools\setup\windows-check.ps1 -Watch
# Prints OK / VARNING / FEL per line. -Watch logs free memory and GPU memory every
# 5 seconds to a CSV file (used for performance measurements).
[CmdletBinding()]
param(
    [switch]$Watch,
    [string]$WatchFile = "searchdrone-watch.csv"
)

$ErrorActionPreference = "Continue"
$failed = 0

function Report([string]$status, [string]$text) {
    $color = @{ "OK" = "Green"; "VARNING" = "Yellow"; "FEL" = "Red" }[$status]
    Write-Host ("{0,-8} {1}" -f $status, $text) -ForegroundColor $color
    if ($status -eq "FEL") { $script:failed++ }
}

# 1. WSL with Ubuntu 24.04, version 2
$wslList = (wsl --list --verbose 2>$null) -join "`n" -replace "`0", ""
if ($wslList -match "Ubuntu-24\.04\s+\S+\s+2") {
    Report "OK" "WSL: Ubuntu-24.04 med version 2"
} else {
    Report "FEL" "WSL: Ubuntu-24.04 (version 2) saknas. Kör: wsl --install -d Ubuntu-24.04"
}

# 2. .wslconfig has the lines from tools/setup/wslconfig.example
$wslconfig = Join-Path $env:USERPROFILE ".wslconfig"
$required = @("memory=10GB", "processors=12", "swap=8GB", "networkingMode=mirrored", "autoMemoryReclaim=gradual")
if (Test-Path $wslconfig) {
    $content = Get-Content $wslconfig -Raw
    $missing = $required | Where-Object { $content -notmatch [regex]::Escape($_) }
    if ($missing) {
        Report "FEL" (".wslconfig saknar: " + ($missing -join ", ") + ". Se tools/setup/wslconfig.example.")
    } else {
        Report "OK" ".wslconfig har alla rader"
    }
} else {
    Report "FEL" ".wslconfig saknas i $env:USERPROFILE. Kopiera tools/setup/wslconfig.example dit."
}

# 3. NVIDIA graphics card with at least 6 GB memory (warning only)
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$gpuOk = $false
if ($smi) {
    $lines = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    foreach ($line in $lines) {
        $parts = $line -split ","
        if ($parts.Count -ge 2 -and [int]$parts[1].Trim() -ge 6000) {
            Report "OK" ("Grafikkort: " + $parts[0].Trim() + ", " + $parts[1].Trim() + " MB")
            $gpuOk = $true
        }
    }
}
if (-not $gpuOk) {
    Report "VARNING" "Inget NVIDIA-kort med 6 GB. Använd reservvägarna i installationsguiden (gui=false, just sim-headless)."
}

# 4. The WSL firewall blocks inbound traffic (Windows default)
try {
    $fw = Get-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -ErrorAction Stop
    if ("$($fw.DefaultInboundAction)" -eq "Block") {
        Report "OK" "Brandväggen för WSL blockerar inkommande trafik"
    } else {
        Report "FEL" "Brandväggen för WSL släpper in trafik utifrån. Simulatorns styrportar kan då nås från nätverket."
    }
} catch {
    Report "VARNING" "Kunde inte läsa brandväggen för WSL. Kör PowerShell som administratör och försök igen."
}

if ($Watch) {
    Write-Host "Loggar minne var 5:e sekund till $WatchFile. Avsluta med Ctrl-C."
    if (-not (Test-Path $WatchFile)) { "utc,available_mb,gpu_used_mb" | Out-File -FilePath $WatchFile -Encoding utf8 }
    while ($true) {
        $avail = [int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
        $gpu = if ($smi) { (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim() } else { "" }
        "{0},{1},{2}" -f (Get-Date).ToUniversalTime().ToString("o"), $avail, $gpu | Out-File -FilePath $WatchFile -Append -Encoding utf8
        Start-Sleep -Seconds 5
    }
}

if ($failed -gt 0) { exit 1 } else { exit 0 }
