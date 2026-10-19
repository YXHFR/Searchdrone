# Checks the Windows side of the Searchdrone simulator setup. Changes nothing.
# Run in PowerShell (as administrator for the firewall check):
#   powershell -ExecutionPolicy Bypass -File tools\setup\windows-check.ps1
#   powershell -ExecutionPolicy Bypass -File tools\setup\windows-check.ps1 -Watch
#   powershell -ExecutionPolicy Bypass -File tools\setup\windows-check.ps1 -Watch `
#       -OutFile perf.csv -StopFile perf.stop -MaxSeconds 3600
# Prints OK / VARNING / FEL per line. -Watch logs available memory and GPU memory every
# 5 seconds to a CSV file (header utc,available_mb,gpu_used_mb; used by `just perf`).
# In -Watch mode the checks still print, but a failed check never stops the logging.
#   -WatchFile / -OutFile  the CSV file (appended to; the header is written once)
#   -StopFile              logging stops cleanly as soon as this file exists
#   -MaxSeconds            logging stops after this many seconds (0 = until Ctrl-C)
# Exit code: 1 if a check failed (FEL), else 0.
[CmdletBinding()]
param(
    [switch]$Watch,
    [Alias("OutFile")]
    [string]$WatchFile = "searchdrone-watch.csv",
    [string]$StopFile = "",
    [ValidateRange(0, [int]::MaxValue)]
    [int]$MaxSeconds = 0
)

$ErrorActionPreference = "Continue"
$failed = 0
# Swedish letters as character codes: the file stays ASCII, so Windows PowerShell 5.1
# (which reads a file without a byte order mark as the local code page) prints them right.

function Report([string]$status, [string]$text) {
    $color = @{ "OK" = "Green"; "VARNING" = "Yellow"; "FEL" = "Red" }[$status]
    Write-Host ("{0,-8} {1}" -f $status, $text) -ForegroundColor $color
    if ($status -eq "FEL") { $script:failed++ }
}

# 1. WSL 2 with Ubuntu 24.04: a distro named Ubuntu-24.04, or any WSL 2 distro whose
#    /etc/os-release says VERSION_ID="24.04" (for example one installed as "Ubuntu").
$wslList = (wsl --list --verbose 2>$null) -join "`n" -replace "`0", ""
$distros = @()
foreach ($line in ($wslList -split "`r?`n")) {
    $parts = ($line.Trim() -replace "^\*\s*", "") -split "\s+"
    if ($parts.Count -ge 3 -and $parts[-1] -match "^\d+$" -and $parts[0] -ne "NAME") {
        $distros += [pscustomobject]@{ Name = $parts[0]; Version = [int]$parts[-1] }
    }
}
$wsl2 = @($distros | Where-Object { $_.Version -eq 2 })
$found = $wsl2 | Where-Object { $_.Name -eq "Ubuntu-24.04" } | Select-Object -First 1
if (-not $found) {
    foreach ($d in $wsl2) {
        $release = (wsl -d $d.Name -- grep VERSION_ID /etc/os-release 2>$null) -join "`n" -replace "`0", ""
        if ($release -match 'VERSION_ID="?24\.04"?') { $found = $d; break }
    }
}
if ($found) {
    Report "OK" ("WSL: " + $found.Name + " (Ubuntu 24.04) med version 2")
} else {
    Report "FEL" "WSL: Ubuntu 24.04 (version 2) saknas. Kör: wsl --install -d Ubuntu-24.04"
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
    Report "FEL" ".wslconfig saknas i din Windows-hemkatalog. Kopiera tools/setup/wslconfig.example dit."
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

# Available memory in MB. Win32_OperatingSystem works on every Windows language; the
# performance counter name is translated on non-English Windows.
function Get-AvailableMb {
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        return [int][math]::Floor([double]$os.FreePhysicalMemory / 1024)
    } catch {
        return ""
    }
}

function Get-GpuUsedMb {
    if (-not $smi) { return "" }
    try {
        $out = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ("$out".Trim() -match "^\d+$") { return "$out".Trim() }
    } catch { }
    return ""
}

function Test-Stop {
    if ($StopFile -and (Test-Path -LiteralPath $StopFile)) { return $true }
    if ($MaxSeconds -gt 0 -and $clock.Elapsed.TotalSeconds -ge $MaxSeconds) { return $true }
    return $false
}

if ($Watch) {
    $until = if ($MaxSeconds -gt 0) { " i högst $MaxSeconds sekunder" } else { "" }
    $how = if ($StopFile) { "Avslutas när stoppfilen finns eller med Ctrl-C." } else { "Avsluta med Ctrl-C." }
    Write-Host "Loggar minne var 5:e sekund till $WatchFile$until. $how"
    if (-not (Test-Path -LiteralPath $WatchFile)) {
        "utc,available_mb,gpu_used_mb" | Out-File -FilePath $WatchFile -Encoding utf8
    }
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Stop)) {
        $row = "{0},{1},{2}" -f (Get-Date).ToUniversalTime().ToString("o"), (Get-AvailableMb), (Get-GpuUsedMb)
        $row | Out-File -FilePath $WatchFile -Append -Encoding utf8
        # Wait 5 s in short steps so a stop file is noticed within half a second.
        $next = $clock.Elapsed.TotalSeconds + 5
        while ($clock.Elapsed.TotalSeconds -lt $next -and -not (Test-Stop)) {
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Host "Loggningen är klar."
}

if ($failed -gt 0) { exit 1 } else { exit 0 }
