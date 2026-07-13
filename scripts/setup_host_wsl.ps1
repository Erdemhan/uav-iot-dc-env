# setup_host_wsl.ps1
# Automates WSL2 installation, Mirrored Networking configuration, and Windows Firewall rules for Ray Cluster.
# Run this script on the Windows Host side as Administrator before starting WSL2.

# Ensure the script runs with Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "This script requires Administrator privileges. Relaunching as Admin..."
    Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    Exit
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Ray Cluster - Windows Host Setup for WSL2      " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Configure Windows Firewall Port Rules
Write-Host "`n[1/3] Configuring Windows Defender Firewall rules..." -ForegroundColor Yellow
$ports = @(
    @{ Name="Ray Head"; Port="6379" },
    @{ Name="Ray Dashboard"; Port="8265" },
    @{ Name="Ray Workers"; Port="10001-10100" }
)

foreach ($rule in $ports) {
    $ruleName = $rule.Name
    $localPort = $rule.Port
    
    # Check if rule already exists
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Firewall rule '$ruleName' already exists. Skipping." -ForegroundColor Green
    } else {
        try {
            New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $localPort -ErrorAction Stop | Out-Null
            Write-Host "Successfully created firewall rule '$ruleName' (Port: $localPort)." -ForegroundColor Green
        } catch {
            Write-Error "Failed to create firewall rule '$ruleName': $_"
        }
    }
}

# 2. Configure WSL2 Mirrored Networking & RAM Limits (.wslconfig)
Write-Host "`n[2/3] Configuring WSL2 Mirrored Networking and RAM allocation..." -ForegroundColor Yellow
$wslConfigPath = Join-Path $env:USERPROFILE ".wslconfig"

# Calculate safe maximum RAM for WSL2 (Total RAM - 3GB for Windows host stability)
$physicalMem = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum
$totalGb = [Math]::Floor($physicalMem / 1GB)
$wslGb = $totalGb - 3
if ($wslGb -lt 4) { $wslGb = 4 }

Write-Host "System RAM: ${totalGb}GB detected. Allocating ${wslGb}GB to WSL2 (saving 3GB for Windows stability)." -ForegroundColor Cyan

$configContent = @"
[wsl2]
networkingMode=mirrored
memory=${wslGb}GB
"@

$needsWrite = $true
if (Test-Path $wslConfigPath) {
    $currentContent = Get-Content $wslConfigPath -Raw
    if ($currentContent -match "networkingMode\s*=\s*mirrored" -and $currentContent -match "memory\s*=\s*${wslGb}GB") {
        Write-Host ".wslconfig is already configured with mirrored networking and ${wslGb}GB RAM." -ForegroundColor Green
        $needsWrite = $false
    } else {
        Write-Host "Updating .wslconfig with optimal networking and memory settings..." -ForegroundColor Cyan
        Set-Content -Path $wslConfigPath -Value $configContent -Force
    }
} else {
    Write-Host "Creating new .wslconfig with mirrored networking and ${wslGb}GB RAM..." -ForegroundColor Cyan
    Set-Content -Path $wslConfigPath -Value $configContent -Force
}

if ($needsWrite) {
    Write-Host "Successfully configured mirrored networking in: $wslConfigPath" -ForegroundColor Green
    # Terminate active WSL instances to force reloading config
    wsl --shutdown
    Write-Host "WSL instances shut down to apply new network settings." -ForegroundColor Cyan
}

# 3. Check and Install WSL2 (Ubuntu 24.04 LTS)
Write-Host "`n[3/3] Checking WSL2 and Ubuntu 24.04 LTS installation..." -ForegroundColor Yellow
$wslInstalled = $false
try {
    $null = Get-Command wsl -ErrorAction Stop
    $null = wsl --status 2>&1
    if ($LASTEXITCODE -eq 0) {
        $wslInstalled = $true
    }
} catch {
    $wslInstalled = $false
}

if (-not $wslInstalled) {
    Write-Host "WSL2 is not installed. Registering this script for automatic startup and initiating WSL2 installation..." -ForegroundColor Cyan
    
    # Register script to RunOnce registry key so it resumes after reboot
    try {
        $runOncePath = "HKLM:\Software\Microsoft\Windows\CurrentVersion\RunOnce"
        $runCmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
        Set-ItemProperty -Path $runOncePath -Name "RayWSLSetup" -Value $runCmd -ErrorAction Stop
        Write-Host "Registered script to RunOnce for automatic resumption after reboot." -ForegroundColor Green
    } catch {
        Write-Warning "Failed to register script in RunOnce registry: $_"
    }

    Write-Host "Installing WSL2 and Ubuntu 24.04 LTS..." -ForegroundColor Yellow
    wsl --install -d Ubuntu-24.04 --no-launch
    
    Write-Host "`n[IMPORTANT] WSL2 installation started. Restarting your PC in 5 seconds to complete setup..." -ForegroundColor Red
    Start-Sleep -Seconds 5
    Restart-Computer -Force
    Exit
} else {
    # Check if Ubuntu 24.04 LTS distribution is installed
    $distList = wsl --list --quiet 2>&1
    if ($distList -match "Ubuntu-24.04") {
        Write-Host "WSL2 and Ubuntu 24.04 LTS are already installed." -ForegroundColor Green
    } else {
        Write-Host "Ubuntu 24.04 LTS distribution not found. Installing Ubuntu 24.04 LTS..." -ForegroundColor Cyan
        wsl --install -d Ubuntu-24.04
        Write-Host "Ubuntu 24.04 LTS installation initiated. Complete the user registration in the Ubuntu window." -ForegroundColor Yellow
    }
}

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "   Windows Host configuration completed!          " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Read-Host "Press Enter to exit..."
