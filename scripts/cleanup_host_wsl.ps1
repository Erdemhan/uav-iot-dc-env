# cleanup_host_wsl.ps1
# Automates the cleanup process: removes firewall rules, cleans .wslconfig, and unregisters WSL Ubuntu.
# Run this script on the Windows Host side as Administrator when you want to restore the PCs to their original state.

# Ensure the script runs with Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "This script requires Administrator privileges. Relaunching as Admin..."
    Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    Exit
}

Write-Host "==================================================" -ForegroundColor Yellow
Write-Host "   Ray Cluster - WSL2 Cleanup & Restore Script    " -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Yellow

# 1. Remove Windows Firewall Rules
Write-Host "`n[1/3] Removing Windows Defender Firewall rules..." -ForegroundColor Yellow
$rules = @("Ray Head", "Ray Dashboard", "Ray Workers")

foreach ($ruleName in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) {
        try {
            Remove-NetFirewallRule -DisplayName $ruleName -ErrorAction Stop
            Write-Host "Successfully removed firewall rule '$ruleName'." -ForegroundColor Green
        } catch {
            Write-Error "Failed to remove firewall rule '$ruleName': $_"
        }
    } else {
        Write-Host "Firewall rule '$ruleName' does not exist. Skipping." -ForegroundColor DarkGray
    }
}

# 2. Revert .wslconfig Mirrored Networking
Write-Host "`n[2/3] Cleaning up WSL2 configuration (.wslconfig)..." -ForegroundColor Yellow
$wslConfigPath = Join-Path $env:USERPROFILE ".wslconfig"

if (Test-Path $wslConfigPath) {
    $currentContent = Get-Content $wslConfigPath -Raw
    
    # Check if config has other settings beside wsl2 mirrored
    # If it only has wsl2/networkingMode, we can delete the file safely.
    $cleanedContent = $currentContent -replace "(?ms)\[wsl2\].*?networkingMode\s*=\s*mirrored\r?\n?", ""
    $cleanedContent = $cleanedContent.Trim()

    if ([string]::IsNullOrWhiteSpace($cleanedContent)) {
        try {
            Remove-Item $wslConfigPath -Force -ErrorAction Stop
            Write-Host "Successfully deleted .wslconfig as it had no other settings." -ForegroundColor Green
        } catch {
            Write-Error "Failed to delete .wslconfig: $_"
        }
    } else {
        try {
            Set-Content -Path $wslConfigPath -Value $cleanedContent -Force -ErrorAction Stop
            Write-Host "Successfully removed mirrored networking setting from .wslconfig." -ForegroundColor Green
        } catch {
            Write-Error "Failed to update .wslconfig: $_"
        }
    }
    
    # Force shutdown of any active WSL instances
    wsl --shutdown
} else {
    Write-Host ".wslconfig file not found. Skipping." -ForegroundColor DarkGray
}

# 3. Clean up WSL Ubuntu Distribution (Optional / Interactive)
Write-Host "`n[3/4] WSL Ubuntu Distribution Clean Up..." -ForegroundColor Yellow
$distList = wsl --list --quiet 2>&1
if ($distList -match "Ubuntu") {
    Write-Host "[WARNING] Unregistering Ubuntu will COMPLETELY DELETE the Ubuntu filesystem" -ForegroundColor Red
    Write-Host "and all files, environments, packages installed inside it." -ForegroundColor Red
    $choice = Read-Host "Do you want to completely uninstall and delete Ubuntu? (Y/N)"
    
    if ($choice -eq "Y" -or $choice -eq "y") {
        Write-Host "Unregistering Ubuntu distribution..." -ForegroundColor Cyan
        wsl --unregister Ubuntu
        Write-Host "Ubuntu distribution has been successfully unregistered and deleted." -ForegroundColor Green
    } else {
        Write-Host "Skipping Ubuntu deletion as requested." -ForegroundColor Green
    }
} else {
    Write-Host "Ubuntu distribution not found. Skipping." -ForegroundColor DarkGray
}

# 4. Clean up temporary Windows host files (setup_worker_wsl.sh)
Write-Host "`n[4/4] Cleaning up temporary host files..." -ForegroundColor Yellow
$tempPaths = @(
    "C:\setup_worker_wsl.sh",
    (Join-Path $env:USERPROFILE "Desktop\setup_worker_wsl.sh"),
    (Join-Path $PSScriptRoot "setup_worker_wsl.sh")
)

foreach ($filePath in $tempPaths) {
    if (Test-Path $filePath) {
        try {
            Remove-Item $filePath -Force -ErrorAction Stop
            Write-Host "Successfully deleted temporary host file: $filePath" -ForegroundColor Green
        } catch {
            Write-Warning "Failed to delete temporary file $filePath: $_"
        }
    }
}

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "   Cleanup and restoration completed!            " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Read-Host "Press Enter to exit..."
