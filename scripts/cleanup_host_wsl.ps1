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
$rules = @("Ray Head", "Ray Dashboard", "Ray Workers", "SSH Port 22")

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

# 3. Clean up all WSL Distributions (Optional / Interactive)
Write-Host "`n[3/5] WSL Distributions Clean Up..." -ForegroundColor Yellow

$dists = @()
try {
    $previousOutputEncoding = [Console]::OutputEncoding
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    $rawDists = wsl --list --quiet 2>&1
    [Console]::OutputEncoding = $previousOutputEncoding
    
    foreach ($line in $rawDists) {
        $clean = $line.Trim()
        if ($clean -and $clean -notmatch "is not recognized" -and $clean -notmatch "bulunamadı" -and $clean -notmatch "command not found") {
            $dists += $clean
        }
    }
} catch {
    Write-Warning "Failed to retrieve WSL distribution list: $_"
}

if ($dists.Count -gt 0) {
    Write-Host "Found the following WSL distributions: $($dists -join ', ')" -ForegroundColor Cyan
    Write-Host "[WARNING] Unregistering a distribution will COMPLETELY DELETE its filesystem and all files inside it." -ForegroundColor Red
    
    $choice = Read-Host "Do you want to completely uninstall and delete ALL found WSL distributions? (Y/N)"
    if ($choice -eq "Y" -or $choice -eq "y") {
        foreach ($dist in $dists) {
            try {
                Write-Host "Unregistering distribution '$dist'..." -ForegroundColor Cyan
                wsl --unregister $dist | Out-Null
                Write-Host "Successfully unregistered and deleted '$dist'." -ForegroundColor Green
            } catch {
                Write-Error "Failed to unregister '$dist': $_"
            }
        }
    } else {
        Write-Host "Skipping WSL distribution deletion." -ForegroundColor Green
    }
} else {
    Write-Host "No installed WSL distributions found." -ForegroundColor DarkGray
}

# 4. Clean up temporary Windows host files (setup_worker_wsl.sh)
Write-Host "`n[4/5] Cleaning up temporary host files..." -ForegroundColor Yellow
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
            Write-Warning "Failed to delete temporary file ${filePath} - $_"
        }
    }
}

# 5. Completely Uninstall WSL and features (Optional / Interactive)
Write-Host "`n[5/5] Completely Uninstall WSL and Virtualization Features..." -ForegroundColor Yellow
$wslUninstallChoice = Read-Host "Do you want to completely disable WSL and Virtual Machine Platform features on Windows? (Y/N)"
if ($wslUninstallChoice -eq "Y" -or $wslUninstallChoice -eq "y") {
    Write-Host "Disabling Windows Subsystem for Linux feature..." -ForegroundColor Cyan
    try {
        Disable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart -ErrorAction Stop | Out-Null
        Write-Host "Successfully disabled Microsoft-Windows-Subsystem-Linux feature." -ForegroundColor Green
    } catch {
        Write-Warning "Failed to disable Microsoft-Windows-Subsystem-Linux: $_"
    }

    Write-Host "Disabling Virtual Machine Platform feature..." -ForegroundColor Cyan
    try {
        Disable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart -ErrorAction Stop | Out-Null
        Write-Host "Successfully disabled VirtualMachinePlatform feature." -ForegroundColor Green
    } catch {
        Write-Warning "Failed to disable VirtualMachinePlatform: $_"
    }

    # Also uninstall the Windows Subsystem for Linux Store package if installed
    Write-Host "Uninstalling Windows Subsystem for Linux App package..." -ForegroundColor Cyan
    try {
        Get-AppxPackage -Name "*WindowsSubsystemForLinux*" -AllUsers | Remove-AppxPackage -AllUsers -ErrorAction Stop | Out-Null
        Write-Host "Successfully uninstalled WSL App package." -ForegroundColor Green
    } catch {
        # Silent ignore if not found or cannot remove
    }

    Write-Host "`n[IMPORTANT] WSL and Virtualization features have been disabled. You must restart your PC to complete uninstallation." -ForegroundColor Red
} else {
    Write-Host "Skipping WSL features uninstallation." -ForegroundColor Green
}

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "   Cleanup and restoration completed!            " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Read-Host "Press Enter to exit..."
