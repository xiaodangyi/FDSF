param([switch]$VisibleChild, [switch]$Full)
$ErrorActionPreference = 'Stop'

if (-not $VisibleChild) {
    $hostExecutable = (Get-Process -Id $PID).Path
    $quotedScript = '"' + $PSCommandPath + '"'
    $childArguments = @(
        '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', $quotedScript, '-VisibleChild'
    )
    if ($Full) { $childArguments += '-Full' }
    Start-Process -FilePath $hostExecutable -WindowStyle Normal -ArgumentList $childArguments
    Write-Host '已打开独立训练 PowerShell 窗口；新窗口会显示条件检查和实时训练进度。'
    return
}

$workspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspaceRoot
. (Join-Path $PSScriptRoot 'activate-ai-cuda.ps1')
$env:DESKVISION_VISIBLE_TRAINING = '1'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $OutputEncoding
$terminalLogFolder = Join-Path $workspaceRoot 'runs\training-terminal'
New-Item -ItemType Directory -Path $terminalLogFolder -Force | Out-Null
$terminalLogPath = Join-Path $terminalLogFolder ((Get-Date -Format 'yyyyMMdd-HHmmss') + '.txt')
Start-Transcript -Path $terminalLogPath
if ($Full) {
    Write-Host '即将执行：python -u -m deskvision.train --config configs/training-full.yaml --run --evaluate-after-training'
    python -u -m deskvision.train --config configs/training-full.yaml --run --evaluate-after-training
} else {
    Write-Host '即将执行：python -m deskvision.train --run'
    python -u -m deskvision.train --run
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "训练进程退出代码：$LASTEXITCODE" -ForegroundColor Red
}
Stop-Transcript
