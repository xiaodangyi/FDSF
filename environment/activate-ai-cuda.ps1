$workspaceRoot = Split-Path -Parent $PSScriptRoot
$taskConfigPath = Join-Path $workspaceRoot '.local\config.json'
if (-not (Test-Path -LiteralPath $taskConfigPath -PathType Leaf)) {
    throw '缺少 .local/config.json，请参考 configs/local.example.json 填写。'
}
$taskConfig = Get-Content -LiteralPath $taskConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$aiCudaRoot = $taskConfig.ai_cuda_root
if ([string]::IsNullOrWhiteSpace($aiCudaRoot)) {
    throw '本机配置缺少 ai_cuda_root 路径。'
}
if (-not [System.IO.Path]::IsPathRooted($aiCudaRoot)) {
    $aiCudaRoot = Join-Path $workspaceRoot $aiCudaRoot
}
$aiCudaRoot = [System.IO.Path]::GetFullPath($aiCudaRoot)
if (-not (Test-Path -LiteralPath (Join-Path $aiCudaRoot 'python.exe'))) {
    throw '未找到 ai-cuda 的 Python，请检查本机配置。'
}
$env:CONDA_PREFIX = $aiCudaRoot
$env:CONDA_DEFAULT_ENV = 'ai-cuda'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$taskEnvironmentPaths = @(
    $aiCudaRoot
    (Join-Path $aiCudaRoot 'Library\bin')
    (Join-Path $aiCudaRoot 'Scripts')
)
# 重复加载脚本时不反复追加同一组路径，只影响当前终端。
$env:PATH = (@($taskEnvironmentPaths) + @($env:PATH -split ';' | Where-Object { $_ -and $_ -notin $taskEnvironmentPaths })) -join ';'
$env:YOLO_CONFIG_DIR = Join-Path $workspaceRoot 'environment\ultralytics-settings'
$env:MPLCONFIGDIR = Join-Path $workspaceRoot 'environment\matplotlib-cache'
New-Item -ItemType Directory -Path $env:YOLO_CONFIG_DIR,$env:MPLCONFIGDIR -Force | Out-Null
Write-Host '当前 PowerShell 已启用 ai-cuda；本次操作没有修改系统 PATH。'
& (Join-Path $aiCudaRoot 'python.exe') --version
