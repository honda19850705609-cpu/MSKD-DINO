# 复制/同步项目前运行：删除 MSDA 编译目录，避免 Windows「目标路径太长」。
# 训练前在 models/dino/ops 下重新: pip install -e .
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$ops = Join-Path $root 'models\dino\ops'

function Remove-TreeRobocopy {
    param([string]$Target)
    if (-not (Test-Path -LiteralPath $Target)) { return }
    $full = (Resolve-Path -LiteralPath $Target).Path
    $empty = Join-Path $env:TEMP ("empty_robocopy_" + [guid]::NewGuid().ToString('n'))
    New-Item -ItemType Directory -Path $empty -Force | Out-Null
    try {
        & robocopy $empty $full /MIR /R:0 /W:0 /NFL /NDL /NJH /NJS | Out-Null
        Remove-Item -LiteralPath $empty -Force -Recurse -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $full -Force -Recurse -ErrorAction SilentlyContinue
    } finally {
        if (Test-Path -LiteralPath $empty) {
            Remove-Item -LiteralPath $empty -Force -Recurse -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path -LiteralPath $full) {
        # 兜底：长路径前缀
        $long = if ($full -match '^\\\\') { $full } else { "\\?\$full" }
        cmd /c "rmdir /s /q `"$long`"" 2>$null
    }
}

if (-not (Test-Path -LiteralPath $ops)) {
    Write-Host 'Skip: models\dino\ops not found'
    exit 0
}

foreach ($name in @('build', 'dist', '.eggs')) {
    $p = Join-Path $ops $name
    if (Test-Path -LiteralPath $p) {
        Write-Host "Removing $p ..."
        Remove-TreeRobocopy $p
    }
}

Get-ChildItem -Path $ops -Filter '*.egg-info' -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Removing $($_.FullName) ..."
    Remove-TreeRobocopy $_.FullName
}

Write-Host 'Done. Reinstall MSDA: cd models\dino\ops ; pip install -e .'
