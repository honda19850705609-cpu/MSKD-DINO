# Training curves from log.txt (one JSON object per line) -> SVG files. No Python required.
param(
    [Parameter(Mandatory = $true)]
    [string]$ResultDir,
    [string]$LogName = "log.txt",
    [string]$OutSubdir = "training_curves"
)

$ErrorActionPreference = "Stop"
$logPath = Join-Path $ResultDir $LogName
if (-not (Test-Path -LiteralPath $logPath)) { throw "Missing: $logPath" }

$outDir = Join-Path $ResultDir $OutSubdir
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$rows = New-Object System.Collections.Generic.List[object]
Get-Content -LiteralPath $logPath -Encoding UTF8 | ForEach-Object {
    $t = $_.Trim()
    if (-not $t) { return }
    try { $rows.Add(($t | ConvertFrom-Json)) } catch { }
}
if ($rows.Count -eq 0) { throw "No valid JSON lines in log" }

function New-SvgPolylineY {
    param($Ys, [int]$W, [int]$H, [int]$Margin, [double]$Pad = 0.05)
    $valid = @()
    foreach ($y in $Ys) {
        if ($null -ne $y) { $valid += [double]$y }
    }
    if ($valid.Count -eq 0) { return @("", 0.0, 1.0) }
    $minY = ($valid | Measure-Object -Minimum).Minimum
    $maxY = ($valid | Measure-Object -Maximum).Maximum
    $span = [Math]::Max([double]$maxY - [double]$minY, 1e-9)
    $minY = [double]$minY - $span * $Pad
    $maxY = [double]$maxY + $span * $Pad
    $span = $maxY - $minY
    $iw = $W - 2 * $Margin
    $ih = $H - 2 * $Margin
    $n = $Ys.Count
    $pts = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $n; $i++) {
        $v = $Ys[$i]
        if ($null -eq $v) { continue }
        $fv = [double]$v
        $vx = $Margin + ($i / [Math]::Max($n - 1, 1)) * $iw
        $vy = $Margin + ($maxY - $fv) / $span * $ih
        $pts.Add(("{0:F1},{1:F1}" -f $vx, $vy))
    }
    return @(($pts -join " "), $minY, $maxY)
}

function Write-SvgFile {
    param([string]$Path, [string]$Inner, [string]$SvgTitle)
    $xml = "<?xml version=""1.0"" encoding=""UTF-8""?>`n"
    $xml += "<svg xmlns=""http://www.w3.org/2000/svg"" width=""920"" height=""460"" viewBox=""0 0 920 460"">`n"
    $xml += "  <rect width=""100%"" height=""100%"" fill=""#fafafa""/>`n"
    $xml += "  <text x=""460"" y=""28"" text-anchor=""middle"" font-size=""16"" font-family=""Segoe UI,Arial,sans-serif"" fill=""#222"">$SvgTitle</text>`n"
    $xml += $Inner
    $xml += "`n</svg>"
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $xml, $utf8NoBom)
}

$W = 900
$H = 420
$M = 55

$trainLoss = foreach ($r in $rows) { [double]$r.train_loss }
$testLoss = foreach ($r in $rows) {
    if ($r.PSObject.Properties.Name -contains "test_loss" -and $null -ne $r.test_loss) { [double]$r.test_loss } else { $null }
}
$lrArr = foreach ($r in $rows) { [double]$r.train_lr }
$map50 = foreach ($r in $rows) {
    if ($r.test_coco_eval_bbox -and $r.test_coco_eval_bbox.Count -ge 2) { [double]$r.test_coco_eval_bbox[1] } else { $null }
}
$map95 = foreach ($r in $rows) {
    if ($r.test_coco_eval_bbox -and $r.test_coco_eval_bbox.Count -ge 1) { [double]$r.test_coco_eval_bbox[0] } else { $null }
}

$plT, $null, $null = New-SvgPolylineY -Ys $trainLoss -W $W -H $H -Margin $M
$plV, $null, $null = New-SvgPolylineY -Ys $testLoss -W $W -H $H -Margin $M
$inner1 = @"
  <text x="$M" y="48" font-size="11" fill="#555" font-family="Segoe UI,Arial,sans-serif">Loss (train blue, val red)</text>
  <line x1="$M" y1="$($H-$M)" x2="$($W-$M)" y2="$($H-$M)" stroke="#ccc"/>
  <line x1="$M" y1="$M" x2="$M" y2="$($H-$M)" stroke="#ccc"/>
  <polyline fill="none" stroke="#1f77b4" stroke-width="2" points="$plT"/>
"@
if ($plV) { $inner1 += "`n  <polyline fill=`"none`" stroke=`"#d62728`" stroke-width=`"2`" points=`"$plV`"/>" }
Write-SvgFile -Path (Join-Path $outDir "curves_loss.svg") -Inner $inner1 -SvgTitle "train / val loss"

$plLR, $null, $null = New-SvgPolylineY -Ys $lrArr -W $W -H $H -Margin $M
$inner2 = @"
  <text x="$M" y="48" font-size="11" fill="#555" font-family="Segoe UI,Arial,sans-serif">Learning rate</text>
  <line x1="$M" y1="$($H-$M)" x2="$($W-$M)" y2="$($H-$M)" stroke="#ccc"/>
  <line x1="$M" y1="$M" x2="$M" y2="$($H-$M)" stroke="#ccc"/>
  <polyline fill="none" stroke="#2ca02c" stroke-width="2" points="$plLR"/>
"@
Write-SvgFile -Path (Join-Path $outDir "curves_lr.svg") -Inner $inner2 -SvgTitle "learning rate"

$pl50, $null, $null = New-SvgPolylineY -Ys $map50 -W $W -H $H -Margin $M
$pl95, $null, $null = New-SvgPolylineY -Ys $map95 -W $W -H $H -Margin $M
$inner3 = @"
  <text x="$M" y="48" font-size="11" fill="#555" font-family="Segoe UI,Arial,sans-serif">COCO bbox AP (magenta mAP@0.5, cyan mAP@[.5:.95])</text>
  <line x1="$M" y1="$($H-$M)" x2="$($W-$M)" y2="$($H-$M)" stroke="#ccc"/>
  <line x1="$M" y1="$M" x2="$M" y2="$($H-$M)" stroke="#ccc"/>
"@
if ($pl50) { $inner3 += "`n  <polyline fill=`"none`" stroke=`"#9467bd`" stroke-width=`"2`" points=`"$pl50`"/>" }
if ($pl95) { $inner3 += "`n  <polyline fill=`"none`" stroke=`"#17becf`" stroke-width=`"2`" points=`"$pl95`"/>" }
Write-SvgFile -Path (Join-Path $outDir "curves_map.svg") -Inner $inner3 -SvgTitle "validation mAP"

Write-Output $outDir
