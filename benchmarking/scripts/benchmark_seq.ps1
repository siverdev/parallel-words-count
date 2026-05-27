param (
    [string]$exe,
    [int]$runs = 3,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$rest
)

if (-not (Test-Path $exe)) {
    Write-Error "Executable not found: $exe"
    exit 1
}

$outDir = Join-Path -Path $PSScriptRoot -ChildPath "..\output"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
$outputFile = Join-Path -Path $outDir -ChildPath "seq_benchmark.csv"

"Mode,UniqueWords,AvgTimeMs" | Out-File $outputFile

Write-Host "`n================================="
Write-Host "SEQUENTIAL BENCHMARK (Baseline)"
Write-Host "=================================`n"

$totalTime = 0
$wordsCount = 0

for ($r = 1; $r -le $runs; $r++) {
    $output = & $exe @rest

    $timeMatch = $output | Select-String -Pattern "Execution time:\s+(\d+)\s+ms"
    $wordsMatch = $output | Select-String -Pattern "Total unique words:\s+(\d+)"

    if ($timeMatch -and $wordsMatch) {
        $timeMs = [int]$timeMatch.Matches.Groups[1].Value
        $wordsCount = [int]$wordsMatch.Matches.Groups[1].Value
        $totalTime += $timeMs
        
        Write-Host "  Run $r -> $timeMs ms"
    } else {
        Write-Warning "  Run $r -> Failed to parse output."
    }
}

if ($runs -gt 0) {
    $avgTime = [math]::Round($totalTime / $runs, 2)
    $line = "SEQ,$wordsCount,$avgTime"
    Add-Content $outputFile $line
    
    Write-Host "`nResult: $wordsCount words | Average Time: $avgTime ms"
}

Write-Host "`nSaved benchmark to: $outputFile"