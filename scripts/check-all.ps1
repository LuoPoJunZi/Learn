param(
    [string]$Root = '.'
)

$ErrorActionPreference = 'Stop'

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$failures = New-Object System.Collections.Generic.List[string]
$python = Get-Command python -ErrorAction SilentlyContinue

if ($null -eq $python) {
    $failures.Add('Python was not found; repository checks could not run.')
} else {
    & $python.Source (Join-Path $PSScriptRoot 'check-repository.py') --root $rootPath
    if ($LASTEXITCODE -ne 0) {
        $failures.Add('Repository structure checks failed.')
    }
}

$powerShellPath = (Get-Process -Id $PID).Path
$checkScripts = @(
    'check-code.ps1',
    'check-docs.ps1'
)

foreach ($checkScript in $checkScripts) {
    $scriptPath = Join-Path $PSScriptRoot $checkScript
    & $powerShellPath -NoProfile -ExecutionPolicy Bypass -File $scriptPath -Root $rootPath
    if ($LASTEXITCODE -ne 0) {
        $failures.Add($checkScript + ' failed.')
    }
}

$git = Get-Command git -ErrorAction SilentlyContinue
if ($null -eq $git) {
    $failures.Add('Git was not found; whitespace checks could not run.')
} else {
    & $git.Source -C $rootPath diff --check
    if ($LASTEXITCODE -ne 0) {
        $failures.Add('Unstaged changes contain whitespace errors.')
    }

    & $git.Source -C $rootPath diff --cached --check
    if ($LASTEXITCODE -ne 0) {
        $failures.Add('Staged changes contain whitespace errors.')
    }
}

if ($failures.Count -eq 0) {
    Write-Host 'All repository, code, documentation, and whitespace checks passed.'
    exit 0
}

Write-Host 'Combined checks failed:'
$failures | ForEach-Object { Write-Host ('  ' + $_) }
exit 1
