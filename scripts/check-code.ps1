param(
    [string]$Root = '.'
)

$ErrorActionPreference = 'Stop'

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$failures = New-Object System.Collections.Generic.List[string]
$ignoredPathPattern = '[\\/]\.(git|agents|codex|venv)([\\/]|$)|[\\/]node_modules([\\/]|$)'

$python = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $python) {
    $failures.Add('Python was not found; Python checks could not run.')
} else {
    & $python.Source (Join-Path $PSScriptRoot 'check-python-code.py') --root $rootPath
    if ($LASTEXITCODE -ne 0) {
        $failures.Add('Python checks failed.')
    }

    & $python.Source (Join-Path $PSScriptRoot 'check-html-code.py') --root $rootPath
    if ($LASTEXITCODE -ne 0) {
        $failures.Add('HTML attribute checks failed.')
    }
}

$javascriptFiles = Get-ChildItem -LiteralPath $rootPath -Recurse -File -Filter '*.js' |
    Where-Object { $_.FullName -notmatch $ignoredPathPattern }
$node = Get-Command node -ErrorAction SilentlyContinue
if ($javascriptFiles.Count -gt 0 -and $null -eq $node) {
    $failures.Add('Node.js was not found; JavaScript checks could not run.')
} elseif ($null -ne $node) {
    foreach ($file in $javascriptFiles) {
        & $node.Source --check $file.FullName
        if ($LASTEXITCODE -ne 0) {
            $failures.Add('JavaScript syntax error: ' + $file.FullName)
        }
    }
}

$powershellFiles = Get-ChildItem -LiteralPath $rootPath -Recurse -File -Filter '*.ps1' |
    Where-Object { $_.FullName -notmatch $ignoredPathPattern }
foreach ($file in $powershellFiles) {
    $tokens = $null
    $parseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $file.FullName,
        [ref]$tokens,
        [ref]$parseErrors
    ) | Out-Null
    foreach ($parseError in $parseErrors) {
        $failures.Add(
            $file.FullName + ':' + $parseError.Extent.StartLineNumber + ': ' +
            $parseError.Message
        )
    }
}

$htmlFiles = Get-ChildItem -LiteralPath $rootPath -Recurse -File -Filter '*.html' |
    Where-Object { $_.FullName -notmatch $ignoredPathPattern }
foreach ($file in $htmlFiles) {
    $content = Get-Content -Raw -LiteralPath $file.FullName
    $requirements = @(
        @('doctype', '(?is)<!doctype\s+html>'),
        @('html lang', '(?is)<html[^>]*\blang='),
        @('charset', '(?is)<meta[^>]*charset='),
        @('viewport', '(?is)<meta[^>]*name=["'']viewport["'']'),
        @('title', '(?is)<title>\s*.+?\s*</title>')
    )
    foreach ($requirement in $requirements) {
        if ($content -notmatch $requirement[1]) {
            $failures.Add($file.FullName + ': missing ' + $requirement[0])
        }
    }
}

$matlabFiles = Get-ChildItem -LiteralPath $rootPath -Recurse -File -Filter '*.m' |
    Where-Object { $_.FullName -notmatch $ignoredPathPattern }
foreach ($file in $matlabFiles) {
    $firstCodeLine = Get-Content -LiteralPath $file.FullName |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace($_) -and
            -not $_.TrimStart().StartsWith('%')
        } |
        Select-Object -First 1

    $functionName = $null
    if ($firstCodeLine -match '^\s*function\s+(?:\[[^\]]+\]|\w+)\s*=\s*([A-Za-z]\w*)') {
        $functionName = $matches[1]
    } elseif ($firstCodeLine -match '^\s*function\s+([A-Za-z]\w*)') {
        $functionName = $matches[1]
    }

    if ($null -ne $functionName -and $functionName -cne $file.BaseName) {
        $failures.Add(
            $file.FullName + ': primary function ' + $functionName +
            ' must match the file name exactly'
        )
    }
}

if ($failures.Count -eq 0) {
    Write-Host (
        'Code checks passed: Python, JavaScript, PowerShell, HTML, and Matlab names.'
    )
    exit 0
}

Write-Host 'Code checks failed:'
$failures | ForEach-Object { Write-Host ('  ' + $_) }
exit 1
