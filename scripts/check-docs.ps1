param(
    [string]$Root = '.'
)

$ErrorActionPreference = 'Stop'

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$markdownFiles = Get-ChildItem -LiteralPath $rootPath -Recurse -File -Filter '*.md' |
    Where-Object { $_.FullName -notmatch '\\.git\\' }

$brokenLinks = New-Object System.Collections.Generic.List[string]
$unclosedFences = New-Object System.Collections.Generic.List[string]
$missingReadmes = New-Object System.Collections.Generic.List[string]

foreach ($file in $markdownFiles) {
    $content = Get-Content -LiteralPath $file.FullName
    $inFence = $false
    $fenceStart = 0
    $fenceLength = 0

    for ($i = 0; $i -lt $content.Count; $i++) {
        $line = $content[$i]

        if ($line -match '^\s*(`{3,})') {
            $currentFenceLength = $matches[1].Length
            if (-not $inFence) {
                $inFence = $true
                $fenceStart = $i + 1
                $fenceLength = $currentFenceLength
            } else {
                if ($currentFenceLength -ge $fenceLength) {
                    $inFence = $false
                    $fenceLength = 0
                }
            }
            continue
        }

        if ($inFence) {
            continue
        }

        $scanLine = [regex]::Replace($line, '`+[^`]*`+', '')
        foreach ($match in ([regex]::Matches($scanLine, '\[[^\]]+\]\(([^)]+)\)'))) {
            $target = $match.Groups[1].Value
            $openAngle = [string][char]60
            $closeAngle = [string][char]62
            if ($target.StartsWith($openAngle) -and $target.EndsWith($closeAngle)) {
                $target = $target.Substring(1, $target.Length - 2)
            }

            if ($target -match '^(https?:|mailto:|#)') {
                continue
            }

            $hashIndex = $target.IndexOf([string][char]35)
            if ($hashIndex -ge 0) {
                $targetPath = $target.Substring(0, $hashIndex)
            } else {
                $targetPath = $target
            }

            $targetPath = [uri]::UnescapeDataString($targetPath)
            if ([string]::IsNullOrWhiteSpace($targetPath)) {
                continue
            }

            $resolved = Join-Path $file.DirectoryName $targetPath
            if (-not (Test-Path -LiteralPath $resolved)) {
                $message = $file.FullName + ':' + ($i + 1).ToString() + ' link ' + $target
                $brokenLinks.Add($message)
            }
        }
    }

    if ($inFence) {
        $message = $file.FullName + ': fence opened at line ' + $fenceStart.ToString()
        $unclosedFences.Add($message)
    }
}

$topDirs = Get-ChildItem -LiteralPath $rootPath -Directory |
    Where-Object { $_.Name -notin @('.git', '.agents', '.codex') }

foreach ($dir in $topDirs) {
    $readme = Join-Path $dir.FullName 'README.md'
    if (-not (Test-Path -LiteralPath $readme)) {
        $missingReadmes.Add($dir.FullName)
    }
}

if ($brokenLinks.Count -eq 0 -and $unclosedFences.Count -eq 0 -and $missingReadmes.Count -eq 0) {
    Write-Host 'Documentation checks passed.'
    exit 0
}

if ($brokenLinks.Count -gt 0) {
    Write-Host 'Broken markdown links:'
    $brokenLinks | ForEach-Object { Write-Host ('  ' + $_) }
}

if ($unclosedFences.Count -gt 0) {
    Write-Host 'Unclosed code fences:'
    $unclosedFences | ForEach-Object { Write-Host ('  ' + $_) }
}

if ($missingReadmes.Count -gt 0) {
    Write-Host 'Top-level directories without README.md:'
    $missingReadmes | ForEach-Object { Write-Host ('  ' + $_) }
}

exit 1
