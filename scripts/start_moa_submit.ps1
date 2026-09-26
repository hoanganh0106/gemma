$ErrorActionPreference = 'Stop'
try {
    $menuPath = Join-Path $PSScriptRoot 'moa_submit_menu.py'
    $linuxPath = & wsl.exe --exec wslpath -a $menuPath
    if ($LASTEXITCODE -ne 0) { throw 'Khong mo duoc WSL. Hay kiem tra WSL da cai va khoi dong duoc.' }
    & wsl.exe --exec python3 -u ($linuxPath.Trim())
    exit $LASTEXITCODE
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
