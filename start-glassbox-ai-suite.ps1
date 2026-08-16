param(
  [int]$Port = 8000,
  [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$suiteRoot = $PSScriptRoot

function Test-GlassboxSuiteServer {
  param([int]$CandidatePort)
  try {
    $response = Invoke-WebRequest `
      -Uri "http://127.0.0.1:$CandidatePort/index.html?app=glassbox-ai-suite" `
      -UseBasicParsing `
      -TimeoutSec 1
    return $response.Content -match '<meta name="application-name" content="glassbox-ai-suite">'
  } catch {
    return $false
  }
}

function Test-PortAvailable {
  param([int]$CandidatePort)
  $listener = $null
  try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $CandidatePort)
    $listener.Start()
    return $true
  } catch {
    return $false
  } finally {
    if ($listener) { $listener.Stop() }
  }
}

if (Test-GlassboxSuiteServer -CandidatePort $Port) {
  $url = "http://127.0.0.1:$Port/index.html?app=glassbox-ai-suite"
  Write-Host "Glassbox AI Suite is already running: $url"
  if (-not $NoBrowser) { Start-Process $url }
  exit 0
}

$selectedPort = $Port
while (-not (Test-PortAvailable -CandidatePort $selectedPort)) {
  $selectedPort += 1
  if ($selectedPort -gt ($Port + 99)) {
    throw "No free port found from $Port to $($Port + 99)."
  }
}

$server = Start-Process `
  -FilePath 'python' `
  -ArgumentList @('-m', 'http.server', $selectedPort, '--bind', '127.0.0.1', '--directory', $suiteRoot) `
  -WorkingDirectory $suiteRoot `
  -WindowStyle Hidden `
  -PassThru

$ready = $false
for ($attempt = 0; $attempt -lt 40; $attempt += 1) {
  Start-Sleep -Milliseconds 100
  if ($server.HasExited) { throw 'Python HTTP server exited before startup completed.' }
  if (Test-GlassboxSuiteServer -CandidatePort $selectedPort) {
    $ready = $true
    break
  }
}

if (-not $ready) {
  Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
  throw 'Glassbox AI Suite server did not become ready.'
}

$url = "http://127.0.0.1:$selectedPort/index.html?app=glassbox-ai-suite"
Write-Host "Glassbox AI Suite: $url"
Write-Host "Server PID: $($server.Id)"
Write-Host "Stop with: Stop-Process -Id $($server.Id)"
if (-not $NoBrowser) { Start-Process $url }
