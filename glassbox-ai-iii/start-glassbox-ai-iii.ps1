param(
  [ValidateRange(1, 65535)]
  [int]$Port = 8103,
  [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$appName = 'glassbox-ai-iii'
$applicationMarker = '<meta name="application-name" content="glassbox-ai-iii">'

function Test-ExpectedApplication {
  param([int]$CandidatePort)
  try {
    $response = Invoke-WebRequest `
      -Uri "http://127.0.0.1:$CandidatePort/index.html?app=$appName&path=auto" `
      -UseBasicParsing `
      -TimeoutSec 1
    return $response.Content.Contains($applicationMarker)
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

if (Test-ExpectedApplication -CandidatePort $Port) {
  $url = "http://127.0.0.1:$Port/index.html?app=$appName&path=auto"
  Write-Host "Glassbox AI III is already running: $url"
  if (-not $NoBrowser) { Start-Process $url }
  exit 0
}

$selectedPort = $Port
while (-not (Test-PortAvailable -CandidatePort $selectedPort)) {
  $selectedPort += 1
  if ($selectedPort -gt [Math]::Min($Port + 99, 65535)) {
    throw "No free port found from $Port to $([Math]::Min($Port + 99, 65535))."
  }
}

$server = Start-Process `
  -FilePath 'python' `
  -ArgumentList @('-m', 'http.server', $selectedPort, '--bind', '127.0.0.1', '--directory', $appRoot) `
  -WorkingDirectory $appRoot `
  -WindowStyle Hidden `
  -PassThru

$ready = $false
for ($attempt = 0; $attempt -lt 40; $attempt += 1) {
  Start-Sleep -Milliseconds 100
  if ($server.HasExited) { throw 'Glassbox AI III server exited before startup completed.' }
  if (Test-ExpectedApplication -CandidatePort $selectedPort) {
    $ready = $true
    break
  }
}

if (-not $ready) {
  Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
  throw 'Glassbox AI III server did not become ready.'
}

$url = "http://127.0.0.1:$selectedPort/index.html?app=$appName&path=auto"
Write-Host "Glassbox AI III: $url"
Write-Host "Server PID: $($server.Id)"
Write-Host "Stop with: Stop-Process -Id $($server.Id)"
if (-not $NoBrowser) { Start-Process $url }
