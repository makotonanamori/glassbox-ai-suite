param(
  [ValidateRange(1, 65535)]
  [int]$Port = 8000,
  [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$applicationMarker = '<meta name="application-name" content="glassbox-ai">'

function Get-PortState {
  param([int]$CandidatePort)

  try {
    $response = Invoke-WebRequest `
      -Uri "http://127.0.0.1:$CandidatePort/index.html?app=glassbox-ai" `
      -UseBasicParsing `
      -TimeoutSec 1
    if ($response.Content.Contains($applicationMarker)) {
      return 'glassbox-ai'
    }
    return 'occupied-by-another-app'
  } catch {
    $listener = Get-NetTCPConnection -LocalPort $CandidatePort -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
      return 'occupied-without-http-response'
    }
    return 'free'
  }
}

$selectedPort = $Port
$portState = Get-PortState -CandidatePort $selectedPort

if ($portState -eq 'glassbox-ai') {
  $url = "http://127.0.0.1:$selectedPort/index.html?app=glassbox-ai"
  Write-Host "Glassbox AI is already running: $url"
  if (-not $NoBrowser) {
    Start-Process $url
  }
  exit 0
}

if ($portState -ne 'free') {
  $candidatePorts = if ($Port -lt 65535) {
    ($Port + 1)..([Math]::Min($Port + 10, 65535))
  } else {
    @()
  }

  $existingPort = $candidatePorts |
    Where-Object { (Get-PortState -CandidatePort $_) -eq 'glassbox-ai' } |
    Select-Object -First 1

  if ($existingPort) {
    $url = "http://127.0.0.1:$existingPort/index.html?app=glassbox-ai"
    Write-Host "Glassbox AI is already running: $url"
    if (-not $NoBrowser) {
      Start-Process $url
    }
    exit 0
  }

  $fallbackPort = $candidatePorts |
    Where-Object { (Get-PortState -CandidatePort $_) -eq 'free' } |
    Select-Object -First 1

  if (-not $fallbackPort) {
    throw "No free port was found between $Port and $($Port + 10)."
  }

  Write-Warning "Port $Port is used by another app. Glassbox AI will start on port $fallbackPort."
  $selectedPort = $fallbackPort
}

$serverArguments = @(
  '-m',
  'http.server',
  "$selectedPort",
  '--bind',
  '127.0.0.1',
  '--directory',
  "`"$projectDirectory`""
)

$serverProcess = Start-Process `
  -FilePath 'python' `
  -ArgumentList $serverArguments `
  -WindowStyle Hidden `
  -PassThru

$url = "http://127.0.0.1:$selectedPort/index.html?app=glassbox-ai"
$ready = $false
for ($attempt = 0; $attempt -lt 50; $attempt += 1) {
  Start-Sleep -Milliseconds 100
  if ($serverProcess.HasExited) {
    throw "The Glassbox AI local server exited during startup."
  }
  if ((Get-PortState -CandidatePort $selectedPort) -eq 'glassbox-ai') {
    $ready = $true
    break
  }
}

if (-not $ready) {
  Stop-Process -Id $serverProcess.Id -ErrorAction SilentlyContinue
  throw "Glassbox AI did not become ready within five seconds."
}

Write-Host "Glassbox AI started: $url"
Write-Host "Server PID: $($serverProcess.Id)"
if (-not $NoBrowser) {
  Start-Process $url
}
