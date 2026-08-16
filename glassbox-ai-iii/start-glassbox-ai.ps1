Write-Warning 'start-glassbox-ai.ps1 is a compatibility alias. Use start-glassbox-ai-iii.ps1.'
& "$PSScriptRoot\start-glassbox-ai-iii.ps1" @args
exit $LASTEXITCODE
