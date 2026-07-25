$files = @(
  "src\components\VideoUpload.tsx",
  "src\components\VideosList.tsx",
  "src\components\EventTimeline.tsx",
  "src\components\ChatPanel.tsx",
  "src\components\ReportViewer.tsx",
  "src\components\Layout.tsx"
)

foreach ($file in $files) {
  $content = Get-Content $file -Raw
  $content = $content -replace "brand-", "indigo-"
  Set-Content $file $content -NoNewline
  Write-Host "Fixed: $file"
}
