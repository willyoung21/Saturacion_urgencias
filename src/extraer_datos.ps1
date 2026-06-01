
<#
.SYNOPSIS
    Extrae los 3 CSVs de entrenamiento desde el modelo semántico de Fabric.
    Usa el módulo oficial MicrosoftPowerBIMgmt — maneja el tenant automáticamente.

.USO
    .\src\extraer_datos.ps1
    .\src\extraer_datos.ps1 -Forzar   # re-descarga aunque ya existan los CSVs
#>

param(
    [switch]$Forzar
)

$ErrorActionPreference = "Stop"
$NIT       = "805023423CL"
$Workspace = "[3] CMI [PROD]"
$Dataset   = "Ospedale Mensual [Dataset]"
$DataDir   = Join-Path $PSScriptRoot "..\data"

# ──────────────────────────────────────────────────────────────────────────────
# 1. Instalar módulo si no existe
# ──────────────────────────────────────────────────────────────────────────────
if (-not (Get-Module -ListAvailable -Name MicrosoftPowerBIMgmt)) {
    Write-Host "Instalando MicrosoftPowerBIMgmt..." -ForegroundColor Cyan
    Install-Module -Name MicrosoftPowerBIMgmt -Scope CurrentUser -Force -AllowClobber
}
Import-Module MicrosoftPowerBIMgmt -ErrorAction Stop

# ──────────────────────────────────────────────────────────────────────────────
# 2. Login interactivo (abre el navegador del sistema)
# ──────────────────────────────────────────────────────────────────────────────
Write-Host "`n=== Login Microsoft - usa la cuenta con acceso a Fabric ==="
Write-Host "Se abrira el navegador para iniciar sesion..." -ForegroundColor Yellow
$loginResult = Connect-PowerBIServiceAccount
if (-not $loginResult) { throw "Login fallido. Intenta de nuevo." }
Write-Host "Login OK: $($loginResult.UserName)`n" -ForegroundColor Green

# ──────────────────────────────────────────────────────────────────────────────
# 3. Resolver workspace ID y dataset ID
# ──────────────────────────────────────────────────────────────────────────────
Write-Host "Buscando workspace '$Workspace'..." -NoNewline
$groups = Invoke-PowerBIRestMethod -Url "groups" -Method Get | ConvertFrom-Json
$group  = $groups.value | Where-Object { $_.name -eq $Workspace }
if (-not $group) { throw "Workspace '$Workspace' no encontrado." }
$groupId = $group.id
Write-Host " OK ($groupId)" -ForegroundColor Green

Write-Host "Buscando dataset '$Dataset'..." -NoNewline
$datasets = Invoke-PowerBIRestMethod -Url "groups/$groupId/datasets" -Method Get | ConvertFrom-Json
$ds = $datasets.value | Where-Object { $_.name -like "*Ospedale*" }
if (-not $ds) { throw "Dataset '$Dataset' no encontrado." }
$datasetId = $ds.id
Write-Host " OK ($datasetId)" -ForegroundColor Green

# ──────────────────────────────────────────────────────────────────────────────
# 4. Función helper para ejecutar DAX y devolver DataFrame (array de hashes)
# ──────────────────────────────────────────────────────────────────────────────
function Invoke-DAX {
    param([string]$Dax, [string]$Label = "")
    $url     = "groups/$groupId/datasets/$datasetId/executeQueries"
    $payload = @{ queries = @(@{ query = $Dax }); serializerSettings = @{ includeNulls = $true } } | ConvertTo-Json -Depth 5
    Write-Host "  Ejecutando $Label..." -NoNewline
    $resp  = Invoke-PowerBIRestMethod -Url $url -Method Post -Body $payload | ConvertFrom-Json
    $rows  = $resp.results[0].tables[0].rows
    Write-Host " $($rows.Count) filas" -ForegroundColor Green
    return $rows
}

function Invoke-DAX-Chunks {
    param([scriptblock]$BuildQuery, [array]$Ranges, [string]$Label)
    $all = @()
    for ($i = 0; $i -lt $Ranges.Count; $i++) {
        $start = $Ranges[$i][0]; $end = $Ranges[$i][1]
        $lbl = "$Label chunk $($i+1)/$($Ranges.Count) ($($start.ToString('yyyy-MM-dd')) → $($end.ToString('yyyy-MM-dd')))"
        $dax = & $BuildQuery $start $end
        $rows = Invoke-DAX -Dax $dax -Label $lbl
        $all += $rows
        Start-Sleep -Seconds 1
    }
    return $all
}

function Save-CSV {
    param([array]$Rows, [string]$FilePath)
    if ($Rows.Count -eq 0) { Write-Warning "Sin filas para $FilePath"; return }
    $Rows | ForEach-Object {
        $h = [ordered]@{}
        $_.PSObject.Properties | ForEach-Object { $h[$_.Name] = $_.Value }
        [pscustomobject]$h
    } | Export-Csv -Path $FilePath -NoTypeInformation -Encoding UTF8
    $kb = [int]((Get-Item $FilePath).Length / 1024)
    Write-Host "  Guardado: $FilePath ($($Rows.Count) filas, ${kb}KB)" -ForegroundColor Cyan
}

New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

# ──────────────────────────────────────────────────────────────────────────────
# 5. Q1 — Métricas triage horarias (una sola consulta)
# ──────────────────────────────────────────────────────────────────────────────
$csv1 = Join-Path $DataDir "triage_metricas_horarias.csv"
if ($Forzar -or -not (Test-Path $csv1)) {
    Write-Host "`n[Q1] Métricas triage por fecha+hora..." -ForegroundColor Magenta
    $q1 = @"
EVALUATE
SUMMARIZECOLUMNS(
    'Triage'[FECHA],
    'Triage'[Hora],
    FILTER(ALL('Triage'), 'Triage'[Nit_Clinica] = "$NIT"),

    "cant_triage",
        CALCULATE(COUNTROWS('Triage'), 'Triage'[Cla_Triage] <> 0, 'Triage'[Cla_Triage] <> 1),

    "abandonos_triage",
        CALCULATE(COUNTROWS('Triage'), 'Triage'[Tipo_Abandono_Triage] = "Abandono Post-Triage"),

    "tiempo_turnero_min",
        CALCULATE(AVERAGE('Triage'[Oportunidad_Digiturno]), 'Triage'[Cla_Triage] <> 1),

    "tiempo_consulta_min",
        CALCULATE(AVERAGE('Triage'[Oportunidad_Consulta]), 'Triage'[Marca] = "Urgencia"),

    "pct_cumpl_oportunidad_triage",
        DIVIDE(
            CALCULATE(SUM('Triage'[Cumplimiento Oportunidad Triage]), 'Triage'[Cla_Triage] <> 0, 'Triage'[Cla_Triage] <> 1),
            CALCULATE(COUNTROWS('Triage'), 'Triage'[Cla_Triage] <> 0, 'Triage'[Cla_Triage] <> 1)
        ),

    "pct_cumpl_consulta_t2",
        DIVIDE(
            CALCULATE(SUM('Triage'[Cumplimiento Consulta Urgencias (Triage 2 y 3)]), 'Triage'[Cla_Triage] = 2, 'Triage'[Marca] = "Urgencia"),
            CALCULATE(COUNTROWS('Triage'), 'Triage'[Cla_Triage] = 2, 'Triage'[Marca] = "Urgencia")
        )
)
ORDER BY [FECHA], [Hora]
"@
    $rows1 = Invoke-DAX -Dax $q1 -Label "Q1 triage horario"
    Save-CSV -Rows $rows1 -FilePath $csv1
} else {
    Write-Host "[Q1] Ya existe $csv1, saltando (usa -Forzar para re-descargar)"
}

# ──────────────────────────────────────────────────────────────────────────────
# 6. Q2 — Timestamps urgencias (2 chunks anuales)
# ──────────────────────────────────────────────────────────────────────────────
$csv2 = Join-Path $DataDir "urgencias_timestamps.csv"
if ($Forzar -or -not (Test-Path $csv2)) {
    Write-Host "`n[Q2] Timestamps *Urgencias (ingresos/egresos)..." -ForegroundColor Magenta
    $buildQ2 = {
        param([datetime]$s, [datetime]$e)
        $y1=$s.Year; $m1=$s.Month; $d1=$s.Day
        $y2=$e.Year; $m2=$e.Month; $d2=$e.Day
        return @"
EVALUATE
SELECTCOLUMNS(
    FILTER(
        '*Urgencias',
        '*Urgencias'[Nit_Clinica] = "$NIT"
        && '*Urgencias'[Fecha Ingreso Pabellon] >= DATE($y1,$m1,$d1)
        && '*Urgencias'[Fecha Ingreso Pabellon] <  DATE($y2,$m2,$d2)
    ),
    "cnsc_ing",         '*Urgencias'[Cnsc_Ing],
    "fecha_ingreso",    '*Urgencias'[Fecha Ingreso Pabellon],
    "hora_ingreso",     '*Urgencias'[Hora Ingreso Pab],
    "fecha_egreso",     '*Urgencias'[Fecha Egreso Pabellon],
    "hora_egreso",      '*Urgencias'[Hora Egreso Pab],
    "tipo_estancia",    '*Urgencias'[Tipo Estancia],
    "horas_estancia",   '*Urgencias'[Sum Horas Estancia]
)
ORDER BY [fecha_ingreso], [hora_ingreso]
"@
    }
    $ranges2 = @(
        @([datetime]"2024-01-01", [datetime]"2025-01-01"),
        @([datetime]"2025-01-01", [datetime]"2026-06-01")
    )
    $rows2 = Invoke-DAX-Chunks -BuildQuery $buildQ2 -Ranges $ranges2 -Label "Q2 *Urgencias"
    Save-CSV -Rows $rows2 -FilePath $csv2
} else {
    Write-Host "[Q2] Ya existe $csv2, saltando (usa -Forzar para re-descargar)"
}

# ──────────────────────────────────────────────────────────────────────────────
# 7. Q3 — Timestamps triage (5 chunks semestrales)
# ──────────────────────────────────────────────────────────────────────────────
$csv3 = Join-Path $DataDir "triage_timestamps.csv"
if ($Forzar -or -not (Test-Path $csv3)) {
    Write-Host "`n[Q3] Timestamps Triage (admisión/resolución)..." -ForegroundColor Magenta
    $buildQ3 = {
        param([datetime]$s, [datetime]$e)
        $y1=$s.Year; $m1=$s.Month; $d1=$s.Day
        $y2=$e.Year; $m2=$e.Month; $d2=$e.Day
        return @"
EVALUATE
SELECTCOLUMNS(
    FILTER(
        'Triage',
        'Triage'[Nit_Clinica] = "$NIT"
        && 'Triage'[FECHA] >= DATE($y1,$m1,$d1)
        && 'Triage'[FECHA] <  DATE($y2,$m2,$d2)
        && 'Triage'[Cla_Triage] <> 0
        && 'Triage'[Cla_Triage] <> 1
    ),
    "cnsc_ing",               'Triage'[Cnsc_Ing],
    "fecha_triage",           'Triage'[FECHA],
    "cla_triage",             'Triage'[Cla_Triage],
    "fecha_admision",         'Triage'[Admision - Fecha],
    "hora_admision",          'Triage'[Admision - Hora],
    "fecha_resolucion",       'Triage'[Fecha_ResolucionC],
    "fecha_ingreso_consulta", 'Triage'[Fecha_Ingreso_Consulta]
)
ORDER BY [fecha_admision], [hora_admision]
"@
    }
    $ranges3 = @(
        @([datetime]"2024-01-01", [datetime]"2024-07-01"),
        @([datetime]"2024-07-01", [datetime]"2025-01-01"),
        @([datetime]"2025-01-01", [datetime]"2025-07-01"),
        @([datetime]"2025-07-01", [datetime]"2026-01-01"),
        @([datetime]"2026-01-01", [datetime]"2026-06-01")
    )
    $rows3 = Invoke-DAX-Chunks -BuildQuery $buildQ3 -Ranges $ranges3 -Label "Q3 Triage"
    Save-CSV -Rows $rows3 -FilePath $csv3
} else {
    Write-Host "[Q3] Ya existe $csv3, saltando (usa -Forzar para re-descargar)"
}

# ──────────────────────────────────────────────────────────────────────────────
# 8. Resumen final
# ──────────────────────────────────────────────────────────────────────────────
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Extracción completada" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Get-ChildItem $DataDir -Filter "*.csv" | Select-Object Name, @{N='Filas';E={(Import-Csv $_.FullName).Count}}, @{N='KB';E={[int]($_.Length/1024)}} | Format-Table -AutoSize

Write-Host "`nPróximo paso:" -ForegroundColor Yellow
Write-Host "  python src/preparar_dataset.py" -ForegroundColor White
