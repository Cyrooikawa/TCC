@echo off
REM ==========================================================================
REM  Reproduz os graficos do TCC (Windows). De dois cliques neste arquivo.
REM  Na primeira vez, instala o ambiente (alguns minutos). Depois, so roda.
REM  Tudo o que aparece na tela fica gravado em log_reproducao.txt.
REM ==========================================================================
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set LOG=%~dp0log_reproducao.txt
echo Reproducao dos graficos do TCC - %date% %time% > "%LOG%"

echo.
echo [1/5] Conferindo a pasta...
if not exist "gerar_figuras.py" (
    echo ERRO: gerar_figuras.py nao esta nesta pasta: %cd%
    echo Este arquivo .bat precisa ficar dentro de 02_codigo_e_resultados.
    echo ERRO: pasta errada %cd% >> "%LOG%"
    goto fim
)
if not exist "dados\processados\series_diarias.csv" (
    echo ERRO: falta dados\processados\series_diarias.csv
    echo ERRO: falta series_diarias.csv >> "%LOG%"
    goto fim
)
echo Pasta OK: %cd%

echo.
echo [2/5] Procurando o Python 3.12...
set PY=
py -3.12 --version > nul 2>&1 && set PY=py -3.12
if "%PY%"=="" (
    python --version > nul 2>&1 && set PY=python
)
if "%PY%"=="" (
    echo ERRO: Python nao encontrado. Instale o Python 3.12 em python.org
    echo e marque "Add python.exe to PATH" na primeira tela do instalador.
    echo ERRO: python nao encontrado >> "%LOG%"
    goto fim
)
%PY% --version
%PY% --version >> "%LOG%" 2>&1

echo.
echo [3/5] Preparando o ambiente (so demora na primeira vez)...
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo ERRO ao criar o ambiente. Veja log_reproducao.txt
        goto fim
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1
    echo Instalando bibliotecas, aguarde alguns minutos...
    ".venv\Scripts\python.exe" -m pip install numpy==2.4.6 pandas==2.3.3 scipy==1.17.1 matplotlib==3.11.2 statsmodels==0.15.0 prophet==1.4.0 holidays==0.104 python-dateutil >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo ERRO na instalacao das bibliotecas. Veja log_reproducao.txt
        rmdir /s /q .venv
        goto fim
    )
)
echo Ambiente OK.
set VPY=".venv\Scripts\python.exe"

echo.
echo [4/5] Conferindo os procedimentos (71 verificacoes)...
%VPY% validar_procedimentos.py >> "%LOG%" 2>&1
if errorlevel 1 (
    echo AVISO: alguma verificacao falhou. Veja log_reproducao.txt
) else (
    echo Todas as verificacoes passaram.
)

echo.
echo [5/5] Gerando os graficos...
for %%S in (gerar_figuras.py resultados_completos.py faixas_antecedencia_direto.py figura_boituva_todos_modelos.py figura_sarima_boituva.py) do (
    echo    %%S
    echo ===== %%S >> "%LOG%"
    %VPY% %%S >> "%LOG%" 2>&1
    if errorlevel 1 echo    ERRO em %%S. Veja log_reproducao.txt
)

echo.
echo Pronto. Os graficos estao em:
echo    %cd%\figuras
echo    %cd%\diagnostico\sarima_boituva_ordem_atual_x_ajuste_final.png
start "" "%cd%\figuras"

:fim
echo.
echo Registro completo em: %LOG%
pause
