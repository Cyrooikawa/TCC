@echo off
REM ==========================================================================
REM  Valida os resultados do capitulo 4 no seu computador (Windows).
REM  De dois cliques neste arquivo. Ele roda os modelos e compara os numeros
REM  obtidos com os esperados (conferir_resultados.py).
REM  Na primeira vez, instala o ambiente (alguns minutos).
REM  Tudo o que aparece na tela fica gravado em log_validacao.txt.
REM
REM  Configuracao: por padrao, a AJUSTADA (a do capitulo 4). Para validar a
REM  original, rode num terminal:  set TCC_CONFIGURACAO=original  e depois
REM  este arquivo, de preferencia numa copia separada da pasta.
REM ==========================================================================
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set LOG=%~dp0log_validacao.txt
echo Validacao dos resultados do TCC - %date% %time% > "%LOG%"

echo.
echo [1/6] Conferindo a pasta...
if not exist "conferir_resultados.py" (
    echo ERRO: conferir_resultados.py nao esta nesta pasta: %cd%
    goto fim
)
if not exist "dados\processados\series_diarias.csv" (
    echo ERRO: falta dados\processados\series_diarias.csv
    echo Copie esse arquivo da sua pasta 02_codigo_e_resultados\dados\processados.
    goto fim
)
if not exist "resultados\redes" mkdir "resultados\redes"
echo Pasta OK: %cd%

echo.
echo [2/6] Procurando o Python 3.12...
set PY=
py -3.12 --version > nul 2>&1 && set PY=py -3.12
if "%PY%"=="" (
    python --version > nul 2>&1 && set PY=python
)
if "%PY%"=="" (
    echo ERRO: Python nao encontrado. Instale o Python 3.12 em python.org
    echo e marque "Add python.exe to PATH" na primeira tela do instalador.
    goto fim
)
%PY% --version
%PY% --version >> "%LOG%" 2>&1

echo.
echo [3/6] Preparando o ambiente (so demora na primeira vez)...
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo ERRO ao criar o ambiente. Veja log_validacao.txt
        goto fim
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1
    echo Instalando bibliotecas, aguarde alguns minutos...
    ".venv\Scripts\python.exe" -m pip install numpy==2.4.6 pandas==2.3.3 scipy==1.17.1 matplotlib==3.11.2 statsmodels==0.15.0 pmdarima==2.1.1 prophet==1.4.0 holidays==0.104 python-dateutil >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo ERRO na instalacao das bibliotecas. Veja log_validacao.txt
        rmdir /s /q .venv
        goto fim
    )
)
echo Ambiente OK.
set VPY=".venv\Scripts\python.exe"
%VPY% -c "import config; print('Configuracao em uso:', config.CONFIGURACAO.upper())"

echo.
echo [4/6] Modelos estatisticos (cerca de 10 minutos)...
for %%S in (modelo_ingenuo.py arima_baseline.py prophet_modelo.py) do (
    echo    %%S
    echo ===== %%S >> "%LOG%"
    %VPY% %%S >> "%LOG%" 2>&1
    if errorlevel 1 echo    ERRO em %%S. Veja log_validacao.txt
)

echo.
echo [5/6] Redes neurais. Escolha:
echo    1 = pular as redes
echo    2 = so a LSTM de 180 dias, a unica rede que mudou (cerca de 25 minutos)
echo    3 = todas as 15 redes (cerca de 2 a 4 horas)
choice /c 123 /n /m "Opcao (1, 2 ou 3): "
if errorlevel 3 (
    echo Rodando as 15 redes, duas por vez...
    %VPY% redes_neurais.py --processos 2 >> "%LOG%" 2>&1
    goto conferir
)
if errorlevel 2 (
    echo Rodando a LSTM de 180 dias...
    %VPY% redes_neurais.py LSTM 180 >> "%LOG%" 2>&1
)

:conferir
echo.
echo [6/6] Comparando com os resultados esperados...
%VPY% conferir_resultados.py
%VPY% conferir_resultados.py >> "%LOG%" 2>&1

if exist "resultados\nhits_metricas_por_origem.csv" (
    echo.
    echo Gerando as tabelas e figuras do capitulo 4...
    REM As analises de sensibilidade vem antes, porque resultados_completos.py as le.
    for %%S in (analises_sensibilidade.py prophet_sem_feriados.py) do (
        echo    %%S
        echo ===== %%S >> "%LOG%"
        %VPY% %%S >> "%LOG%" 2>&1
        if errorlevel 1 echo    ERRO em %%S. Veja log_validacao.txt
    )
    echo    prophet_sem_feriados.py --sem-anual
    %VPY% prophet_sem_feriados.py --sem-anual >> "%LOG%" 2>&1
    for %%S in (faixas_antecedencia_direto.py resultados_completos.py testes_blocos_por_origem.py gerar_figuras.py figura_boituva_todos_modelos.py validar_procedimentos.py) do (
        echo    %%S
        echo ===== %%S >> "%LOG%"
        %VPY% %%S >> "%LOG%" 2>&1
        if errorlevel 1 echo    ERRO em %%S. Veja log_validacao.txt
    )
    start "" "%cd%\figuras"
)

:fim
echo.
echo Registro completo em: %LOG%
pause
