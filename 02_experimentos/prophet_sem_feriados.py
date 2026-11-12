"""
Terceira análise de sensibilidade da seção 4.4.5: Prophet SEM o calendário de
feriados (ponto 4 da validação de 02/11/2026).

Por que: só o Prophet recebe o calendário de feriados (inclusive os futuros).
O SARIMA e as redes recebem apenas os volumes passados. Para saber quanto da
vantagem do Prophet vem dessa informação, ele é rodado de novo exatamente como
em prophet_modelo.py, mudando uma única coisa: holidays=None.

Continuam iguais: a tendência linear, as sazonalidades semanal e anual, o
reajuste em cada origem (só com os dados até ela), as mesmas praças, origens,
horizontes e métricas.

Leitura do resultado:
  - se o erro subir até o nível das redes, a vantagem do Prophet vem
    principalmente do calendário;
  - se continuar abaixo das redes, a vantagem vem também da estrutura do modelo
    (por exemplo, a sazonalidade anual, que só ele representa).

Entrada:  dados/processados/series_diarias.csv
Saída:    resultados/sensibilidade_prophet_sem_feriados.csv
          (por origem; o nome não termina em "_metricas_por_origem.csv",
           para que comparar_modelos.py e resultados_completos.py não o
           confundam com um modelo da comparação principal)
          resultados/sensibilidade_prophet_sem_feriados_previsoes.npz
          (previsões diárias de até 180 dias por praça e origem)

Variante opcional, --sem-anual: tira também a sazonalidade anual (fica só a
tendência e a sazonalidade semanal). Separa o efeito do calendário de feriados do
efeito da sazonalidade anual, que só o Prophet representa. Saída:
resultados/sensibilidade_prophet_sem_feriados_sem_anual.csv (e _previsoes.npz).

Como executar (de 2 a 20 minutos; o modelo é reajustado em cada origem):
    python prophet_sem_feriados.py
    python prophet_sem_feriados.py --sem-anual
"""

import logging
import sys
import time
import warnings
from pathlib import Path

# Silencia as mensagens internas do Prophet (precisa vir antes de importá-lo).
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("prophet.plot").disabled = True

import numpy as np
import pandas as pd
from prophet import Prophet

import config
import metricas
import prophet_modelo as pm

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent
ARQUIVO_SERIES = RAIZ / "dados" / "processados" / "series_diarias.csv"
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)

SEM_ANUAL = "--sem-anual" in sys.argv
NOME_MODELO = "PROPHET_SEM_FERIADOS_SEM_ANUAL" if SEM_ANUAL else "PROPHET_SEM_FERIADOS"
ARQUIVO = "sensibilidade_" + NOME_MODELO.lower()


def ajustar_sem_feriados(y_ate_origem: pd.Series) -> Prophet:
    """Igual a prophet_modelo.ajustar, mas sem o calendário de feriados."""
    dados = y_ate_origem.dropna().reset_index()
    dados.columns = ["ds", "y"]
    modelo = Prophet(
        growth="linear",
        weekly_seasonality=True,
        yearly_seasonality=not SEM_ANUAL,   # só a variante --sem-anual tira a sazonalidade anual
        daily_seasonality=False,
        holidays=None,          # a diferença em relação ao modelo principal
        uncertainty_samples=0,
    )
    modelo.fit(dados)
    return modelo


def avaliar_praca(y: pd.Series, praca: str, grupo: str):
    n = len(y)
    horizonte_max = max(config.HORIZONTES.values())
    horizonte_min = min(config.HORIZONTES.values())
    origens = metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS, horizonte_min)
    print(f"\n=== {praca} ({grupo}): {len(origens)} origens, reajuste em cada uma ===", flush=True)

    linhas, previsoes = [], []
    for k, origem in enumerate(origens, start=1):
        modelo = ajustar_sem_feriados(y.iloc[: origem + 1])
        passos = min(horizonte_max, n - 1 - origem)
        previsao = pm.prever(modelo, y.index[origem + 1], passos)

        diaria = np.full(horizonte_max, np.nan)
        diaria[:passos] = previsao.values
        previsoes.append((praca, str(y.index[origem].date()), diaria))

        for nome_h, h in config.HORIZONTES.items():
            if origem + h > n - 1:
                continue
            real = y.iloc[origem + 1: origem + 1 + h].values
            resultado = metricas.todas(real, previsao.iloc[:h].values, config.JANELA_DTW)
            resultado.update({
                "MODELO": NOME_MODELO, "GRUPO": grupo, "PRACA": praca,
                "HORIZONTE": nome_h, "DIAS": h, "ORIGEM": y.index[origem].date(),
            })
            linhas.append(resultado)
        if k % 20 == 0 or k == len(origens):
            print(f"  {k}/{len(origens)} origens concluídas", flush=True)
    return linhas, previsoes


if __name__ == "__main__":
    inicio = time.time()
    series = pd.read_csv(ARQUIVO_SERIES, parse_dates=["DATA"])
    alvos = [(p, "NUCLEO") for p in config.PRACAS_NUCLEO] + \
            [(p, "UNIDIRECIONAL") for p in config.PRACAS_UNIDIRECIONAIS]

    linhas, previsoes = [], []
    for praca, grupo in alvos:
        l_, p_ = avaliar_praca(pm.carregar_serie(series, praca), praca, grupo)
        linhas.extend(l_)
        previsoes.extend(p_)

    por_origem = pd.DataFrame(linhas)
    por_origem.to_csv(PASTA_RESULTADOS / f"{ARQUIVO}.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(PASTA_RESULTADOS / f"{ARQUIVO}_previsoes.npz",
                        praca=np.array([p for p, _, _ in previsoes]),
                        origem=np.array([o for _, o, _ in previsoes]),
                        previsao=np.array([v for _, _, v in previsoes], dtype=np.float64))

    # Comparação com o Prophet principal (mesmas praças e origens).
    oficial = pd.read_csv(PASTA_RESULTADOS / "prophet_metricas_por_origem.csv")
    chaves = ["PRACA", "DIAS"]
    a = oficial.groupby(chaves).SMAPE.mean().rename("PROPHET")
    b = por_origem.groupby(chaves).SMAPE.mean().rename(NOME_MODELO)
    tabela = pd.concat([a, b], axis=1)
    tabela["DIFERENCA_PP"] = tabela[NOME_MODELO] - tabela["PROPHET"]
    print(f"\n=== SMAPE médio (%) por praça: Prophet principal x {NOME_MODELO} ===")
    print(tabela.round(2).to_string())
    print("\n=== Média das seis praças ===")
    print(tabela.groupby(level="DIAS").mean().round(2).to_string())
    print(f"\nTempo total: {(time.time() - inicio) / 60:.1f} min. Resultados em {PASTA_RESULTADOS}")
