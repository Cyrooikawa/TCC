"""
Duas análises de sensibilidade, citadas no capítulo 4 da monografia.

(A) ARIMA SEM componente sazonal.
    O modelo de referência do trabalho é o SARIMA (ARIMA com termos sazonais de
    período 7). Para medir o que a parte sazonal representa, o ARIMA é ajustado
    também sem ela, com o mesmo desenho experimental (mesma divisão, mesmas
    datas de partida, mesmas métricas, ajuste em milhares de veículos).
    As ordens (p,d,q) abaixo foram escolhidas por auto_arima (menor AIC, p e q
    até 3, sem parte sazonal) sobre o treino e ficam fixadas aqui, para que
    qualquer computador reproduza o mesmo modelo. Para refazer a busca:
        python analises_sensibilidade.py --buscar

(B) Prophet com AJUSTE ÚNICO.
    Na avaliação principal, o Prophet é reajustado em cada data de partida,
    porque a previsão dele depende só da data e, sem reajuste, ele não "enxerga"
    os últimos dias observados. O SARIMA, ao contrário, mantém os parâmetros do
    treino. Para verificar se o reajuste favorece o Prophet, ele é ajustado uma
    única vez, só com o treino, e avaliado nas mesmas datas.

Entrada:  dados/processados/series_diarias.csv
Saída:    resultados/sensibilidade_arima_sem_sazonalidade.csv
          resultados/sensibilidade_prophet_ajuste_unico.csv
          (por data de partida; os nomes não terminam em
           "_metricas_por_origem.csv", para que comparar_modelos.py nunca os
           confunda com resultados da comparação principal)

Como executar (depois de preparar_series.py; leva poucos minutos):
    python analises_sensibilidade.py
"""

import logging
import sys
import warnings
from pathlib import Path

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("prophet.plot").disabled = True

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config
import metricas
import prophet_modelo as pm

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent
ARQUIVO_SERIES = RAIZ / "dados" / "processados" / "series_diarias.csv"
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)

ESCALA = 1000  # ajuste em milhares de veículos (mesmo procedimento do arima_baseline.py)

ORDENS_SEM_SAZONALIDADE = {
    "ALAMBARI":            (2, 1, 3),
    "BLOQUEIO DE BOITUVA": (3, 1, 3),
    "BOITUVA":             (1, 1, 2),
    "PORTO FELIZ":         (2, 1, 3),
    "SOROCABA":            (0, 1, 1),
    "ITU":                 (2, 1, 3),
}


def todas_as_pracas():
    return [(p, "NUCLEO") for p in config.PRACAS_NUCLEO] + \
           [(p, "UNIDIRECIONAL") for p in config.PRACAS_UNIDIRECIONAIS]


def serie(series: pd.DataFrame, praca: str) -> pd.Series:
    sub = series[(series["PRACA"] == praca) & (series["DATA"] >= config.DATA_INICIO_MODELAGEM)]
    return sub.set_index("DATA")["VOLUME_TOTAL"].sort_index().asfreq("D")


def buscar_ordem(treino: pd.Series):
    import pmdarima as pm_arima
    base = treino.interpolate(method="linear").bfill().ffill()
    modelo = pm_arima.auto_arima(base, seasonal=False, max_p=3, max_q=3, stepwise=True,
                                 suppress_warnings=True, error_action="ignore")
    return modelo.order


def arima_sem_sazonalidade(y, praca, grupo, buscar):
    n = len(y)
    n_treino = int(n * config.PROPORCAO_TREINO)
    treino = y.iloc[:n_treino]
    ordem = buscar_ordem(treino) if buscar else ORDENS_SEM_SAZONALIDADE[praca]
    ajuste = SARIMAX(treino / ESCALA, order=ordem, seasonal_order=(0, 0, 0, 0),
                     trend="c" if ordem[1] == 0 else "n",
                     enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
    print(f"  ARIMA sem sazonalidade, ordem {ordem}")
    horizonte_max = max(config.HORIZONTES.values())
    linhas = []
    for origem in metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS,
                                   min(config.HORIZONTES.values())):
        passos = min(horizonte_max, n - 1 - origem)
        previsao = ajuste.apply(y.iloc[: origem + 1] / ESCALA).forecast(steps=passos) * ESCALA
        for nome_h, h in config.HORIZONTES.items():
            if origem + h > n - 1:
                continue
            r = metricas.todas(y.iloc[origem + 1: origem + 1 + h].values, previsao.iloc[:h].values,
                               config.JANELA_DTW)
            r.update({"MODELO": "ARIMA_SEM_SAZONALIDADE", "GRUPO": grupo, "PRACA": praca,
                      "HORIZONTE": nome_h, "DIAS": h, "ORIGEM": y.index[origem].date(),
                      "ORDEM": str(ordem)})
            linhas.append(r)
    return linhas


def prophet_ajuste_unico(y, praca, grupo):
    n = len(y)
    n_treino = int(n * config.PROPORCAO_TREINO)
    modelo = pm.ajustar(y.iloc[:n_treino])
    previsao_total = pm.prever(modelo, y.index[n_treino], n - n_treino)
    linhas = []
    for origem in metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS,
                                   min(config.HORIZONTES.values())):
        for nome_h, h in config.HORIZONTES.items():
            if origem + h > n - 1:
                continue
            real = y.iloc[origem + 1: origem + 1 + h].values
            prev = previsao_total.loc[y.index[origem + 1]: y.index[origem + h]].values
            r = metricas.todas(real, prev, config.JANELA_DTW)
            r.update({"MODELO": "PROPHET_AJUSTE_UNICO", "GRUPO": grupo, "PRACA": praca,
                      "HORIZONTE": nome_h, "DIAS": h, "ORIGEM": y.index[origem].date()})
            linhas.append(r)
    return linhas


if __name__ == "__main__":
    buscar = "--buscar" in sys.argv
    series = pd.read_csv(ARQUIVO_SERIES, parse_dates=["DATA"])
    linhas_a, linhas_b = [], []
    for praca, grupo in todas_as_pracas():
        print(f"=== {praca} ({grupo}) ===")
        y = serie(series, praca)
        linhas_a.extend(arima_sem_sazonalidade(y, praca, grupo, buscar))
        linhas_b.extend(prophet_ajuste_unico(y, praca, grupo))

    pd.DataFrame(linhas_a).to_csv(PASTA_RESULTADOS / "sensibilidade_arima_sem_sazonalidade.csv",
                                  index=False, encoding="utf-8-sig")
    pd.DataFrame(linhas_b).to_csv(PASTA_RESULTADOS / "sensibilidade_prophet_ajuste_unico.csv",
                                  index=False, encoding="utf-8-sig")

    chaves = ["PRACA", "DIAS"]
    for nome, linhas in [("ARIMA sem sazonalidade", linhas_a), ("Prophet com ajuste único", linhas_b)]:
        r = pd.DataFrame(linhas).groupby(chaves)[["MAPE", "SMAPE", "DTW"]].mean().round(2)
        print(f"\n=== {nome} (média das datas de partida) ===")
        print(r.to_string())
    print(f"\nResultados salvos em: {PASTA_RESULTADOS}")
