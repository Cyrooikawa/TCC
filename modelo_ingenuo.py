"""
Modelo ingênuo sazonal: a previsão é simplesmente repetir a última semana
observada (a próxima segunda-feira terá o mesmo volume da última segunda, e
assim por diante).

Não é um modelo para ser "bom" -- é a régua mínima. Qualquer modelo do
trabalho (ARIMA, Prophet, redes neurais) só se justifica se errar menos que
isto. É avaliado exatamente nas mesmas datas de partida, mesmos horizontes e
mesmas métricas dos demais modelos.

Entrada:  dados/processados/series_diarias.csv
Saída:    resultados/ingenuo_metricas_por_origem.csv
          resultados/ingenuo_resumo.csv

Como executar (leva poucos segundos):
    python modelo_ingenuo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config
import metricas

RAIZ = Path(__file__).resolve().parent
ARQUIVO_SERIES = RAIZ / "dados" / "processados" / "series_diarias.csv"
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)

COLUNAS_METRICAS = ["MAE", "RMSE", "MAPE", "SMAPE", "DTW"]


def carregar_serie(series: pd.DataFrame, praca: str) -> pd.Series:
    sub = series[(series["PRACA"] == praca) & (series["DATA"] >= config.DATA_INICIO_MODELAGEM)]
    return sub.set_index("DATA")["VOLUME_TOTAL"].sort_index().asfreq("D")


def avaliar_praca(y: pd.Series, praca: str, grupo: str) -> list:
    n = len(y)
    periodo = config.PERIODO_SAZONAL
    horizonte_min = min(config.HORIZONTES.values())
    linhas = []

    for origem in metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS, horizonte_min):
        # Última semana conhecida até a origem; se houver dia vazio, usa o
        # valor do dia anterior.
        ultima_semana = y.iloc[: origem + 1].ffill().iloc[-periodo:].values

        for nome_h, h in config.HORIZONTES.items():
            if origem + h > n - 1:
                continue
            previsao = np.tile(ultima_semana, h // periodo + 1)[:h]
            real = y.iloc[origem + 1: origem + 1 + h].values
            resultado = metricas.todas(real, previsao, config.JANELA_DTW)
            resultado.update({
                "MODELO": "INGENUO", "GRUPO": grupo, "PRACA": praca,
                "HORIZONTE": nome_h, "DIAS": h, "ORIGEM": y.index[origem].date(),
            })
            linhas.append(resultado)
    return linhas


if __name__ == "__main__":
    series = pd.read_csv(ARQUIVO_SERIES, parse_dates=["DATA"])
    alvos = [(p, "NUCLEO") for p in config.PRACAS_NUCLEO] + \
            [(p, "UNIDIRECIONAL") for p in config.PRACAS_UNIDIRECIONAIS]

    linhas = []
    for praca, grupo in alvos:
        linhas.extend(avaliar_praca(carregar_serie(series, praca), praca, grupo))

    por_origem = pd.DataFrame(linhas)
    por_origem.to_csv(PASTA_RESULTADOS / "ingenuo_metricas_por_origem.csv", index=False, encoding="utf-8-sig")

    chaves = ["GRUPO", "PRACA", "HORIZONTE", "DIAS"]
    resumo = por_origem.groupby(chaves)[COLUNAS_METRICAS].mean().round(2)
    resumo["N_ORIGENS"] = por_origem.groupby(chaves).size()
    resumo = resumo.reset_index().sort_values(["GRUPO", "PRACA", "DIAS"])
    resumo.to_csv(PASTA_RESULTADOS / "ingenuo_resumo.csv", index=False, encoding="utf-8-sig")

    print("=== MODELO INGÊNUO (repetir a última semana) ===")
    print(resumo.to_string(index=False))
    print(f"\nResultados salvos em: {PASTA_RESULTADOS}")
