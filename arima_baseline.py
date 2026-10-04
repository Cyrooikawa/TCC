"""
ARIMA (SARIMA com sazonalidade semanal) como modelo de referência (baseline).

Desenho, conforme combinado com o orientador e definido em config.py:
  - série diária de 2021-01-01 em diante (2020 fora, por causa da pandemia);
  - 70% iniciais para treino, 30% finais para teste;
  - horizontes de 7, 90 e 180 dias.

Como funciona, passo a passo, para cada praça:
  1. Teste de Dickey-Fuller Aumentado no treino (registrado para o texto).
  2. Escolha automática da ordem do modelo (p,d,q)(P,D,Q)[7] com auto_arima.
     Como o auto_arima não aceita dias vazios, só nesta etapa de ESCOLHA as
     lacunas são preenchidas por interpolação linear.
  3. Ajuste final com SARIMAX (statsmodels) na série ORIGINAL, com as lacunas
     preservadas -- o SARIMAX trata dias vazios nativamente (filtro de
     Kalman), sem inventar valores.
  4. Avaliação no período de teste a partir de várias origens, de 7 em 7
     dias: em cada origem, o modelo (com os parâmetros estimados só no
     treino) recebe os dados reais até aquele dia e prevê 7, 90 e 180 dias.

Entrada:  dados/processados/series_diarias.csv  (gerado por preparar_series.py)
Saída:    resultados/arima_metricas_por_origem.csv
          resultados/arima_resumo.csv
          resultados/arima_modelos.csv
          resultados/arima_<PRACA>.png

Como executar:
    python arima_baseline.py                  (todas as praças, alguns minutos)
    python arima_baseline.py "PORTO FELIZ"    (só uma praça, para teste rápido)
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

import config
import metricas

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent
ARQUIVO_SERIES = RAIZ / "dados" / "processados" / "series_diarias.csv"
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)


def carregar_serie(series: pd.DataFrame, praca: str) -> pd.Series:
    sub = series[(series["PRACA"] == praca) & (series["DATA"] >= config.DATA_INICIO_MODELAGEM)]
    y = sub.set_index("DATA")["VOLUME_TOTAL"].sort_index()
    return y.asfreq("D")


def escolher_ordem(treino: pd.Series):
    treino_sem_lacuna = treino.interpolate(method="linear").bfill().ffill()
    modelo = pm.auto_arima(
        treino_sem_lacuna,
        seasonal=True,
        m=config.PERIODO_SAZONAL,
        max_p=3, max_q=3, max_P=2, max_Q=2,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    return modelo.order, modelo.seasonal_order


def avaliar_praca(y: pd.Series, praca: str, grupo: str):
    n = len(y)
    n_treino = int(n * config.PROPORCAO_TREINO)
    treino = y.iloc[:n_treino]

    print(f"\n=== {praca} ({grupo}) ===")
    print(f"Treino: {treino.index.min().date()} a {treino.index.max().date()} "
          f"({n_treino} dias, {int(treino.isna().sum())} sem dado)")
    print(f"Teste:  {y.index[n_treino].date()} a {y.index.max().date()} ({n - n_treino} dias)")

    adf = adfuller(treino.dropna())
    print(f"Dickey-Fuller aumentado no treino: p-valor = {adf[1]:.4f}")

    print("Escolhendo a ordem do modelo (auto_arima)...")
    ordem, ordem_sazonal = escolher_ordem(treino)
    d_total = ordem[1] + ordem_sazonal[1]
    tendencia = "c" if d_total == 0 else "n"
    print(f"Modelo escolhido: SARIMA{ordem}x{ordem_sazonal}")

    # As restrições de estacionariedade e invertibilidade ficam LIGADAS (padrão
    # do statsmodels). Com elas desligadas, a estimação pode divergir e gerar
    # previsões vazias -- foi o que aconteceu com Boituva num primeiro teste.
    ajuste = SARIMAX(
        treino, order=ordem, seasonal_order=ordem_sazonal, trend=tendencia,
        enforce_stationarity=True, enforce_invertibility=True,
    ).fit(disp=False, maxiter=200)

    if not np.isfinite(ajuste.llf) or not np.isfinite(ajuste.forecast(1)).all():
        raise RuntimeError(f"Ajuste do ARIMA inválido para {praca} -- revisar antes de seguir.")

    horizonte_max = max(config.HORIZONTES.values())
    horizonte_min = min(config.HORIZONTES.values())
    linhas = []
    previsao_primeira_origem = None

    for origem in range(n_treino - 1, n - 1 - horizonte_min + 1, config.PASSO_ORIGENS_DIAS):
        passos = min(horizonte_max, n - 1 - origem)
        ajuste_origem = ajuste.apply(y.iloc[: origem + 1])
        previsao = ajuste_origem.forecast(steps=passos)

        if previsao_primeira_origem is None:
            previsao_primeira_origem = previsao

        for nome_h, h in config.HORIZONTES.items():
            if origem + h > n - 1:
                continue
            real = y.iloc[origem + 1: origem + 1 + h].values
            prev = previsao.iloc[:h].values
            resultado = metricas.todas(real, prev, config.JANELA_DTW)
            resultado.update({
                "MODELO": "ARIMA", "GRUPO": grupo, "PRACA": praca,
                "HORIZONTE": nome_h, "DIAS": h,
                "ORIGEM": y.index[origem].date(),
            })
            linhas.append(resultado)

    info_modelo = {
        "MODELO": "ARIMA", "GRUPO": grupo, "PRACA": praca,
        "ORDEM": str(ordem), "ORDEM_SAZONAL": str(ordem_sazonal),
        "ADF_P_VALOR": round(adf[1], 4), "AIC": round(ajuste.aic, 1),
        "INICIO_TREINO": treino.index.min().date(), "FIM_TREINO": treino.index.max().date(),
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    janela_plot = y.iloc[n_treino - 120: n_treino + horizonte_max]
    ax.plot(janela_plot.index, janela_plot.values, label="Real", linewidth=1)
    ax.plot(previsao_primeira_origem.index, previsao_primeira_origem.values,
            label="ARIMA (previsão a partir do fim do treino)", linewidth=1)
    ax.axvline(y.index[n_treino - 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{praca} — ARIMA, previsão de {horizonte_max} dias a partir do fim do treino")
    ax.set_ylabel("veículos/dia")
    ax.legend()
    plt.tight_layout()
    nome_arquivo = praca.replace(" ", "_")
    plt.savefig(PASTA_RESULTADOS / f"arima_{nome_arquivo}.png", dpi=100)
    plt.close()

    return linhas, info_modelo


if __name__ == "__main__":
    series = pd.read_csv(ARQUIVO_SERIES, parse_dates=["DATA"])

    alvos = [(p, "NUCLEO") for p in config.PRACAS_NUCLEO] + \
            [(p, "CASO_SOROCABA") for p in config.PRACAS_CASO_SOROCABA]
    if len(sys.argv) > 1:
        alvos = [(p, g) for p, g in alvos if p == sys.argv[1].upper()]
        if not alvos:
            print(f"Praça '{sys.argv[1]}' não encontrada. Opções: "
                  f"{config.PRACAS_NUCLEO + config.PRACAS_CASO_SOROCABA}")
            sys.exit(1)

    todas_linhas, todos_modelos = [], []
    for praca, grupo in alvos:
        y = carregar_serie(series, praca)
        linhas, info = avaliar_praca(y, praca, grupo)
        todas_linhas.extend(linhas)
        todos_modelos.append(info)

    por_origem = pd.DataFrame(todas_linhas)
    por_origem.to_csv(PASTA_RESULTADOS / "arima_metricas_por_origem.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(todos_modelos).to_csv(PASTA_RESULTADOS / "arima_modelos.csv", index=False, encoding="utf-8-sig")

    resumo = (por_origem
              .groupby(["GRUPO", "PRACA", "HORIZONTE", "DIAS"])[["MAE", "RMSE", "MAPE", "SMAPE", "DTW"]]
              .mean().round(2).reset_index()
              .sort_values(["GRUPO", "PRACA", "DIAS"]))
    resumo["N_ORIGENS"] = (por_origem.groupby(["GRUPO", "PRACA", "HORIZONTE", "DIAS"])
                           .size().reindex(pd.MultiIndex.from_frame(resumo[["GRUPO", "PRACA", "HORIZONTE", "DIAS"]])).values)
    resumo.to_csv(PASTA_RESULTADOS / "arima_resumo.csv", index=False, encoding="utf-8-sig")

    print("\n=== RESUMO (média das origens de teste) ===")
    print(resumo.to_string(index=False))
    print(f"\nResultados salvos em: {PASTA_RESULTADOS}")
