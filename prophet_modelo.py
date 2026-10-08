"""
Prophet (Taylor e Letham, 2018): modelo aditivo de tendência + sazonalidades +
feriados.

    volume(t) = tendência(t) + sazonalidade semanal(t) + sazonalidade anual(t)
                + efeito de feriados(t) + erro

Diferença em relação ao ARIMA: o ARIMA prevê o amanhã a partir dos dias
anteriores; o Prophet descreve o volume como uma curva que depende da DATA
(qual dia da semana, qual época do ano, se é feriado), e por isso consegue
antecipar feriados e a sazonalidade do ano, que o ARIMA não enxerga.

Desenho igual ao do ARIMA (config.py): série de 2021 em diante, 70% iniciais
para treino, horizontes de 7, 90 e 180 dias, mesmas datas de partida, mesmas
métricas.

Uma diferença necessária na avaliação: o Prophet não "lê" os últimos dias
observados ao prever -- a previsão depende só da data. Por isso, para usar a
informação disponível até cada data de partida (como o ARIMA faz), o modelo é
reajustado em cada data de partida, só com os dados até ela. É o mesmo
procedimento da validação cruzada da própria biblioteca. Nenhum dado posterior
à data de partida é usado.

Feriados: nacionais e estaduais de SP (biblioteca holidays), mais Carnaval
(segunda e terça) e Corpus Christi, que são pontos facultativos mas afetam
muito o tráfego. O efeito de cada feriado é estimado também nos 2 dias
anteriores (saída para viagem) e no dia seguinte (retorno).

Entrada:  dados/processados/series_diarias.csv
Saída:    resultados/prophet_metricas_por_origem.csv
          resultados/prophet_resumo.csv
          resultados/prophet_<PRACA>.png              (previsão)
          resultados/prophet_componentes_<PRACA>.png  (o que o modelo aprendeu)

Como executar:
    python prophet_modelo.py                  (todas as praças, cerca de 5 a 15 minutos)
    python prophet_modelo.py "PORTO FELIZ"    (só uma praça, para teste; salva em
                                               arquivos *_teste_*, sem apagar os
                                               resultados completos)
"""

import logging
import sys
import warnings
from datetime import timedelta
from pathlib import Path

# Silencia mensagens internas do Prophet e do otimizador (duas linhas por ajuste,
# mais de mil no total, e um aviso inofensivo sobre gráficos interativos), para
# que só os avisos deste script apareçam na tela. Precisa vir antes de importar
# o Prophet.
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("prophet.plot").disabled = True

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.easter import easter
from prophet import Prophet

import config
import metricas

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent
ARQUIVO_SERIES = RAIZ / "dados" / "processados" / "series_diarias.csv"
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)

COLUNAS_METRICAS = ["MAE", "RMSE", "MAPE", "SMAPE", "DTW"]

# Janela de efeito de cada feriado: 2 dias antes e 1 depois.
DIAS_ANTES_FERIADO = 2
DIAS_DEPOIS_FERIADO = 1


def tabela_feriados(ano_inicio: int, ano_fim: int) -> pd.DataFrame:
    anos = range(ano_inicio, ano_fim + 1)
    linhas = [{"holiday": nome, "ds": pd.Timestamp(data)}
              for data, nome in holidays.Brazil(subdiv="SP", years=anos).items()]
    for ano in anos:
        pascoa = easter(ano)
        linhas.append({"holiday": "Carnaval", "ds": pd.Timestamp(pascoa - timedelta(days=48))})
        linhas.append({"holiday": "Carnaval", "ds": pd.Timestamp(pascoa - timedelta(days=47))})
        linhas.append({"holiday": "Corpus Christi", "ds": pd.Timestamp(pascoa + timedelta(days=60))})
    feriados = pd.DataFrame(linhas)
    feriados["lower_window"] = -DIAS_ANTES_FERIADO
    feriados["upper_window"] = DIAS_DEPOIS_FERIADO
    return feriados


FERIADOS = tabela_feriados(2020, 2027)


def carregar_serie(series: pd.DataFrame, praca: str) -> pd.Series:
    sub = series[(series["PRACA"] == praca) & (series["DATA"] >= config.DATA_INICIO_MODELAGEM)]
    return sub.set_index("DATA")["VOLUME_TOTAL"].sort_index().asfreq("D")


def ajustar(y_ate_origem: pd.Series) -> Prophet:
    """Ajusta o Prophet só com os dias que têm dado (dias vazios são ignorados)."""
    dados = y_ate_origem.dropna().reset_index()
    dados.columns = ["ds", "y"]
    modelo = Prophet(
        growth="linear",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        holidays=FERIADOS,
        uncertainty_samples=0,  # não calcula intervalos de confiança (mais rápido)
    )
    modelo.fit(dados)
    return modelo


def prever(modelo: Prophet, inicio: pd.Timestamp, dias: int) -> pd.Series:
    datas = pd.DataFrame({"ds": pd.date_range(inicio, periods=dias, freq="D")})
    return modelo.predict(datas).set_index("ds")["yhat"]


def avaliar_praca(y: pd.Series, praca: str, grupo: str) -> list:
    n = len(y)
    n_treino = int(n * config.PROPORCAO_TREINO)
    horizonte_max = max(config.HORIZONTES.values())
    horizonte_min = min(config.HORIZONTES.values())
    origens = metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS, horizonte_min)

    print(f"\n=== {praca} ({grupo}) ===")
    print(f"Treino inicial: {y.index[0].date()} a {y.index[n_treino - 1].date()}; "
          f"{len(origens)} datas de partida no teste (o modelo é reajustado em cada uma)")

    linhas = []
    for k, origem in enumerate(origens, start=1):
        modelo = ajustar(y.iloc[: origem + 1])
        passos = min(horizonte_max, n - 1 - origem)
        previsao = prever(modelo, y.index[origem + 1], passos)

        if k == 1:
            salvar_graficos(y, n_treino, modelo, previsao, praca)

        for nome_h, h in config.HORIZONTES.items():
            if origem + h > n - 1:
                continue
            real = y.iloc[origem + 1: origem + 1 + h].values
            resultado = metricas.todas(real, previsao.iloc[:h].values, config.JANELA_DTW)
            resultado.update({
                "MODELO": "PROPHET", "GRUPO": grupo, "PRACA": praca,
                "HORIZONTE": nome_h, "DIAS": h, "ORIGEM": y.index[origem].date(),
            })
            linhas.append(resultado)

        if k % 20 == 0 or k == len(origens):
            print(f"  {k}/{len(origens)} datas de partida concluídas")
    return linhas


def salvar_graficos(y, n_treino, modelo, previsao, praca):
    nome = praca.replace(" ", "_")

    fig, ax = plt.subplots(figsize=(12, 4))
    janela = y.iloc[n_treino - 120: n_treino + len(previsao)]
    ax.plot(janela.index, janela.values, label="Real", linewidth=1)
    ax.plot(previsao.index, previsao.values, label="Prophet (previsão a partir do fim do treino)", linewidth=1)
    ax.axvline(y.index[n_treino - 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{praca} — Prophet, previsão de {len(previsao)} dias a partir do fim do treino")
    ax.set_ylabel("veículos/dia")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / f"prophet_{nome}.png", dpi=100)
    plt.close()

    # Componentes: tendência, efeito dos feriados, padrão semanal e anual.
    datas = pd.DataFrame({"ds": pd.date_range(y.index[0], previsao.index[-1], freq="D")})
    comp = modelo.predict(datas).set_index("ds")
    fig, axes = plt.subplots(4, 1, figsize=(12, 11))
    axes[0].plot(comp.index, comp["trend"]); axes[0].set_title("Tendência")
    axes[1].plot(comp.index, comp["holidays"]); axes[1].set_title("Efeito dos feriados")
    semana = comp["weekly"].groupby(comp.index.dayofweek).mean()
    axes[2].bar(["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"], semana.values)
    axes[2].set_title("Sazonalidade semanal (diferença em relação à média)")
    ano = comp["yearly"].groupby(comp.index.dayofyear).mean()
    axes[3].plot(ano.index, ano.values); axes[3].set_title("Sazonalidade anual (dia do ano)")
    for ax in axes:
        ax.set_ylabel("veículos/dia")
    fig.suptitle(f"{praca} — componentes aprendidos pelo Prophet (treino inicial)")
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / f"prophet_componentes_{nome}.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    series = pd.read_csv(ARQUIVO_SERIES, parse_dates=["DATA"])
    alvos = [(p, "NUCLEO") for p in config.PRACAS_NUCLEO] + \
            [(p, "UNIDIRECIONAL") for p in config.PRACAS_UNIDIRECIONAIS]
    if len(sys.argv) > 1:
        alvos = [(p, g) for p, g in alvos if p == sys.argv[1].upper()]
        if not alvos:
            sys.exit(f"Praça '{sys.argv[1]}' não encontrada. Opções: "
                     f"{config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS}")

    linhas = []
    for praca, grupo in alvos:
        linhas.extend(avaliar_praca(carregar_serie(series, praca), praca, grupo))

    sufixo = "" if len(sys.argv) == 1 else "_teste_" + sys.argv[1].upper().replace(" ", "_")
    por_origem = pd.DataFrame(linhas)
    # O arquivo de teste não termina em "_metricas_por_origem.csv", para que
    # comparar_modelos.py nunca o confunda com um resultado oficial.
    nome_por_origem = "prophet_metricas_por_origem.csv" if not sufixo else f"prophet{sufixo}_metricas.csv"
    por_origem.to_csv(PASTA_RESULTADOS / nome_por_origem, index=False, encoding="utf-8-sig")

    chaves = ["GRUPO", "PRACA", "HORIZONTE", "DIAS"]
    resumo = por_origem.groupby(chaves)[COLUNAS_METRICAS].mean().round(2)
    resumo["N_ORIGENS"] = por_origem.groupby(chaves).size()
    resumo = resumo.reset_index().sort_values(["GRUPO", "PRACA", "DIAS"])
    resumo.to_csv(PASTA_RESULTADOS / f"prophet_resumo{sufixo}.csv", index=False, encoding="utf-8-sig")

    print("\n=== RESUMO PROPHET (média das datas de partida) ===")
    print(resumo.to_string(index=False))
    print(f"\nResultados salvos em: {PASTA_RESULTADOS}")
