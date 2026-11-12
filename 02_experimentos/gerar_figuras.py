"""
Gera TODAS as figuras usadas nos capítulos 3 e 4 da monografia e os números da
análise exploratória (EDA) citados no texto, a partir dos arquivos do projeto.

Assim, cada figura e cada número do documento pode ser refeito por qualquer
pessoa que rode o pipeline, sem depender de gráficos feitos à mão.

Entradas (geradas pelos outros scripts):
    dados/processados/series_diarias.csv         (preparar_series.py)
    resultados/ingenuo_metricas_por_origem.csv   (modelo_ingenuo.py)
    resultados/arima_metricas_por_origem.csv     (arima_baseline.py)
    resultados/prophet_metricas_por_origem.csv   (prophet_modelo.py)

Saídas:
    figuras/*.png                      (as figuras do documento, já no tamanho de inserção)
    resultados/eda_*.csv               (números da seção 4.2.1 e do feriado de abril de 2025)

Correspondência com o documento:
    Figura 1  pipeline.png
    Figura 2  series.png
    Figura 3  perfil_semanal.png
    Figura 4  perfil_mensal.png
    Figura 5  smape_horizontes.png
    Figura 6  boituva_sarima.png
    Figura 7  prophet_componentes_BOITUVA.png
    Figura 8  boituva_prophet.png
    Figura 9  dtw_relativo.png

Como executar (depois de rodar todos os modelos; leva cerca de 1 minuto, porque
reajusta o SARIMA e o Prophet de Boituva para desenhar as previsões):
    python gerar_figuras.py
"""

import logging
import sys
import warnings
from pathlib import Path

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("prophet.plot").disabled = True
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config
import prophet_modelo as pm

RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "resultados"
SAIDA = RAIZ / "figuras"
SAIDA.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})
W = 6.3  # largura em polegadas (cerca de 16 cm, a largura útil da página)

# Ordem de apresentação das praças (a mesma das tabelas do documento).
ORD = ["ALAMBARI", "BLOQUEIO DE BOITUVA", "BOITUVA", "PORTO FELIZ", "ITU", "SOROCABA"]
assert set(ORD) == set(config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS), "praças do config.py mudaram: revisar ORD"
NM = {"ALAMBARI": "Alambari", "BLOQUEIO DE BOITUVA": "Bloqueio de Boituva", "BOITUVA": "Boituva",
      "PORTO FELIZ": "Porto Feliz", "ITU": "Itu", "SOROCABA": "Sorocaba"}
ROTULO_CURTO = {"ALAMBARI": "Alambari", "BLOQUEIO DE BOITUVA": "Bloq. Boituva", "BOITUVA": "Boituva",
                "PORTO FELIZ": "P. Feliz", "ITU": "Itu", "SOROCABA": "Sorocaba"}
GRUPO = {p: ("núcleo" if p in config.PRACAS_NUCLEO else "unidirecional") for p in ORD}
COR_MODELO = {"INGENUO": "#bdbdbd", "ARIMA": "#6baed6", "PROPHET": "#08519c"}
ROTULO_MODELO = {"INGENUO": "Ingênuo sazonal", "ARIMA": "SARIMA", "PROPHET": "Prophet"}
CORES_PRACA = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#8c564b"]


def carregar():
    series = pd.read_csv(RAIZ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
    faltam = [m for m in ("ingenuo", "arima", "prophet")
              if not (RESULTADOS / f"{m}_metricas_por_origem.csv").exists()]
    if faltam:
        sys.exit(f"Faltam resultados de: {faltam}. Rode antes modelo_ingenuo.py, arima_baseline.py e prophet_modelo.py.")
    d = pd.concat([pd.read_csv(RESULTADOS / f"{m}_metricas_por_origem.csv") for m in ("ingenuo", "arima", "prophet")])
    medias = d.groupby(["MODELO", "PRACA", "DIAS"])[["SMAPE", "DTW"]].mean()
    return series, medias


def salvar(fig, nome):
    fig.savefig(SAIDA / nome, dpi=220, bbox_inches="tight" if nome == "pipeline.png" else None)
    plt.close(fig)
    print("  figura gravada:", nome)


# ------------------------------------------------------------------ Figura 2: séries
def fig_series(series):
    fig, axes = plt.subplots(3, 2, figsize=(W, 5.7), sharex=True)
    for ax, p in zip(axes.flatten(), ORD):
        x = series[series.PRACA == p].sort_values("DATA")
        ax.plot(x.DATA, x.VOLUME_TOTAL / 1000, lw=0.4, color="#1f4e79")
        for status, cor in [("AUSENTE", "crimson"), ("INVALIDADO", "darkorange")]:
            for dia in x[x.STATUS == status].DATA:
                ax.axvspan(dia, dia + pd.Timedelta(days=1), color=cor, alpha=0.35, lw=0)
        ax.axvline(pd.Timestamp(config.DATA_INICIO_MODELAGEM), color="gray", ls="--", lw=0.7)
        n = len(x[x.DATA >= config.DATA_INICIO_MODELAGEM]); fim_treino = x[x.DATA >= config.DATA_INICIO_MODELAGEM].DATA.iloc[int(n * config.PROPORCAO_TREINO) - 1]
        ax.axvline(fim_treino, color="black", ls=":", lw=0.7)
        ax.set_title(f"{NM[p]} ({GRUPO[p]})", fontsize=8, loc="left"); ax.tick_params(labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("mil veículos/dia", fontsize=7)
    plt.tight_layout(); salvar(fig, "series.png")


# ------------------------------------------------------------------ Figuras 3 e 4: perfis
def perfis(series):
    ok = series[(series.STATUS == "OK") & (series.DATA >= config.DATA_INICIO_MODELAGEM)].copy()
    ok["dow"] = ok.DATA.dt.dayofweek
    semanal = ok.groupby(["PRACA", "dow"]).VOLUME_TOTAL.mean().unstack()
    semanal = semanal.div(semanal.mean(axis=1), axis=0)
    c = ok[ok.DATA.dt.year.between(2021, 2025)].copy(); c["ano"] = c.DATA.dt.year; c["mes"] = c.DATA.dt.month
    a = c.groupby(["PRACA", "ano", "mes"]).VOLUME_TOTAL.mean().reset_index()
    a["idx"] = a.VOLUME_TOTAL / a.groupby(["PRACA", "ano"]).VOLUME_TOTAL.transform("mean")
    mensal = a.groupby(["PRACA", "mes"]).idx.mean().unstack()

    def desenhar(nome, dados, xs, rotulos, ylabel):
        fig, ax = plt.subplots(figsize=(W, 2.9))
        for p, cor in zip(ORD, CORES_PRACA):
            ax.plot(xs, dados.loc[p].values, marker="o", ms=2.8, lw=1.1, ls="-" if GRUPO[p] == "núcleo" else "--",
                    color=cor, label=NM[p])
        ax.set_xticks(list(xs)); ax.set_xticklabels(rotulos, fontsize=7.5); ax.axhline(1, color="gray", lw=0.6)
        ax.grid(alpha=0.25); ax.set_ylabel(ylabel); ax.legend(fontsize=6.8, ncol=3, loc="upper left", framealpha=0.9)
        plt.tight_layout(); salvar(fig, nome)

    desenhar("perfil_semanal.png", semanal, range(7), ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"], "Volume / média semanal")
    desenhar("perfil_mensal.png", mensal, range(1, 13),
             ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"], "Volume / média anual")
    semanal.loc[ORD].round(3).to_csv(RESULTADOS / "eda_perfil_semanal.csv", encoding="utf-8-sig")
    mensal.loc[ORD].round(3).to_csv(RESULTADOS / "eda_perfil_mensal.csv", encoding="utf-8-sig")


# ------------------------------------------------------------------ Figuras 5 e 9: SMAPE e DTW por horizonte
def horizontes(medias):
    def tres(nome, relativo):
        fig, axes = plt.subplots(1, 3, figsize=(W, 2.9), sharey=True)
        for ax, h in zip(axes, config.HORIZONTES.values()):
            xs = np.arange(len(ORD))
            if not relativo:
                larg = 0.27
                for i, m in enumerate(["INGENUO", "ARIMA", "PROPHET"]):
                    ax.bar(xs + (i - 1) * larg, [medias.loc[(m, p, h), "SMAPE"] for p in ORD], larg, color=COR_MODELO[m], label=ROTULO_MODELO[m])
            else:
                larg = 0.36
                for i, m in enumerate(["ARIMA", "PROPHET"]):
                    ax.bar(xs + (i - 0.5) * larg, [medias.loc[(m, p, h), "DTW"] / medias.loc[("INGENUO", p, h), "DTW"] for p in ORD],
                           larg, color=COR_MODELO[m], label=ROTULO_MODELO[m])
                ax.axhline(1, color="black", lw=0.9)
            ax.set_xticks(xs); ax.set_xticklabels([ROTULO_CURTO[p] for p in ORD], rotation=45, ha="right", fontsize=6.8)
            ax.axvline(len(config.PRACAS_NUCLEO) - 0.5, color="gray", ls="--", lw=0.7)
            ax.set_title(f"{h} dias", fontsize=8.5); ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("DTW relativo ao ingênuo" if relativo else "SMAPE médio (%)")
        axes[0].legend(loc="upper left", fontsize=6.8)
        plt.tight_layout(); salvar(fig, nome)
    tres("smape_horizontes.png", False)
    tres("dtw_relativo.png", True)


# ------------------------------------------------------------------ Figura 1: pipeline
def pipeline():
    """A Figura 1 fica em figura_pipeline.py (versão com as redes neurais)."""
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parent / "figura_pipeline.py"))


# ------------------------------------------------------------------ Figuras 6, 7 e 8: Boituva
def boituva(series):
    y = series[(series.PRACA == "BOITUVA") & (series.DATA >= config.DATA_INICIO_MODELAGEM)].set_index("DATA")["VOLUME_TOTAL"].asfreq("D")
    n = len(y); nt = int(n * config.PROPORCAO_TREINO); treino = y.iloc[:nt]; h = max(config.HORIZONTES.values())
    ordem, sazonal = config.ORDENS_ARIMA["BOITUVA"]
    aj = SARIMAX(treino / 1000, order=ordem, seasonal_order=sazonal, trend="n",
                 enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
    prev_s = aj.forecast(h) * 1000; prev_s.index = pd.date_range(y.index[nt], periods=h, freq="D")
    modelo = pm.ajustar(treino); prev_p = pm.prever(modelo, y.index[nt], h)
    janela = y.iloc[nt - 120: nt + h]
    for nome, prev, rot in [("boituva_sarima.png", prev_s, "SARIMA"), ("boituva_prophet.png", prev_p, "Prophet")]:
        fig, ax = plt.subplots(figsize=(W, 2.7))
        ax.plot(janela.index, janela.values / 1000, lw=0.8, color="#1f77b4", label="Observado")
        ax.plot(prev.index, prev.values / 1000, lw=0.9, color="#ff7f0e", label=rot)
        ax.axvline(y.index[nt - 1], color="gray", ls="--", lw=0.7); ax.set_ylabel("mil veículos/dia")
        ax.set_ylim(18, 63); ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=7, ncol=2)
        plt.tight_layout(); salvar(fig, nome)
    datas = pd.DataFrame({"ds": pd.date_range(y.index[0], prev_p.index[-1], freq="D")})
    comp = modelo.predict(datas).set_index("ds")
    fig, axes = plt.subplots(4, 1, figsize=(W, 6.2))
    axes[0].plot(comp.index, comp["trend"] / 1000); axes[0].set_title("Tendência", fontsize=8.5, loc="left")
    axes[1].plot(comp.index, comp["holidays"] / 1000, lw=0.7); axes[1].set_title("Efeito dos feriados", fontsize=8.5, loc="left")
    sem = comp["weekly"].groupby(comp.index.dayofweek).mean() / 1000
    axes[2].bar(["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"], sem.values, color="#1f77b4")
    axes[2].set_title("Sazonalidade semanal (diferença em relação à média)", fontsize=8.5, loc="left")
    ano = comp["yearly"].groupby(comp.index.dayofyear).mean() / 1000
    axes[3].plot(ano.index, ano.values); axes[3].set_title("Sazonalidade anual (dia do ano)", fontsize=8.5, loc="left")
    for ax in axes:
        ax.set_ylabel("mil veíc./dia", fontsize=7.5); ax.grid(alpha=0.25)
    plt.tight_layout(); salvar(fig, "prophet_componentes_BOITUVA.png")
    return y, prev_p


# ------------------------------------------------------------------ números citados no texto (seção 4.2.1 e 4.3.1)
def numeros(series, y_boituva, prev_prophet):
    ok = series[series.STATUS == "OK"]
    m = ok[(ok.DATA >= config.DATA_INICIO_MODELAGEM)].copy(); m["ano"] = m.DATA.dt.year
    anual = m[m.ano.between(2021, 2025)].groupby(["PRACA", "ano"]).VOLUME_TOTAL.mean().unstack()
    cresc = ((anual[2025] / anual[2021]) ** 0.25 - 1) * 100
    mensal = ok.assign(mes=ok.DATA.dt.to_period("M")).groupby(["PRACA", "mes"]).VOLUME_TOTAL.mean().unstack(0)
    mediana = mensal["2021-01":"2025-12"].median()
    pandemia = mensal["2020-04":"2020-05"].mean() / mediana * 100
    resumo = pd.DataFrame({
        "media_diaria_2025": anual[2025].round(0), "crescimento_anual_pct": cresc.round(1),
        "dias_ausentes": series.groupby("PRACA").STATUS.apply(lambda s: int((s == "AUSENTE").sum())),
        "dias_invalidados": series.groupby("PRACA").STATUS.apply(lambda s: int((s == "INVALIDADO").sum())),
        "abr_mai_2020_pct_do_normal": pandemia.round(0)}).loc[ORD]
    resumo.to_csv(RESULTADOS / "eda_resumo_pracas.csv", encoding="utf-8-sig")
    feriado = pd.DataFrame({"observado": y_boituva.loc["2025-04-15":"2025-04-25"],
                            "prophet_previsto": prev_prophet.loc["2025-04-15":"2025-04-25"].round(0)})
    feriado["dia_semana"] = feriado.index.day_name()
    feriado.to_csv(RESULTADOS / "eda_feriado_abril_2025_boituva.csv", encoding="utf-8-sig")
    print("\nResumo por praça (seção 4.2.1):"); print(resumo.to_string())
    print("\nMediana diária de Boituva em 2025:", round(y_boituva["2025"].median()),
          "| maior valor observado:", int(y_boituva.max()), "em", y_boituva.idxmax().date())
    print("\nFeriado de abril de 2025 em Boituva (observado x Prophet):"); print(feriado.to_string())


if __name__ == "__main__":
    series, medias = carregar()
    print("Gerando figuras em", SAIDA)
    pipeline(); fig_series(series); perfis(series); horizontes(medias)
    y_b, prev_p = boituva(series)
    numeros(series, y_b, prev_p)
    print("\nConcluído.")
