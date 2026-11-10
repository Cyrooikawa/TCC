"""
Figura da seção 4.3: previsão de 180 dias feita no fim do treino (26/11/2024) em
Boituva, pelos oito modelos, sobre o volume observado. Substitui a antiga Figura 6
(só o SARIMA) e a antiga Figura 8 (só o Prophet), para comparar todos os modelos.

Redes: previsão da rede de 180 dias na primeira origem, média das cinco sementes
(arquivo resultados/redes/<REDE>_180_previsoes.npz). Estatísticos: refeitos aqui,
com o mesmo procedimento dos scripts oficiais.

Como executar (segundos):
    python figura_boituva_todos_modelos.py
Saída: figuras/boituva_todos_modelos.png
"""

import logging
import sys
import warnings
from pathlib import Path

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.statespace.sarimax import SARIMAX

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402
import metricas  # noqa: E402
import prophet_modelo as pm  # noqa: E402

PRACA, H = "BOITUVA", 180
MODELOS = [("INGENUO", "Ingênuo sazonal", "#7a7a7a"), ("ARIMA", "SARIMA", "#eb6834"), ("PROPHET", "Prophet", "#2a78d6"),
           ("LSTM", "LSTM", "#1baf7a"), ("GRU", "GRU", "#6250d6"), ("BILSTM", "Bi-LSTM", "#1baf7a"),
           ("BIGRU", "Bi-GRU", "#6250d6"), ("NHITS", "N-HiTS", "#e87ba4")]

series = pd.read_csv(RAIZ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
y = pm.carregar_serie(series, PRACA)
n_treino = int(len(y) * config.PROPORCAO_TREINO)
origem = y.index[n_treino - 1]
datas = pd.date_range(origem + pd.Timedelta(days=1), periods=H, freq="D")
real = y.loc[datas].values

prev = {"INGENUO": np.tile(y.iloc[:n_treino].ffill().iloc[-7:].values, H // 7 + 1)[:H]}
ordem, saz = config.ORDENS_ARIMA[PRACA]
aj = SARIMAX(y.iloc[:n_treino] / 1000, order=ordem, seasonal_order=saz, trend="n",
             enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
prev["ARIMA"] = aj.forecast(H).values * 1000
prev["PROPHET"] = pm.prever(pm.ajustar(y.iloc[:n_treino]), datas[0], H).values
for m in ["LSTM", "GRU", "BILSTM", "BIGRU", "NHITS"]:
    z = np.load(RAIZ / "resultados" / "redes" / f"{m}_{H}_previsoes.npz")
    i = [k for k in range(len(z["praca"])) if str(z["praca"][k]) == PRACA and str(z["origem"][k]) == str(origem.date())][0]
    prev[m] = z["previsao"][:, i, :].astype(float).mean(axis=0)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
fig, axes = plt.subplots(4, 2, figsize=(6.3, 7.4), sharex=True, sharey=True)
inicio = origem - pd.Timedelta(days=27)
obs = y.loc[inicio: datas[-1]]
meses = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun", 7: "jul", 8: "ago", 9: "set", 10: "out",
         11: "nov", 12: "dez"}
for ax, (m, rotulo, cor) in zip(axes.ravel(), MODELOS):
    ax.plot(obs.index, obs.values / 1000, color="#b3b2ad", lw=0.7)
    ax.plot(datas, prev[m] / 1000, color=cor, lw=0.9, ls=(0, (1.2, 1.2)) if m.startswith("BI") else "-")
    ax.axvline(origem, color="#52514e", ls=":", lw=0.7)
    sm = metricas.smape(real, prev[m])
    ax.set_title(f"{rotulo}: SMAPE {sm:.1f}%".replace(".", ","), fontsize=8, loc="left")
    ax.set_ylim(20, 62)
    ax.grid(axis="y", color="#e4e3df", lw=0.5)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12, 2, 4]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{meses[mdates.num2date(v).month]}/{mdates.num2date(v).year % 100}"))
for ax in axes[:, 0]:
    ax.set_ylabel("Mil veículos/dia")
fig.text(0.5, 0.008, "Cinza: volume observado. Linha pontilhada vertical: fim do treino (26/11/2024).\n"
         "Redes: rede de 180 dias, média das cinco sementes.", ha="center", fontsize=6.6, color="#52514e")
fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.08, hspace=0.42, wspace=0.08)
fig.savefig(RAIZ / "figuras" / "boituva_todos_modelos.png", dpi=220)
print("figura gravada: figuras/boituva_todos_modelos.png")
for m, rotulo, _ in MODELOS:
    print(f"  {rotulo:16s} SMAPE {metricas.smape(real, prev[m]):5.2f}%  DTW {metricas.dtw(real, prev[m], 7):6.0f}")
