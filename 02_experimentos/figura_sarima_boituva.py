"""
Figura do relatório de 08/11/2026 (A2): previsão de 180 dias do SARIMA em Boituva,
com a ordem atual e com a ordem de menor AIC do ajuste final.

Como executar (segundos):
    python figura_sarima_boituva.py
Saída: diagnostico/sarima_boituva_ordem_atual_x_ajuste_final.png
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.statespace.sarimax import SARIMAX

raiz = Path(__file__).resolve().parent
saidas = [raiz / "diagnostico" / "sarima_boituva_ordem_atual_x_ajuste_final.png"]
sys.path.insert(0, str(raiz))
import arima_baseline as ab  # noqa: E402
import config  # noqa: E402

series = pd.read_csv(raiz / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
y = ab.carregar_serie(series, "BOITUVA")
n_treino = int(len(y) * config.PROPORCAO_TREINO)
treino = y.iloc[:n_treino]
variantes = [("Ordem atual (3,1,2)(0,0,2)₇: busca passo a passo", (3, 1, 2), (0, 0, 2, 7), "#eb6834", (0, (4, 2))),
             ("Menor AIC do ajuste final (3,1,2)(1,0,2)₇", (3, 1, 2), (1, 0, 2, 7), "#eb6834", "-")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(9.2, 3.3))
janela = y.iloc[n_treino - 56: n_treino + 180]
ax.plot(janela.index, janela.values / 1000, color="#52514e", lw=0.9, label="Volume observado")
for rotulo, o, s, cor, ls in variantes:
    r = SARIMAX(treino / 1000, order=o, seasonal_order=s, trend="n",
                enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
    prev = r.forecast(180)
    ax.plot(prev.index, prev.values, color=cor, lw=1.4 if ls == "-" else 1.1, ls=ls, label=rotulo)
ax.axvline(y.index[n_treino - 1], color="#9a9893", ls=":", lw=0.9)
ax.text(y.index[n_treino - 1], 61.5, " fim do treino (26/11/2024)", fontsize=7.5, color="#52514e", va="top")
ax.set_ylabel("Mil veículos por dia")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
meses = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun", 7: "jul", 8: "ago", 9: "set", 10: "out",
         11: "nov", 12: "dez"}
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{meses[mdates.num2date(v).month]}/{mdates.num2date(v).year % 100}"))
ax.set_ylim(20, 62)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.grid(axis="y", color="#e4e3df", lw=0.6)
ax.legend(loc="upper left", frameon=False, fontsize=7.6, ncol=3, bbox_to_anchor=(0, 1.13))
fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.11)
for s in saidas:
    s.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(s, dpi=200)
    print("salvo", s)
