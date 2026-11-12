"""
Teste de mesa: a previsão de 7 dias dos oito modelos em Boituva, dia a dia, em três
semanas do teste. As três semanas mostram onde está o erro de cada modelo:
  - semana comum (origem 16/09/2025);
  - semana com feriado (origem 15/04/2025: Sexta-feira Santa e Tiradentes);
  - semana seguinte ao feriado (origem 22/04/2025).

Cada célula traz a previsão (mil veículos) e o erro do dia no SMAPE,
|real − previsto| ÷ ((real + previsto) ÷ 2). Nas redes, a previsão mostrada é a média das
cinco sementes, e o erro do dia é a média dos erros das cinco sementes, como na avaliação
oficial. A última linha é a média dos sete dias, que é o SMAPE da janela, e o script
confere esse valor com os arquivos *_metricas_por_origem.csv (no Prophet, com tolerância
de 0,1 ponto, porque o ajuste varia um pouco entre versões das bibliotecas).

Como executar (cerca de 30 segundos):
    python teste_mesa_semanas_boituva.py
Saída: figuras/teste_mesa_semana_<tipo>.png
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402
import prophet_modelo as pm  # noqa: E402

PRACA = "BOITUVA"
SEMANAS = [("comum", "2025-09-16", "Semana comum, sem feriado"),
           ("feriado", "2025-04-15", "Semana com feriado: Sexta-feira Santa (18/04) e Tiradentes (21/04)"),
           ("pos_feriado", "2025-04-22", "Semana seguinte ao feriado")]
MODELOS = [("INGENUO", "Ingênuo"), ("ARIMA", "SARIMA"), ("PROPHET", "Prophet"), ("LSTM", "LSTM"), ("GRU", "GRU"),
           ("BILSTM", "Bi-LSTM"), ("BIGRU", "Bi-GRU"), ("NHITS", "N-HiTS")]
DIAS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
br = lambda v, nd=1: f"{v:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")  # noqa: E731

series = pd.read_csv(RAIZ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
y = pm.carregar_serie(series, PRACA)
n_treino = int(len(y) * config.PROPORCAO_TREINO)
ordem, saz = config.ORDENS_ARIMA[PRACA]
ajuste = SARIMAX(y.iloc[:n_treino] / 1000, order=ordem, seasonal_order=saz, trend="n",
                 enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
oficiais = pd.concat([pd.read_csv(f) for f in (RAIZ / "resultados").glob("*_metricas_por_origem.csv")])
oficiais = oficiais[(oficiais.PRACA == PRACA) & (oficiais.DIAS == 7)]


def cor(erro):
    """Verde até 3%, amarelo até 8%, laranja até 15%, vermelho acima."""
    return "#d8f0e3" if erro <= 3 else "#fbf0c8" if erro <= 8 else "#fbdcc6" if erro <= 15 else "#f6c6c6"


for chave, data_origem, titulo in SEMANAS:
    origem = y.index.get_loc(pd.Timestamp(data_origem))
    datas = y.index[origem + 1: origem + 8]
    real = y.iloc[origem + 1: origem + 8].values
    ultima = y.iloc[: origem + 1].ffill().iloc[-7:]
    prev, sementes = {"INGENUO": ultima.values}, {}
    prev["ARIMA"] = ajuste.apply(y.iloc[: origem + 1] / 1000).forecast(7).values * 1000
    prev["PROPHET"] = pm.prever(pm.ajustar(y.iloc[: origem + 1]), datas[0], 7).values
    for m in ["LSTM", "GRU", "BILSTM", "BIGRU", "NHITS"]:
        z = np.load(RAIZ / "resultados" / "redes" / f"{m}_7_previsoes.npz")
        i = [k for k in range(len(z["praca"])) if str(z["praca"][k]) == PRACA and str(z["origem"][k]) == data_origem][0]
        sementes[m] = z["previsao"][:, i, :].astype(float)
        prev[m] = sementes[m].mean(axis=0)

    erro_dia = lambda p: np.abs(real - p) / ((np.abs(real) + np.abs(p)) / 2) * 100  # noqa: E731
    erros = {m: erro_dia(p) for m, p in prev.items()}
    for m, s in sementes.items():  # nas redes, a métrica é a média do erro de cada semente
        erros[m] = np.mean([erro_dia(si) for si in s], axis=0)
    for m, _ in MODELOS:  # confere com a avaliação oficial
        ofi = oficiais[(oficiais.MODELO == m) & (oficiais.ORIGEM == data_origem)].SMAPE.iloc[0]
        tol = 0.1 if m == "PROPHET" else 0.01  # Prophet: o otimizador varia um pouco entre versões
        assert abs(erros[m].mean() - ofi) < tol, (m, erros[m].mean(), ofi)

    ncol = 2 + len(MODELOS)
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.set_xlim(0, ncol)
    ax.set_ylim(-0.9, 13.7)
    ax.axis("off")
    larg = [1.15, 0.95] + [1.0] * len(MODELOS)
    x0 = np.concatenate([[0], np.cumsum(larg)])
    escala = ncol / x0[-1]
    x0 = x0 * escala
    larg = np.array(larg) * escala
    ax.text(0, 13.35, f"Boituva · {titulo}", fontsize=12, fontweight="bold", va="center")
    ax.text(0, 12.3, f"Origem: terça-feira {pd.Timestamp(data_origem):%d/%m/%Y}. O ingênuo repete a semana de "
            f"{ultima.index[0]:%d/%m} a {ultima.index[-1]:%d/%m}. Valores em mil veículos/dia; abaixo, o erro do dia "
            f"= |real − previsto| ÷ média(real, previsto).\nRedes: previsão média das cinco sementes; o erro é a média dos erros "
            f"das cinco sementes, como na avaliação oficial.", fontsize=8.2, color="#3b3a37", va="center")

    def celula(c, linha, texto, sub=None, fundo="#ffffff", negrito=False, cor_txt="#111"):
        yb = 10.4 - linha * 1.18
        ax.add_patch(plt.Rectangle((x0[c] + 0.02, yb), larg[c] - 0.04, 1.1, fc=fundo, ec="#d6d5d0", lw=0.6))
        if sub is None:
            ax.text(x0[c] + larg[c] / 2, yb + 0.55, texto, ha="center", va="center", fontsize=9,
                    fontweight="bold" if negrito else "normal", color=cor_txt)
        else:
            ax.text(x0[c] + larg[c] / 2, yb + 0.73, texto, ha="center", va="center", fontsize=9.2,
                    fontweight="bold" if negrito else "normal", color=cor_txt)
            ax.text(x0[c] + larg[c] / 2, yb + 0.32, sub, ha="center", va="center", fontsize=7.6, color="#3b3a37")

    celula(0, 0, "Dia", fundo="#e6e5e0", negrito=True)
    celula(1, 0, "Real", fundo="#1d5fae", negrito=True, cor_txt="white")
    for j, (m, rot) in enumerate(MODELOS):
        fam = "#eb6834" if m in ("INGENUO", "ARIMA", "PROPHET") else "#1b8f63"
        celula(2 + j, 0, rot, fundo=fam, negrito=True, cor_txt="white")
    for i, d in enumerate(datas):
        celula(0, i + 1, f"{DIAS[d.dayofweek]} {d:%d/%m}", fundo="#f4f3ef")
        celula(1, i + 1, br(real[i] / 1000), fundo="#e8f0fa", negrito=True)
        for j, (m, _) in enumerate(MODELOS):
            e = erros[m][i]
            celula(2 + j, i + 1, br(prev[m][i] / 1000), f"erro {br(e)}%", fundo=cor(e))
    celula(0, 8, "SMAPE", sub="média dos 7 erros", fundo="#e6e5e0", negrito=True)
    celula(1, 8, "", fundo="#f4f3ef")
    melhor = min(MODELOS, key=lambda mm: erros[mm[0]].mean())[0]
    for j, (m, _) in enumerate(MODELOS):
        s = erros[m].mean()
        celula(2 + j, 8, f"{br(s)}%", sub="o menor" if m == melhor else None, fundo=cor(s), negrito=True)
    leg = [("#d8f0e3", "erro até 3%"), ("#fbf0c8", "3% a 8%"), ("#fbdcc6", "8% a 15%"), ("#f6c6c6", "acima de 15%")]
    for k, (c, t) in enumerate(leg):
        ax.add_patch(plt.Rectangle((x0[2] + k * 1.7, -0.6), 0.35, 0.4, fc=c, ec="#b3b2ad", lw=0.5))
        ax.text(x0[2] + k * 1.7 + 0.45, -0.4, t, fontsize=8, va="center")
    ax.text(0, -0.4, "Laranja: estatísticos · verde: redes", fontsize=8, va="center", color="#3b3a37")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.04)
    fig.savefig(RAIZ / "figuras" / f"teste_mesa_semana_{chave}.png", dpi=200)
    plt.close(fig)
    print(f"{chave}: " + ", ".join(f"{rot} {erros[m].mean():.1f}" for m, rot in MODELOS))
