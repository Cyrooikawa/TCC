"""
Diagnóstico da escolha da ordem do SARIMA (achado de outubro de 2026).

Motivação: a previsão do SARIMA de Boituva (Figura 6 da monografia) vira uma
reta depois de poucas semanas. A investigação mostrou que a busca automática
(auto_arima, passo a passo, menor AIC) não encontrou o modelo de menor AIC do
próprio espaço de busca em algumas praças.

Etapas (podem ser rodadas separadamente):

  selecao       Refaz a busca do auto_arima, exatamente como arima_baseline.py,
                e compara com a ordem fixada em config.ORDENS_ARIMA.
  alternativas  Para Alambari, Boituva e Itu, ajusta o mesmo (p,d,q) com termo
                autorregressivo sazonal, (1,0,1,7) e (1,0,0,7), e compara AIC
                (só treino) e SMAPE médio no teste com a ordem fixada.
  escala        Refaz a busca com a série em milhares de veículos (mesma escala
                do ajuste final), para testar se a escala explica o problema.
  amplitude     Mede, em Boituva, quanto da oscilação semanal a previsão de 180
                dias preserva com a ordem fixada e com a alternativa (1,0,1,7).

Saída: resultados/diagnostico_ordem_sarima.json (acumula as etapas rodadas).

Como executar (depois de preparar_series.py):
    python diagnostico_ordem_sarima.py alternativas
    python diagnostico_ordem_sarima.py tudo          (demora: cerca de 10 a 15 minutos)
"""

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

import arima_baseline as ab
import config
import metricas

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "resultados" / "diagnostico_ordem_sarima.json"
ORD = ["ALAMBARI", "BLOQUEIO DE BOITUVA", "BOITUVA", "PORTO FELIZ", "ITU", "SOROCABA"]
ACHATAM = ["ALAMBARI", "BOITUVA", "ITU"]  # praças cuja ordem fixada só tem médias móveis sazonais


def serie_e_treino(series, praca):
    y = ab.carregar_serie(series, praca)
    return y, y.iloc[: int(len(y) * config.PROPORCAO_TREINO)]


def ajustar(treino, ordem, sazonal):
    tendencia = "c" if ordem[1] + sazonal[1] == 0 else "n"
    return SARIMAX(treino / 1000, order=ordem, seasonal_order=sazonal, trend=tendencia,
                   enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)


def smape_medio(ajuste, y):
    n = len(y); r = {7: [], 90: [], 180: []}
    for o in metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS, 7):
        prev = ajuste.apply(y.iloc[: o + 1] / 1000).forecast(min(180, n - 1 - o)) * 1000
        for h in r:
            if o + h <= n - 1:
                r[h].append(metricas.smape(y.iloc[o + 1: o + 1 + h].values, prev.iloc[:h].values))
    return {str(h): round(float(np.mean(v)), 2) for h, v in r.items()}


def etapa_selecao(series, out):
    out["selecao"] = {}
    for p in ORD:
        _, tr = serie_e_treino(series, p); t0 = time.time()
        o, so = ab.escolher_ordem(tr)
        fix = config.ORDENS_ARIMA[p]
        out["selecao"][p] = {"auto_arima_agora": [list(o), list(so)], "fixada_config": [list(fix[0]), list(fix[1])],
                             "igual": (tuple(o), tuple(so)) == (tuple(fix[0]), tuple(fix[1])), "segundos": round(time.time() - t0, 1)}
        print(f"  {p:20s} agora {o}{so} | fixada {fix[0]}{fix[1]} | {'igual' if out['selecao'][p]['igual'] else 'DIFERENTE'}")


def etapa_alternativas(series, out):
    out["alternativas"] = {}
    for p in ACHATAM:
        y, tr = serie_e_treino(series, p); o, so = config.ORDENS_ARIMA[p]
        out["alternativas"][p] = {}
        for nome, (oo, ss) in {"fixada": (o, so), "P=1,Q=1": (o, (1, 0, 1, 7)), "P=1,Q=0": (o, (1, 0, 0, 7))}.items():
            aj = ajustar(tr, oo, ss)
            out["alternativas"][p][nome] = {"ordem": [list(oo), list(ss)], "AIC": round(float(aj.aic), 1), "SMAPE_7_90_180": smape_medio(aj, y)}
            print(f"  {p:9s} {nome:8s} {oo}{ss}  AIC={aj.aic:9.1f}  SMAPE={out['alternativas'][p][nome]['SMAPE_7_90_180']}")


def etapa_escala(series, out):
    out["escala"] = {}
    for p in ACHATAM:
        y, tr = serie_e_treino(series, p)
        base = tr.interpolate(method="linear").bfill().ffill() / 1000
        m = pm.auto_arima(base, seasonal=True, m=config.PERIODO_SAZONAL, max_p=3, max_q=3, max_P=2, max_Q=2,
                          stepwise=True, suppress_warnings=True, error_action="ignore")
        aj = ajustar(tr, m.order, m.seasonal_order)
        out["escala"][p] = {"ordem_busca_em_milhares": [list(m.order), list(m.seasonal_order)], "AIC": round(float(aj.aic), 1),
                            "SMAPE_7_90_180": smape_medio(aj, y)}
        print(f"  {p:9s} {m.order}{m.seasonal_order}  AIC={aj.aic:9.1f}  SMAPE={out['escala'][p]['SMAPE_7_90_180']}")


def etapa_amplitude(series, out):
    y, tr = serie_e_treino(series, "BOITUVA"); o, so = config.ORDENS_ARIMA["BOITUVA"]
    res = {}
    for nome, ss in [("fixada", so), ("P=1,Q=1", (1, 0, 1, 7))]:
        f = (ajustar(tr, o, ss).forecast(180) * 1000).values
        res[nome] = [round(float(np.ptp(f[i * 7:(i + 1) * 7])), 0) for i in range(25)]
    out["amplitude_semanal_boituva"] = res
    print("  amplitude por semana (fixada):", res["fixada"][:8], "...")
    print("  amplitude por semana (P=1,Q=1):", res["P=1,Q=1"][:8], "...")


if __name__ == "__main__":
    etapas = {"selecao": etapa_selecao, "alternativas": etapa_alternativas, "escala": etapa_escala, "amplitude": etapa_amplitude}
    pedido = sys.argv[1] if len(sys.argv) > 1 else "alternativas"
    if pedido != "tudo" and pedido not in etapas:
        sys.exit(f"Etapa desconhecida. Use: {', '.join(etapas)} ou tudo")
    series = pd.read_csv(RAIZ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
    out = json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() else {}
    for nome, func in etapas.items():
        if pedido in (nome, "tudo"):
            print(f"=== {nome} ==="); func(series, out)
            SAIDA.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nResultados em: {SAIDA}")
