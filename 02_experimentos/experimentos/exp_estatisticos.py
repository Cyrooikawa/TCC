"""
Experimentos de refinamento dos modelos estatísticos (seções 3.1 a 3.3 do Guia dos modelos).

Cada variante muda UMA coisa em relação ao modelo oficial (as variantes "combinado"
juntam mudanças que já foram testadas sozinhas, e estão marcadas como tal).
A avaliação é a oficial: mesmas praças, origens, horizontes e métricas.

Modos:
    teste       mede no período de teste (as 87 / 75 / 62 origens oficiais)
    validacao   mede em origens semanais dos últimos 20% do TREINO, com o modelo
                conhecendo só os dados anteriores a elas. É o critério para decidir
                sem olhar o teste (item 3 da seção 6 do guia). Só para o Prophet e o
                ingênuo; no SARIMA o critério é o AIC do treino.

Uso:
    python exp_estatisticos.py ingenuo
    python exp_estatisticos.py sarima
    python exp_estatisticos.py prophet [validacao|teste] [variante ...]

Saídas: experimentos/resultados/<modelo>__<variante>.csv            (teste, por origem)
        experimentos/resultados/validacao/<modelo>__<variante>.csv  (validação, por origem)
        experimentos/resultados/sarima_aic.csv
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
import sys
import warnings
from datetime import timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

import comum
from comum import config

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("prophet.plot").disabled = True

VALID = comum.SAIDA / "validacao"
VALID.mkdir(exist_ok=True)
PROCESSOS = 2


def origens_e_limite(y, modo):
    n = len(y)
    if modo == "teste":
        return comum.origens_teste(n), n - 1
    return comum.origens_validacao(n), comum.divisao(n)[0] - 1


# ---------------------------------------------------------------------------
# 3.1 Ingênuo sazonal
# ---------------------------------------------------------------------------

def prever_ingenuo(y, origem, variante):
    """Previsão de H_MAX dias a partir da origem."""
    conhecido = y.iloc[: origem + 1].ffill()
    H = comum.H_MAX
    if variante == "oficial":
        base = conhecido.iloc[-7:].values
    elif variante == "media4":
        # média das últimas 4 semanas, dia da semana a dia da semana
        base = conhecido.iloc[-28:].values.reshape(4, 7).mean(axis=0)
    elif variante == "ano_anterior":
        # mesma semana do ano anterior (52 semanas antes do dia previsto, mantendo o dia da semana),
        # corrigida pelo crescimento: razão entre as últimas 4 semanas e as mesmas 4 semanas um ano antes
        v = conhecido.values
        fator = np.nanmean(v[-28:]) / np.nanmean(v[-28 - 364:-364])
        prev = np.array([v[len(v) - 1 + k - 364] for k in range(1, H + 1)]) * fator
        return prev
    return np.tile(base, H // 7 + 1)[:H]


def rodar_ingenuo():
    series = comum.carregar_series()
    for modo, pasta in (("teste", comum.SAIDA), ("validacao", VALID)):
        for variante in ("oficial", "media4", "ano_anterior"):
            linhas = []
            for praca, grupo in comum.ALVOS:
                y = series[praca]
                origens, limite = origens_e_limite(y, modo)
                prev = {o: prever_ingenuo(y, o, variante) for o in origens}
                linhas += comum.avaliar(y, praca, grupo, "INGENUO", prev, limite)
            d = comum.salvar(linhas, f"ingenuo__{variante}", pasta)
            print(modo, variante, d.groupby("DIAS").SMAPE.mean().round(2).to_dict())


# ---------------------------------------------------------------------------
# 3.2 SARIMA
# ---------------------------------------------------------------------------

ESCALA = 1000


def ordens_menor_aic():
    t = pd.read_csv(comum.PROJ / "diagnostico" / "sarima_busca_completa_ordens.csv", encoding="utf-8-sig")
    saida = {}
    for _, r in t.iterrows():
        a, b = r["ORDEM_MENOR_AIC_AJUSTE_FINAL"].rstrip("7").split(")(")
        o = tuple(int(x) for x in a.strip("()").split(","))
        s = tuple(int(x) for x in b.strip("()").split(",")) + (7,)
        saida[r["PRACA"]] = (o, s)
    return saida


def exogenas(indice, feriados=False, fourier=0):
    cols = {}
    if feriados:
        fer = comum.datas_feriado()
        d = [x.date() for x in indice]
        cols["feriado"] = [float(x in fer) for x in d]
        cols["vespera"] = [float((x + timedelta(days=1)) in fer) for x in d]
        cols["dia_seguinte"] = [float((x - timedelta(days=1)) in fer) for x in d]
    t = np.arange(len(indice))
    # fase contada em dias corridos desde 01/01/2021 (determinística, conhecida no futuro)
    for k in range(1, fourier + 1):
        cols[f"sen{k}"] = np.sin(2 * np.pi * k * t / 365.25)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / 365.25)
    return pd.DataFrame(cols, index=indice) if cols else None


VARIANTES_SARIMA = {
    "oficial":          {},
    "menor_aic":        {"ordem": "menor_aic"},
    "feriados":         {"feriados": True},
    "deriva":           {"deriva": True},
    "fourier1":         {"fourier": 1},
    "fourier2":         {"fourier": 2},
    "fourier3":         {"fourier": 3},
    "combinado_menor_aic_feriados": {"ordem": "menor_aic", "feriados": True},
    "combinado_menor_aic_feriados_fourier2": {"ordem": "menor_aic", "feriados": True, "fourier": 2},
}


def _sarima_praca(args):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    variante, praca, grupo = args
    opc = VARIANTES_SARIMA[variante]
    y = comum.carregar_series()[praca]
    n = len(y)
    n_treino = comum.divisao(n)[0]
    ordem, sazonal = (ordens_menor_aic() if opc.get("ordem") == "menor_aic" else config.ORDENS_ARIMA)[praca]
    X = exogenas(y.index, opc.get("feriados", False), opc.get("fourier", 0))
    # No SARIMAX do statsmodels, com d = 1, a constante ("c") é a deriva: crescimento médio por dia.
    tendencia = "c" if (opc.get("deriva") or ordem[1] + sazonal[1] == 0) else "n"
    ajuste = SARIMAX(y.iloc[:n_treino] / ESCALA, exog=None if X is None else X.iloc[:n_treino],
                     order=ordem, seasonal_order=sazonal, trend=tendencia,
                     enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
    prev = {}
    for o in comum.origens_teste(n):
        passos = min(comum.H_MAX, n - 1 - o)
        if X is None:
            p = ajuste.apply(y.iloc[: o + 1] / ESCALA).forecast(steps=passos)
        else:
            p = ajuste.apply(y.iloc[: o + 1] / ESCALA, exog=X.iloc[: o + 1]).forecast(
                steps=passos, exog=X.iloc[o + 1: o + 1 + passos])
        prev[o] = p.values * ESCALA
    linhas = comum.avaliar(y, praca, grupo, "ARIMA", prev)
    info = {"VARIANTE": variante, "PRACA": praca, "ORDEM": f"{ordem}x{sazonal}", "TENDENCIA": tendencia,
            "AIC_TREINO": round(ajuste.aic, 1), "BIC_TREINO": round(ajuste.bic, 1),
            "CONVERGIU": bool(ajuste.mle_retvals.get("converged", False)),
            "N_PARAMETROS": len(ajuste.params)}
    for nome in (X.columns if X is not None else []):
        info[f"COEF_{nome}"] = round(float(ajuste.params[nome]), 3)
    if tendencia == "c" and ordem[1] == 1:
        info["DERIVA_VEIC_DIA"] = round(float(ajuste.params["intercept"]) * ESCALA, 2)
    return linhas, info


def rodar_sarima(variantes):
    infos = []
    with Pool(PROCESSOS) as pool:
        for variante in variantes:
            res = pool.map(_sarima_praca, [(variante, p, g) for p, g in comum.ALVOS])
            d = comum.salvar([l for r in res for l in r[0]], f"sarima__{variante}")
            infos += [r[1] for r in res]
            aic = sum(r[1]["AIC_TREINO"] for r in res)
            print(variante, d.groupby("DIAS").SMAPE.mean().round(2).to_dict(), f"AIC somado {aic:.0f}",
                  "convergiu:", all(r[1]["CONVERGIU"] for r in res), flush=True)
    novo = pd.DataFrame(infos)
    arq = comum.SAIDA / "sarima_aic.csv"
    if arq.exists():
        antigo = pd.read_csv(arq, encoding="utf-8-sig")
        novo = pd.concat([antigo[~antigo.VARIANTE.isin(novo.VARIANTE)], novo], ignore_index=True)
    novo.to_csv(arq, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 3.3 Prophet
# ---------------------------------------------------------------------------

# Feriados "longos" (viagem de vários dias): janela própria, com mais dias antes e depois.
# lower_window negativo = dias antes; upper_window = dias depois.
JANELAS_LONGAS = {
    "Sexta-feira Santa":          (-2, 3),   # quarta a segunda
    "Natal":                      (-4, 2),
    "Confraternização Universal": (-4, 3),
    "Carnaval":                   (-4, 2),   # sexta/sábado anteriores até quinta
}

VARIANTES_PROPHET = {
    "oficial":           {},
    "janelas_longas":    {"janelas_longas": True},
    "multiplicativo":    {"prophet": {"seasonality_mode": "multiplicative"}},
    "ferias_escolares":  {"ferias": True},
    # grade de changepoint_prior_scale (padrão 0,05) e holidays_prior_scale (padrão 10)
    "cps0.005":          {"prophet": {"changepoint_prior_scale": 0.005}},
    "cps0.5":            {"prophet": {"changepoint_prior_scale": 0.5}},
    "hps1":              {"prophet": {"holidays_prior_scale": 1.0}},
    "hps0.1":            {"prophet": {"holidays_prior_scale": 0.1}},
    "combinado_longas_mult": {"janelas_longas": True, "prophet": {"seasonality_mode": "multiplicative"}},
}


def tabela_feriados_prophet(janelas_longas):
    import prophet_modelo as pm
    f = pm.tabela_feriados(2020, 2027)
    if janelas_longas:
        for nome, (antes, depois) in JANELAS_LONGAS.items():
            assert (f.holiday == nome).any(), nome
            f.loc[f.holiday == nome, "lower_window"] = antes
            f.loc[f.holiday == nome, "upper_window"] = depois
    return f


def ferias_escolares(datas):
    """1 em julho, de 15/12 a 31/12 e em janeiro (calendário escolar típico de SP, conhecido de antemão)."""
    d = pd.DatetimeIndex(datas)
    return ((d.month == 7) | (d.month == 1) | ((d.month == 12) & (d.day >= 15))).astype(float)


def _prophet_praca(args):
    from prophet import Prophet
    variante, praca, grupo, modo = args
    opc = VARIANTES_PROPHET[variante]
    y = comum.carregar_series()[praca]
    origens, limite = origens_e_limite(y, modo)
    feriados = tabela_feriados_prophet(opc.get("janelas_longas", False))
    prev = {}
    for o in origens:
        dados = y.iloc[: o + 1].dropna().reset_index()
        dados.columns = ["ds", "y"]
        m = Prophet(growth="linear", weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    holidays=feriados, uncertainty_samples=0, **opc.get("prophet", {}))
        if opc.get("ferias"):
            m.add_regressor("ferias")
            dados["ferias"] = ferias_escolares(dados.ds)
        m.fit(dados)
        passos = min(comum.H_MAX, limite - o)
        fut = pd.DataFrame({"ds": pd.date_range(y.index[o + 1], periods=passos, freq="D")})
        if opc.get("ferias"):
            fut["ferias"] = ferias_escolares(fut.ds)
        prev[o] = m.predict(fut)["yhat"].values
    return comum.avaliar(y, praca, grupo, "PROPHET", prev, limite)


def rodar_prophet(modo, variantes):
    pasta = comum.SAIDA if modo == "teste" else VALID
    with Pool(PROCESSOS) as pool:
        for variante in variantes:
            res = pool.map(_prophet_praca, [(variante, p, g, modo) for p, g in comum.ALVOS], chunksize=1)
            d = comum.salvar([l for r in res for l in r], f"prophet__{variante}", pasta)
            print(modo, variante, d.groupby("DIAS").SMAPE.mean().round(3).to_dict(),
                  d.groupby("DIAS").size().to_dict(), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--processos" in a:
        i = a.index("--processos")
        PROCESSOS = int(a[i + 1])
        del a[i:i + 2]
    if a[0] == "ingenuo":
        rodar_ingenuo()
    elif a[0] == "sarima":
        rodar_sarima(a[1:] or list(VARIANTES_SARIMA))
    elif a[0] == "prophet":
        modo = a[1] if len(a) > 1 else "teste"
        rodar_prophet(modo, a[2:] or list(VARIANTES_PROPHET))
