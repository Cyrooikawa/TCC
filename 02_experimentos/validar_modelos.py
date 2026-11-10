"""
Validação do PROCESSO de cada modelo (complementa validar_procedimentos.py).

Para cada um dos oito modelos, refaz de forma independente uma parte do processo e
confere se ele foi executado como a metodologia descreve:

  Ingênuo sazonal  previsão = última semana observada até a origem; métricas refeitas
  SARIMA           ordem de config.py; ajuste só no treino; convergência; parâmetros
                   iguais em todas as origens; dados após a origem não mudam a previsão;
                   métricas refeitas em todas as origens
  Prophet          reajuste só com dados até a origem; dados após a origem não mudam a
                   previsão; calendário de feriados cobre o teste; métricas refeitas numa
                   amostra de origens
  Redes (5)        configuração declarada (janela, unidades, lote, épocas, paciência, taxa,
                   sementes); métrica de cada origem = média das 5 sementes; métricas
                   refeitas das previsões diárias; época escolhida dentro do treino
  Todos            mesmas praças, horizontes e origens (terças-feiras), 1.344 janelas

As diferenças toleradas para SARIMA e Prophet cobrem variações numéricas entre
computadores (otimizador e BLAS); as médias devem coincidir em centésimos.

Como executar (cerca de 2 minutos):
    python validar_modelos.py
Saída: resultados/validacao_modelos.csv
"""

import json
import logging
import sys
import warnings
from pathlib import Path

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402
import metricas  # noqa: E402

RES = RAIZ / "resultados"
REDES = ["LSTM", "GRU", "BILSTM", "BIGRU", "NHITS"]
LINHAS = []


def registrar(modelo, verificacao, ok, detalhe=""):
    LINHAS.append({"MODELO": modelo, "VERIFICACAO": verificacao, "RESULTADO": "OK" if ok else "FALHOU",
                   "DETALHE": detalhe})
    print(f"[{'OK     ' if ok else 'FALHOU '}] {modelo:8s} {verificacao}" + (f" — {detalhe}" if detalhe else ""))


def series():
    df = pd.read_csv(RAIZ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
    df = df[df.DATA >= config.DATA_INICIO_MODELAGEM]
    return {p: g.set_index("DATA")["VOLUME_TOTAL"].sort_index().asfreq("D")
            for p, g in df.groupby("PRACA") if p in config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS}


def todas_origens(n):
    return metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS, min(config.HORIZONTES.values()))


def comparar(oficial, novo, modelo, nome, tol_media, tol_max):
    m = novo.merge(oficial, on=["PRACA", "DIAS", "ORIGEM"], suffixes=("", "_OF"))
    dif = (m.SMAPE - m.SMAPE_OF).abs()
    medias = m.groupby("DIAS")[["SMAPE", "SMAPE_OF"]].mean()
    dmed = float((medias.SMAPE - medias.SMAPE_OF).abs().max())
    registrar(modelo, nome, len(m) == len(novo) and dmed <= tol_media and dif.max() <= tol_max,
              f"{len(m)} janelas; maior diferença por janela {dif.max():.3f} p.p.; nas médias {dmed:.3f} p.p.")


# ------------------------------------------------------------------ todos
def bloco_comum():
    print("\n== Todos os modelos: mesmas praças, horizontes e origens")
    chaves = None
    for arq in sorted(RES.glob("*_metricas_por_origem.csv")):
        d = pd.read_csv(arq)
        k = set(zip(d.PRACA, d.DIAS, d.ORIGEM))
        dias = {pd.Timestamp(o).weekday() for o in d.ORIGEM}
        registrar(d.MODELO.iloc[0], "1.344 janelas, origens às terças, nenhuma métrica vazia",
                  len(d) == 1344 and dias == {1} and not d[["MAE", "RMSE", "MAPE", "SMAPE", "DTW"]].isna().any().any())
        chaves = k if chaves is None else chaves
        if k != chaves:
            registrar(d.MODELO.iloc[0], "mesmas janelas que os demais modelos", False)


# ------------------------------------------------------------------ ingênuo
def bloco_ingenuo(ys):
    print("\n== Ingênuo sazonal")
    linhas = []
    for p, y in ys.items():
        for o in todas_origens(len(y)):
            semana = y.iloc[:o + 1].ffill().iloc[-7:].values
            assert len(semana) == 7
            for h in config.HORIZONTES.values():
                if o + h <= len(y) - 1:
                    prev = np.tile(semana, h // 7 + 1)[:h]
                    linhas.append({"PRACA": p, "DIAS": h, "ORIGEM": str(y.index[o].date()),
                                   "SMAPE": metricas.smape(y.iloc[o + 1:o + 1 + h].values, prev)})
    comparar(pd.read_csv(RES / "ingenuo_metricas_por_origem.csv"), pd.DataFrame(linhas), "INGENUO",
             "previsão = última semana observada; SMAPE refeito em todas as janelas", 1e-9, 1e-9)


# ------------------------------------------------------------------ SARIMA
def bloco_sarima(ys):
    print("\n== SARIMA")
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    of_modelos = pd.read_csv(RES / "arima_modelos.csv").set_index("PRACA")
    linhas = []
    for p, y in ys.items():
        n = len(y)
        n_treino = int(n * config.PROPORCAO_TREINO)
        ordem, saz = config.ORDENS_ARIMA[p]
        registrar("ARIMA", f"{p}: ordem usada = ordem de config.py",
                  of_modelos.loc[p, "ORDEM"] == str(ordem) and of_modelos.loc[p, "ORDEM_SAZONAL"] == str(saz),
                  f"{ordem}x{saz}")
        aj = SARIMAX(y.iloc[:n_treino] / 1000, order=ordem, seasonal_order=saz, trend="n",
                     enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
        registrar("ARIMA", f"{p}: ajuste só no treino (1.426 dias) e convergiu",
                  aj.nobs == n_treino and bool(aj.mle_retvals.get("converged")),
                  f"AIC {aj.aic:.1f} (arquivo {of_modelos.loc[p, 'AIC_EM_MILHARES']:.1f})")
        origens = todas_origens(n)
        params_iguais, sem_vazamento = True, True
        for k, o in enumerate(origens):
            r = aj.apply(y.iloc[:o + 1] / 1000)
            params_iguais &= np.allclose(r.params, aj.params)
            prev = r.forecast(min(180, n - 1 - o)) * 1000
            if k % 20 == 0:   # vazamento: mudar tudo o que vem depois da origem
                y2 = y.copy()
                y2.iloc[o + 1:] *= 50
                prev2 = aj.apply(y2.iloc[:o + 1] / 1000).forecast(min(180, n - 1 - o)) * 1000
                sem_vazamento &= np.allclose(prev.values, prev2.values)
            for h in config.HORIZONTES.values():
                if o + h <= n - 1:
                    linhas.append({"PRACA": p, "DIAS": h, "ORIGEM": str(y.index[o].date()),
                                   "SMAPE": metricas.smape(y.iloc[o + 1:o + 1 + h].values, prev.iloc[:h].values)})
        registrar("ARIMA", f"{p}: parâmetros do treino mantidos em todas as {len(origens)} origens", params_iguais)
        registrar("ARIMA", f"{p}: dados após a origem não mudam a previsão", sem_vazamento)
    comparar(pd.read_csv(RES / "arima_metricas_por_origem.csv"), pd.DataFrame(linhas), "ARIMA",
             "SMAPE refeito em todas as janelas", 0.05, 1.0)


# ------------------------------------------------------------------ Prophet
def bloco_prophet(ys):
    print("\n== Prophet")
    import prophet_modelo as pm
    fim_teste = max(y.index[-1] for y in ys.values())
    cobre = pm.FERIADOS.ds.max() >= fim_teste and pm.FERIADOS.ds.min() <= pd.Timestamp(config.DATA_INICIO_MODELAGEM)
    registrar("PROPHET", "calendário de feriados cobre treino e teste (feriados futuros conhecidos)", cobre,
              f"{pm.FERIADOS.ds.min().date()} a {pm.FERIADOS.ds.max().date()}, {len(pm.FERIADOS)} datas")
    linhas, sem_vazamento, n_treino_ok = [], True, True
    for p, y in ys.items():
        n = len(y)
        origens = todas_origens(n)
        amostra = origens[::22]                       # 4 origens por praça, espalhadas no teste
        for o in amostra:
            m = pm.ajustar(y.iloc[:o + 1])
            n_treino_ok &= m.history.ds.max() <= y.index[o]
            prev = pm.prever(m, y.index[o + 1], min(180, n - 1 - o))
            if o == amostra[0]:
                y2 = y.copy()
                y2.iloc[o + 1:] *= 50
                prev2 = pm.prever(pm.ajustar(y2.iloc[:o + 1]), y.index[o + 1], min(180, n - 1 - o))
                sem_vazamento &= np.allclose(prev.values, prev2.values)
            for h in config.HORIZONTES.values():
                if o + h <= n - 1:
                    linhas.append({"PRACA": p, "DIAS": h, "ORIGEM": str(y.index[o].date()),
                                   "SMAPE": metricas.smape(y.iloc[o + 1:o + 1 + h].values, prev.iloc[:h].values)})
    registrar("PROPHET", "cada reajuste usa só dados até a origem", n_treino_ok)
    registrar("PROPHET", "dados após a origem não mudam a previsão", sem_vazamento)
    comparar(pd.read_csv(RES / "prophet_metricas_por_origem.csv"), pd.DataFrame(linhas), "PROPHET",
             "SMAPE refeito numa amostra de 24 origens", 0.15, 1.0)


# ------------------------------------------------------------------ redes
def bloco_redes(ys):
    print("\n== Redes neurais")
    import redes_neurais as rn
    esperado_params = {("LSTM", 7): 17351, ("NHITS", 7): 866911}
    for m in REDES:
        for h in sorted(config.HORIZONTES.values()):
            r = json.loads((RES / "redes" / f"{m}_{h}_resumo.json").read_text(encoding="utf-8"))
            cfg_ok = (r["janela_entrada"] == rn.JANELA_ENTRADA[h] and r["sementes"] == rn.SEMENTES
                      and r["max_epocas"] == rn.MAX_EPOCAS and r["paciencia"] == rn.PACIENCIA
                      and r["taxa_aprendizado"] == rn.TAXA_APRENDIZADO and r["recorte_gradiente"] == rn.RECORTE_GRADIENTE
                      and r["lote"] == rn.LOTE["NHITS" if m == "NHITS" else "RECORRENTES"]
                      and r["unidades"] == (rn.UNIDADES_NHITS if m == "NHITS" else rn.UNIDADES_RECORRENTES)
                      and r["dados"]["data_inicio_validacao"] == "2024-02-16"
                      and all(1 <= e <= r["epocas_executadas"] for e in r["melhor_epoca"])
                      and r.get("n_parametros_por_semente") == esperado_params.get((m, h), r.get("n_parametros_por_semente")))
            sem = pd.read_csv(RES / "redes" / f"{m}_{h}_por_semente.csv")
            org = pd.read_csv(RES / "redes" / f"{m}_{h}_por_origem.csv")
            media = sem.groupby(["PRACA", "ORIGEM"]).SMAPE.mean()
            dif_media = float((org.set_index(["PRACA", "ORIGEM"]).SMAPE - media).abs().max())
            z = np.load(RES / "redes" / f"{m}_{h}_previsoes.npz")
            P = z["previsao"].astype(float)
            reais = np.array([ys[str(pr)].iloc[ys[str(pr)].index.get_loc(pd.Timestamp(str(o))) + 1:][:h].values
                              for pr, o in zip(z["praca"], z["origem"])])
            refeito = np.mean(np.abs(P - reais) / ((np.abs(P) + np.abs(reais)) / 2), axis=2).mean(axis=0) * 100
            chave = list(zip([str(x) for x in z["praca"]], [str(x) for x in z["origem"]]))
            dif_npz = float(np.max(np.abs(refeito - org.set_index(["PRACA", "ORIGEM"]).SMAPE.loc[chave].values)))
            registrar(m, f"{h:3d} dias: configuração declarada, métrica = média das 5 sementes, métrica refeita "
                         f"das previsões diárias", cfg_ok and dif_media < 1e-9 and dif_npz < 1e-3,
                      f"janela {r['janela_entrada']}, {r['epocas_executadas']} épocas; dif. das previsões {dif_npz:.1e} p.p.")
    print("   (testes de vazamento e equivalência com o PyTorch: validar_procedimentos.py e "
          "validar_equivalencia_pytorch.py)")


if __name__ == "__main__":
    ys = series()
    bloco_comum()
    bloco_ingenuo(ys)
    bloco_sarima(ys)
    bloco_prophet(ys)
    bloco_redes(ys)
    pd.DataFrame(LINHAS).to_csv(RES / "validacao_modelos.csv", index=False, encoding="utf-8-sig")
    falhas = [l for l in LINHAS if l["RESULTADO"] != "OK"]
    print(f"\n{len(LINHAS) - len(falhas)} de {len(LINHAS)} verificações OK")
    print("TODOS OS MODELOS FORAM EXECUTADOS COMO DESCRITO" if not falhas else f"FALHAS: {[f['VERIFICACAO'] for f in falhas]}")
    sys.exit(0 if not falhas else 1)
