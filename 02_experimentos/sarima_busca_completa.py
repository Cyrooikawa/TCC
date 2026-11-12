"""
Busca completa da ordem do SARIMA (ponto 2 da validação de 02/11/2026).
DIAGNÓSTICO: não altera os resultados oficiais.

O problema: o texto diz que a ordem do SARIMA foi escolhida pelo menor AIC, mas a
busca usada (auto_arima passo a passo, algoritmo de Hyndman e Khandakar, 2008) não
é exaustiva. O diagnóstico (diagnostico/LEIA-ME_diagnostico.md) achou ordens com AIC
menor em Alambari, Boituva e Itu.

Dois detalhes que o diagnóstico não considerou:
  1. o auto_arima não escolhe só pelo AIC. Depois de ajustar cada candidato, ele
     aplica um teste de raízes (função _root_test do pmdarima, adaptada do
     auto.arima do R) e descarta, sem aviso, todo modelo com alguma raiz inversa
     de módulo acima de 0,99. Isso vale na busca passo a passo e na completa;
  2. a ordem é escolhida num ajuste diferente do que gera as previsões: a escolha
     usa a série com as lacunas interpoladas, em veículos, com intercepto e 50
     iterações; o ajuste final usa a série em milhares, com as lacunas preservadas
     no filtro de Kalman, sem intercepto e com até 200 iterações.

O que este script faz, para cada praça, com p e q até 3 e P e Q até 2 (144
combinações), d = 1 e D = 0 (conferidos com os testes KPSS e OCSB do auto_arima):

  Grade A (critério do auto_arima): ajusta as 144 combinações com as mesmas opções
  que o auto_arima usa internamente e registra AIC e raízes. Dela saem duas ordens:
     - AUTOARIMA_COMPLETO: menor AIC entre as que passam no teste de raízes do
       pmdarima. É o que o auto_arima devolveria com stepwise=False e
       max_order=None, a correção sugerida na validação de 02/11;
     - MENOR_AIC_SEM_TESTE: menor AIC entre todas as combinações.
  Grade B (ajuste final): ajusta as 144 combinações exatamente como
  arima_baseline.py ajusta o modelo usado nas previsões. Dela sai:
     - MENOR_AIC_AJUSTE_FINAL: menor AIC entre os ajustes que convergiram.

Depois, avalia as três variantes no teste exatamente como em arima_baseline.py
(parâmetros do treino mantidos, mesmas origens, horizontes e métricas).

A escolha entre as variantes deve ser feita por critério de método, definido antes
de olhar o erro no teste, e é uma decisão do orientador.

Nada muda em config.ORDENS_ARIMA nem em resultados/arima_*. Se o orientador aprovar
uma troca, copie a ordem escolhida (sarima_busca_completa_ordens.csv) para
config.ORDENS_ARIMA e rode de novo arima_baseline.py, comparar_modelos.py,
gerar_figuras.py e resultados_completos.py.

Saídas (pasta diagnostico/):
    sarima_busca_completa_aic.csv         grade A: AIC, raízes e testes das 144 combinações
    sarima_grade_ajuste_final_aic.csv     grade B: AIC, convergência e raízes das 144 combinações
    sarima_busca_completa_ordens.csv      ordem atual x ordens das três variantes, por praça
    sarima_busca_completa_por_origem.csv  métricas por origem das três variantes
    sarima_<variante>_<PRACA>.png         previsão de 180 dias a partir do fim do treino

Como executar (de 15 minutos a 1 hora, conforme o computador):
    python sarima_busca_completa.py --processos 4    grades que faltam (em paralelo) e avaliação
    python sarima_busca_completa.py --refazer        refaz as duas grades do zero
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

DIAG = RAIZ / "diagnostico"
DIAG.mkdir(exist_ok=True)
RESULTADOS = RAIZ / "resultados"
ESCALA = 1000                   # o mesmo de arima_baseline.py
LIMITE_RAIZ = 1 - 1e-2          # o mesmo do pmdarima: raiz inversa acima de 0,99 descarta o modelo
GRADE = list(product(range(4), range(4), range(3), range(3)))   # (p, q, P, Q): 144 combinações
VARIANTES = {"AUTOARIMA_COMPLETO": "ORDEM_AUTOARIMA_COMPLETO",
             "MENOR_AIC_SEM_TESTE": "ORDEM_MENOR_AIC_SEM_TESTE",
             "MENOR_AIC_AJUSTE_FINAL": "ORDEM_MENOR_AIC_AJUSTE_FINAL"}


def alvos():
    return [(p, "NUCLEO") for p in config.PRACAS_NUCLEO] + \
           [(p, "UNIDIRECIONAL") for p in config.PRACAS_UNIDIRECIONAIS]


def treino(praca):
    import arima_baseline as ab
    series = pd.read_csv(ab.ARQUIVO_SERIES, parse_dates=["DATA"])
    y = ab.carregar_serie(series, praca)
    return y.iloc[:int(len(y) * config.PROPORCAO_TREINO)]


_CACHE = {}


def _serie(praca, interpolada):
    chave = (praca, interpolada)
    if chave not in _CACHE:
        t = treino(praca)
        # Igual a arima_baseline.escolher_ordem: só na escolha, as lacunas são interpoladas.
        _CACHE[chave] = t.interpolate(method="linear").bfill().ffill() if interpolada else t
    return _CACHE[chave]


def _raizes(ar, ma):
    ar = np.abs(1 / np.asarray(ar)) if len(ar) else np.array([0.0])
    ma = np.abs(1 / np.asarray(ma)) if len(ma) else np.array([0.0])
    return float(ar.max()), float(ma.max())


def ajustar_criterio_autoarima(tarefa):
    """Grade A: uma combinação com as mesmas opções que o auto_arima usa internamente."""
    import pmdarima as pmd
    praca, d, D, (p, q, P, Q) = tarefa
    y = _serie(praca, interpolada=True)
    linha = {"PRACA": praca, "p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q}
    t0 = time.time()
    try:
        fit = pmd.ARIMA(order=(p, d, q), seasonal_order=(P, D, Q, config.PERIODO_SAZONAL),
                        with_intercept=(d + D) in (0, 1), method="lbfgs", maxiter=50,
                        suppress_warnings=True).fit(y)
        ar, ma = _raizes(fit.arroots() if p + P > 0 else [], fit.maroots() if q + Q > 0 else [])
        linha.update({"AIC": float(fit.aic()), "MAX_RAIZ_INVERSA_AR": ar, "MAX_RAIZ_INVERSA_MA": ma,
                      "CONVERGIU": bool(fit.arima_res_.mle_retvals.get("converged", False))})
        # Regra do pmdarima 2.1.1 (_root_test): havendo termos MA, só as raízes MA são olhadas.
        olhada = ma if q + Q > 0 else ar
        linha["PASSA_TESTE_PMDARIMA"] = bool(np.isfinite(linha["AIC"]) and olhada <= LIMITE_RAIZ)
        # Regra do auto.arima do R: olha as raízes AR e as MA.
        linha["PASSA_TESTE_R"] = bool(np.isfinite(linha["AIC"]) and max(ar, ma) <= LIMITE_RAIZ)
        linha["ERRO"] = ""
    except Exception as e:  # noqa: BLE001 -- registra a falha em vez de escondê-la
        linha.update({"AIC": np.nan, "MAX_RAIZ_INVERSA_AR": np.nan, "MAX_RAIZ_INVERSA_MA": np.nan,
                      "CONVERGIU": False, "PASSA_TESTE_PMDARIMA": False, "PASSA_TESTE_R": False,
                      "ERRO": f"{type(e).__name__}: {str(e)[:200]}"})
    linha["TEMPO_S"] = round(time.time() - t0, 2)
    return linha


def ajustar_como_final(tarefa):
    """Grade B: uma combinação ajustada exatamente como em arima_baseline.py."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    praca, d, D, (p, q, P, Q) = tarefa
    y = _serie(praca, interpolada=False)
    linha = {"PRACA": praca, "p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q}
    t0 = time.time()
    try:
        r = SARIMAX(y / ESCALA, order=(p, d, q), seasonal_order=(P, D, Q, config.PERIODO_SAZONAL),
                    trend="c" if d + D == 0 else "n",
                    enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
        ar, ma = _raizes(r.arroots, r.maroots)
        linha.update({"AIC": float(r.aic), "CONVERGIU": bool(r.mle_retvals.get("converged", False)),
                      "MAX_RAIZ_INVERSA_AR": ar, "MAX_RAIZ_INVERSA_MA": ma, "ERRO": ""})
    except Exception as e:  # noqa: BLE001
        linha.update({"AIC": np.nan, "CONVERGIU": False, "MAX_RAIZ_INVERSA_AR": np.nan,
                      "MAX_RAIZ_INVERSA_MA": np.nan, "ERRO": f"{type(e).__name__}: {str(e)[:200]}"})
    linha["TEMPO_S"] = round(time.time() - t0, 2)
    return linha


def ordens_de_diferenciacao():
    from pmdarima.arima import ndiffs, nsdiffs
    d_D = {}
    for praca, _ in alvos():
        y = _serie(praca, interpolada=True)
        # Mesmos testes e limites padrão do auto_arima (OCSB, máx. D = 1; KPSS, máx. d = 2).
        D = nsdiffs(y, m=config.PERIODO_SAZONAL, test="ocsb", max_D=1)
        d = ndiffs(y.diff(config.PERIODO_SAZONAL).dropna() if D else y, test="kpss", max_d=2)
        d_D[praca] = (d, D)
        print(f"{praca}: d = {d}, D = {D} (testes KPSS e OCSB do auto_arima)", flush=True)
    return d_D


def rodar_grade(funcao, arquivo, processos, nome):
    d_D = ordens_de_diferenciacao()
    tarefas = [(praca, *d_D[praca], c) for praca, _ in alvos() for c in GRADE]
    t0 = time.time()
    print(f"{nome}: {len(tarefas)} ajustes", flush=True)
    if processos > 1:
        from multiprocessing import Pool
        with Pool(processos) as pool:
            linhas = []
            for i, r in enumerate(pool.imap_unordered(funcao, tarefas, chunksize=2), start=1):
                linhas.append(r)
                if i % 144 == 0:
                    print(f"  {i}/{len(tarefas)} ajustes ({(time.time() - t0) / 60:.1f} min)", flush=True)
    else:
        linhas = [funcao(t) for t in tarefas]
    tabela = pd.DataFrame(linhas).sort_values(["PRACA", "AIC"])
    tabela.to_csv(DIAG / arquivo, index=False, encoding="utf-8-sig")
    print(f"{nome} concluída em {(time.time() - t0) / 60:.1f} min", flush=True)
    return tabela


def ordem_texto(r):
    return f"({int(r.p)},{int(r.d)},{int(r.q)})({int(r.P)},{int(r.D)},{int(r.Q)})7"


def escolher(aic, aic_final):
    linhas = []
    for praca, grupo in alvos():
        a = aic[aic.PRACA == praca]
        b = aic_final[aic_final.PRACA == praca]
        (p, d, q), (P, D, Q, _) = config.ORDENS_ARIMA[praca]
        mesma = lambda t: t[(t.p == p) & (t.q == q) & (t.P == P) & (t.Q == Q)]  # noqa: E731
        atual_a, atual_b = mesma(a), mesma(b)
        auto = a[a.PASSA_TESTE_PMDARIMA].sort_values("AIC").iloc[0]
        menor = a.dropna(subset=["AIC"]).sort_values("AIC").iloc[0]
        final = b[b.CONVERGIU].dropna(subset=["AIC"]).sort_values("AIC").iloc[0]
        linhas.append({
            "PRACA": praca, "GRUPO": grupo,
            "ORDEM_ATUAL": f"({p},{d},{q})({P},{D},{Q})7",
            "AIC_A_ORDEM_ATUAL": float(atual_a.AIC.iloc[0]),
            "ATUAL_PASSA_TESTE_PMDARIMA": bool(atual_a.PASSA_TESTE_PMDARIMA.iloc[0]),
            "AIC_B_ORDEM_ATUAL": float(atual_b.AIC.iloc[0]),
            "ORDEM_AUTOARIMA_COMPLETO": ordem_texto(auto), "AIC_A_AUTOARIMA_COMPLETO": float(auto.AIC),
            "ORDEM_MENOR_AIC_SEM_TESTE": ordem_texto(menor), "AIC_A_MENOR_AIC_SEM_TESTE": float(menor.AIC),
            "RAIZ_INV_MAX_MENOR_AIC_SEM_TESTE": max(menor.MAX_RAIZ_INVERSA_AR, menor.MAX_RAIZ_INVERSA_MA),
            "ORDEM_MENOR_AIC_AJUSTE_FINAL": ordem_texto(final), "AIC_B_MENOR_AIC_AJUSTE_FINAL": float(final.AIC),
            "RAIZ_INV_MAX_MENOR_AIC_AJUSTE_FINAL": max(final.MAX_RAIZ_INVERSA_AR, final.MAX_RAIZ_INVERSA_MA),
            "A_PASSAM_TESTE_PMDARIMA": int(a.PASSA_TESTE_PMDARIMA.sum()),
            "A_FALHAS": int(a.AIC.isna().sum()),
            "B_CONVERGIRAM": int(b.CONVERGIU.sum()), "B_FALHAS": int(b.AIC.isna().sum()),
        })
    return pd.DataFrame(linhas)


def para_config(texto):
    """'(3,1,2)(1,0,1)7' -> ((3, 1, 2), (1, 0, 1, 7))"""
    a, b = texto.split(")(")
    o = tuple(int(x) for x in a.strip("(").split(","))
    s = tuple(int(x) for x in b.split(")")[0].split(","))
    return o, s + (config.PERIODO_SAZONAL,)


def etapa_avaliacao(ordens):
    """Avalia cada variante com o mesmo código de arima_baseline.py, sem tocar nos arquivos oficiais."""
    import arima_baseline as ab
    series = pd.read_csv(ab.ARQUIVO_SERIES, parse_dates=["DATA"])
    ab.PASTA_RESULTADOS = DIAG
    oficiais = config.ORDENS_ARIMA.copy()
    linhas = []
    ordens = ordens.copy()
    for variante, coluna in VARIANTES.items():
        aic_final, ljung = {}, {}
        for praca, grupo in alvos():
            texto = ordens.loc[ordens.PRACA == praca, coluna].iloc[0]
            config.ORDENS_ARIMA[praca] = para_config(texto)          # só na memória
            try:
                l_, info = ab.avaliar_praca(ab.carregar_serie(series, praca), praca, grupo)
            except RuntimeError as e:          # a checagem de segurança do arima_baseline recusou o ajuste
                print(f"  {variante} {praca}: ajuste recusado ({e})")
                aic_final[praca], ljung[praca] = np.nan, np.nan
                continue
            for r in l_:
                r.update({"MODELO": f"SARIMA_{variante}", "ORDEM": texto})
            linhas.extend(l_)
            aic_final[praca], ljung[praca] = info["AIC_EM_MILHARES"], info["LJUNGBOX_P_14DIAS"]
            nome = praca.replace(" ", "_")
            if (DIAG / f"arima_{nome}.png").exists():
                (DIAG / f"arima_{nome}.png").replace(DIAG / f"sarima_{variante.lower()}_{nome}.png")
        ordens[f"AIC_FINAL_MILHARES_{variante}"] = ordens.PRACA.map(aic_final)
        ordens[f"LJUNGBOX_P14_{variante}"] = ordens.PRACA.map(ljung)
    config.ORDENS_ARIMA.update(oficiais)
    of = pd.read_csv(RESULTADOS / "arima_modelos.csv").set_index("PRACA")
    ordens["AIC_FINAL_MILHARES_ATUAL"] = ordens.PRACA.map(of["AIC_EM_MILHARES"])
    ordens.to_csv(DIAG / "sarima_busca_completa_ordens.csv", index=False, encoding="utf-8-sig")
    por_origem = pd.DataFrame(linhas)
    por_origem.to_csv(DIAG / "sarima_busca_completa_por_origem.csv", index=False, encoding="utf-8-sig")
    return ordens, por_origem


def resumo(ordens, por_origem):
    pd.set_option("display.width", 250)
    print("\n=== Ordens por praça (AIC A: critério do auto_arima; AIC B: ajuste final, em milhares) ===")
    print(ordens[["PRACA", "ORDEM_ATUAL", "AIC_B_ORDEM_ATUAL", "ORDEM_AUTOARIMA_COMPLETO",
                  "AIC_FINAL_MILHARES_AUTOARIMA_COMPLETO", "ORDEM_MENOR_AIC_SEM_TESTE",
                  "AIC_FINAL_MILHARES_MENOR_AIC_SEM_TESTE", "ORDEM_MENOR_AIC_AJUSTE_FINAL",
                  "AIC_B_MENOR_AIC_AJUSTE_FINAL"]].round(1).to_string(index=False))
    oficial = pd.read_csv(RESULTADOS / "arima_metricas_por_origem.csv")
    oficial["MODELO"] = "SARIMA_ATUAL"
    tudo = pd.concat([oficial, por_origem], ignore_index=True)
    t = tudo.groupby(["PRACA", "DIAS", "MODELO"]).SMAPE.mean().unstack()
    print("\n=== SMAPE médio (%) por praça ===")
    print(t.round(2).to_string())
    print("\n=== Média das seis praças ===")
    print(t.groupby(level="DIAS").mean().round(2).to_string())


def carregar_ou_rodar(arquivo, funcao, processos, nome, refazer):
    if (DIAG / arquivo).exists() and not refazer:
        print(f"{nome}: usando {arquivo} já salvo (use --refazer para refazer)")
        return pd.read_csv(DIAG / arquivo)
    return rodar_grade(funcao, arquivo, processos, nome)


if __name__ == "__main__":
    args = sys.argv[1:]
    processos = int(args[args.index("--processos") + 1]) if "--processos" in args else 1
    refazer = "--refazer" in args
    aic = carregar_ou_rodar("sarima_busca_completa_aic.csv", ajustar_criterio_autoarima, processos,
                            "Grade A (critério do auto_arima)", refazer)
    aic_final = carregar_ou_rodar("sarima_grade_ajuste_final_aic.csv", ajustar_como_final, processos,
                                  "Grade B (ajuste final)", refazer)
    ordens, por_origem = etapa_avaliacao(escolher(aic, aic_final))
    resumo(ordens, por_origem)
    print(f"\nArquivos em {DIAG}")
