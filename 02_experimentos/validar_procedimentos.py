"""
Validação automática dos procedimentos e dos resultados esperados do TCC.

Roda só com NumPy, pandas e SciPy (não precisa de statsmodels, Prophet nem PyTorch).
Coloque na pasta 02_codigo_e_resultados, junto de config.py, metricas.py,
redes_numpy.py e redes_neurais.py, e execute:

    python validar_procedimentos.py                      # todas as verificações rápidas
    python validar_procedimentos.py --reproduzir NHITS 7 # refaz uma rede e compara (5 a 40 min)

Cada verificação imprime OK ou FALHOU, com o detalhe. O código de saída é 0 só
quando todas passam.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402
import metricas  # noqa: E402

RESULTADOS = RAIZ / "resultados"
FALHAS = []

# SMAPE médio das seis praças obtido na execução de referência (resultados esperados).
ESPERADO_SMAPE = {
    "INGENUO": {7: 7.4019, 90: 8.7208, 180: 8.9143},
    "ARIMA":   {7: 6.7654, 90: 7.6607, 180: 7.6043},
    "PROPHET": {7: 4.6358, 90: 4.7092, 180: 4.6810},
    "LSTM":    {7: 6.3745, 90: 7.3793, 180: 10.1473},
    "GRU":     {7: 6.4352, 90: 6.5380, 180: 7.3796},
    "BILSTM":  {7: 6.4627, 90: 6.8852, 180: 7.5963},
    "BIGRU":   {7: 6.4096, 90: 6.5460, 180: 7.1399},
    "NHITS":   {7: 6.1277, 90: 6.5781, 180: 6.8386},
}
# Tolerâncias (pontos percentuais de SMAPE). Os modelos estatísticos são determinísticos;
# as redes podem variar um pouco entre computadores (BLAS e ordem das operações), e a
# LSTM em 180 dias é a mais sensível (platô no treino; ver seção 4.5.3).
TOLERANCIA = {"INGENUO": 0.01, "ARIMA": 0.05, "PROPHET": 0.10, "PADRAO_REDES": 0.50, ("LSTM", 180): 2.0}
# Configuração ajustada (padrão de config.py; capítulo 4): SARIMA com a ordem de menor AIC,
# Prophet com janela própria para os feriados prolongados e LSTM de 180 dias com paciência 40.
if config.CONFIGURACAO == "ajustada":
    ESPERADO_SMAPE.update({
        "ARIMA":   {7: 5.8971, 90: 6.4584, 180: 6.4231},
        "PROPHET": {7: 4.4270, 90: 4.4944, 180: 4.4668},
        "LSTM":    {7: 6.3745, 90: 7.3793, 180: 7.6733},
    })
    # O SARIMA com raiz sazonal quase unitária varia um pouco entre computadores (otimizador).
    TOLERANCIA.update({"ARIMA": 0.10, ("LSTM", 180): 0.50})


def verificar(nome, condicao, detalhe=""):
    estado = "OK     " if condicao else "FALHOU "
    print(f"[{estado}] {nome}" + (f" — {detalhe}" if detalhe else ""))
    if not condicao:
        FALHAS.append(nome)


def carregar_series():
    df = pd.read_csv(RAIZ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
    alvos = [(p, "NUCLEO") for p in config.PRACAS_NUCLEO] + [(p, "UNIDIRECIONAL") for p in config.PRACAS_UNIDIRECIONAIS]
    saida = []
    for praca, grupo in alvos:
        sub = df[(df["PRACA"] == praca) & (df["DATA"] >= config.DATA_INICIO_MODELAGEM)].set_index("DATA").sort_index()
        saida.append((praca, grupo, sub["VOLUME_TOTAL"].asfreq("D"), sub["STATUS"].asfreq("D")))
    return saida


def bloco_dados(series):
    print("\n1. Série, divisão e origens")
    n = len(series[0][2])
    n_treino = int(n * config.PROPORCAO_TREINO)
    for praca, _, y, status in series:
        ok = (len(y) == 2038 and str(y.index[0].date()) == "2021-01-01" and str(y.index[-1].date()) == "2026-07-31")
        verificar(f"{praca}: 2.038 dias de 2021-01-01 a 2026-07-31", ok, f"{len(y)} dias")
        nan_ok = (y.isna() == (status != "OK")).all()
        verificar(f"{praca}: dias ausentes e invalidados são NaN, sem imputação", bool(nan_ok))
        verificar(f"{praca}: nenhum dia ausente no teste", int(y.iloc[n_treino:].isna().sum()) == 0)
    verificar("Treino de 1.426 dias, até 2024-11-26", n_treino == 1426 and
              str(series[0][2].index[n_treino - 1].date()) == "2024-11-26")
    todas = metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS, min(config.HORIZONTES.values()))
    idx = series[0][2].index
    for h, esperado in [(7, 87), (90, 75), (180, 62)]:
        orig = [o for o in todas if o + h <= n - 1]
        dias = {idx[o].weekday() for o in orig}
        passos = set(np.diff(orig))
        verificar(f"Horizonte de {h} dias: {esperado} origens, todas terças, a cada 7 dias", len(orig) == esperado
                  and dias == {1} and passos == {7} and str(idx[orig[0]].date()) == "2024-11-26",
                  f"{len(orig)} origens")


def bloco_metricas(series):
    print("\n2. Métricas")
    rng = np.random.default_rng(0)
    y = rng.uniform(1000, 5000, 30)
    zeros = metricas.todas(y, y, 7)
    verificar("Previsão igual ao real dá erro zero nas cinco métricas", all(abs(v) < 1e-12 for v in zeros.values()))
    alambari = [s for s in series if s[0] == "ALAMBARI"][0][2]
    o = alambari.index.get_loc(pd.Timestamp("2025-05-13"))
    real, prev = alambari.iloc[o + 1:o + 8].values, alambari.iloc[o - 6:o + 1].values
    m = metricas.todas(real, prev, 7)
    esperado = {"MAE": 599.0, "RMSE": 796.6637, "MAPE": 4.6355, "SMAPE": 4.4659, "DTW": 511.0}
    verificar("Teste de mesa (Alambari, 14 a 20/05/2025) bate com o cálculo à mão",
              all(abs(m[k] - v) < 1e-3 for k, v in esperado.items()),
              ", ".join(f"{k} {m[k]:.4f}" for k in esperado))


def bloco_ingenuo(series):
    print("\n3. Modelo ingênuo recalculado do zero")
    arq = RESULTADOS / "ingenuo_metricas_por_origem.csv"
    if not arq.exists():
        verificar("Arquivo ingenuo_metricas_por_origem.csv presente", False)
        return
    oficial = pd.read_csv(arq)
    n = len(series[0][2])
    linhas = []
    for praca, grupo, y, _ in series:
        for o in metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS, 7):
            semana = y.iloc[:o + 1].ffill().iloc[-7:].values
            for nome, h in config.HORIZONTES.items():
                if o + h > n - 1:
                    continue
                r = metricas.todas(y.iloc[o + 1:o + 1 + h].values, np.tile(semana, h // 7 + 1)[:h], 7)
                r.update({"PRACA": praca, "DIAS": h, "ORIGEM": str(y.index[o].date())})
                linhas.append(r)
    novo = pd.DataFrame(linhas)
    m = novo.merge(oficial, on=["PRACA", "DIAS", "ORIGEM"], suffixes=("", "_of"))
    dif = max(float((m[k] - m[f"{k}_of"]).abs().max()) for k in ["MAE", "RMSE", "MAPE", "SMAPE", "DTW"])
    verificar("Ingênuo recalculado = arquivo oficial (1.344 janelas)", len(m) == 1344 and dif < 1e-6,
              f"{len(m)} janelas, maior diferença {dif:.2e}")


def bloco_pareamento():
    print("\n4. Pareamento entre modelos")
    arquivos = sorted(RESULTADOS.glob("*_metricas_por_origem.csv"))
    chaves = {}
    for a in arquivos:
        d = pd.read_csv(a)
        chaves[d.MODELO.iloc[0]] = set(zip(d.PRACA, d.DIAS, d.ORIGEM))
        verificar(f"{a.name}: 1.344 linhas (6 praças × 224 janelas)", len(d) == 1344, f"{len(d)} linhas")
    ref = next(iter(chaves.values()))
    verificar("Todos os modelos avaliados nas mesmas praças, horizontes e origens",
              all(c == ref for c in chaves.values()), ", ".join(chaves))
    return list(chaves)


def bloco_vazamento():
    print("\n5. Redes: ausência de vazamento de dados (testes com dados adulterados)")
    try:
        import redes_neurais as rn
    except Exception as e:  # noqa: BLE001
        verificar("redes_neurais.py importável", False, str(e))
        return
    base = rn.carregar()
    n = len(base[0][2])
    n_treino = int(n * config.PROPORCAO_TREINO)
    ini_val = n_treino - int(round(rn.FRACAO_VALIDACAO * n_treino))
    for h in (7, 90, 180):
        L = rn.JANELA_ENTRADA[h]
        Xtr, Ytr, Xva, Yva, teste, escalas, info = rn.montar_dados(base, L, h)
        verificar(f"h={h}: janelas de treino e validação sem dias ausentes",
                  not (np.isnan(Xtr).any() or np.isnan(Ytr).any() or np.isnan(Xva).any() or np.isnan(Yva).any()))
        # (a) Mudar todo o período de teste não pode alterar treino, validação nem a normalização.
        adult = [(p, g, y.where(np.arange(n) < n_treino, y * 100.0)) for p, g, y in base]
        A = rn.montar_dados(adult, L, h)
        iguais = all(np.array_equal(a, b) for a, b in zip((Xtr, Ytr, Xva, Yva), A[:4])) and A[5] == escalas
        verificar(f"h={h}: teste multiplicado por 100 não muda treino, validação nem Min-Max", iguais)
        # (b) Permutar o período de validação (mesmo mínimo e máximo) não pode alterar as janelas de treino.
        def permutar(y):
            v = y.values.copy()
            trecho = v[ini_val:n_treino].copy()
            np.random.default_rng(1).shuffle(trecho)
            v[ini_val:n_treino] = trecho
            return pd.Series(v, index=y.index)
        B = rn.montar_dados([(p, g, permutar(y)) for p, g, y in base], L, h)
        verificar(f"h={h}: janelas de treino terminam antes da validação",
                  np.array_equal(Xtr, B[0]) and np.array_equal(Ytr, B[1]))
        # (c) A entrada de cada origem só usa dados até a origem: adulterar tudo o que vem
        #     depois de uma origem não pode mudar a entrada dessa origem (nas seis praças).
        originais = {(r["praca"], r["origem"]): r["entrada"] for r in teste}
        todas_origens = sorted({r["origem"] for r in teste})
        amostra = todas_origens[::max(1, len(todas_origens) // 5)]
        ok = True
        for o in amostra:
            futuro = [(p, g, y.where(np.arange(n) <= o, y * 50.0)) for p, g, y in base]
            for r in rn.montar_dados(futuro, L, h)[4]:
                if r["origem"] == o:
                    ok &= np.array_equal(r["entrada"], originais[(r["praca"], o)])
        verificar(f"h={h}: dados posteriores à origem não alteram a entrada da rede", bool(ok),
                  f"{len(amostra)} origens × 6 praças")


def bloco_esperado(modelos):
    print("\n6. Resultados comparados com os esperados (SMAPE médio das seis praças)")
    d = pd.concat([pd.read_csv(a) for a in RESULTADOS.glob("*_metricas_por_origem.csv")])
    smape = d.groupby(["MODELO", "DIAS", "PRACA"]).SMAPE.mean().groupby(["MODELO", "DIAS"]).mean()
    for m in ESPERADO_SMAPE:
        if m not in modelos:
            print(f"[AVISO  ] {m}: sem resultados nesta pasta")
            continue
        for h, esp in ESPERADO_SMAPE[m].items():
            tol = TOLERANCIA.get((m, h), TOLERANCIA.get(m, TOLERANCIA["PADRAO_REDES"]))
            obt = smape.loc[(m, h)]
            verificar(f"{m:8s} {h:3d} dias: SMAPE {obt:.2f}% (esperado {esp:.2f}% ± {tol})", abs(obt - esp) <= tol)
    if {"PROPHET"} <= set(modelos):
        melhor = all(smape.xs(h, level="DIAS").idxmin() == "PROPHET" for h in (7, 90, 180))
        verificar("Prophet com o menor SMAPE médio nos três horizontes", melhor)


def reproduzir(modelo, h):
    print(f"\n7. Reprodutibilidade: refazendo {modelo} em {h} dias (pode levar de 5 a 40 minutos)")
    import redes_neurais as rn
    antigo = pd.read_csv(RESULTADOS / "redes" / f"{modelo}_{h}_por_origem.csv")
    with tempfile.TemporaryDirectory() as tmp:
        rn.SAIDA = Path(tmp)
        rn.main(modelo, h)
        novo = pd.read_csv(Path(tmp) / f"{modelo}_{h}_por_origem.csv")
    dif = float((novo.SMAPE - antigo.SMAPE).abs().max())
    media = abs(novo.SMAPE.mean() - antigo.SMAPE.mean())
    verificar(f"{modelo} {h} dias refeito: diferença na média de SMAPE de {media:.4f} p.p. "
              f"(maior diferença por origem {dif:.4f})", media <= TOLERANCIA["PADRAO_REDES"])


if __name__ == "__main__":
    series = carregar_series()
    bloco_dados(series)
    bloco_metricas(series)
    bloco_ingenuo(series)
    modelos = bloco_pareamento()
    bloco_vazamento()
    bloco_esperado(modelos)
    if "--reproduzir" in sys.argv:
        i = sys.argv.index("--reproduzir")
        reproduzir(sys.argv[i + 1].upper(), int(sys.argv[i + 2]))
    print("\n" + ("TODAS AS VERIFICAÇÕES PASSARAM" if not FALHAS else f"{len(FALHAS)} VERIFICAÇÃO(ÕES) FALHARAM: {FALHAS}"))
    sys.exit(0 if not FALHAS else 1)
