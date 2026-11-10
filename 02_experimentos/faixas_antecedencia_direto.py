"""
Erro por faixa de antecedência calculado DIRETO das previsões diárias.
Corrige a Figura 11 e a seção 4.4.3 (ponto 1 da validação de 02/11/2026).

As Equações 7 e 8 obtêm o SMAPE dos dias 8 a 90 e 91 a 180 a partir do SMAPE das
janelas de 7, 90 e 180 dias de uma mesma origem:

    SMAPE(8 a 90)   = (90 x SMAPE90  - 7 x SMAPE7)  / 83
    SMAPE(91 a 180) = (180 x SMAPE180 - 90 x SMAPE90) / 90

Isso só vale quando as três janelas saem da MESMA trajetória de previsão, como no
ingênuo sazonal, no SARIMA e no Prophet. Nas redes, cada horizonte tem a sua rede
(outra janela de entrada e outros pesos), e a conta mistura três modelos.

O que este script faz:
  - redes: calcula o SMAPE de cada faixa direto da previsão diária da rede de 180
    dias, semente por semente, e tira a média das cinco sementes (o mesmo critério
    das métricas oficiais);
  - estatísticos: mantém as Equações 7 e 8, que valem para eles, e confere isso no
    ingênuo, recalculando a previsão diária do zero;
  - compara, nos mesmos dias, as redes treinadas para cada horizonte (por exemplo,
    os dias 1 a 7 previstos pelas redes de 7, 90 e 180 dias).
Usa as 62 origens que têm os três horizontes, como a Figura 11.

Entradas: dados/processados/series_diarias.csv, resultados/*_metricas_por_origem.csv
          e resultados/redes/<REDE>_<h>_previsoes.npz
Saídas:   resultados/completos_faixas_antecedencia.csv      (por praça e modelo; substitui o anterior)
          resultados/faixas_redes_por_horizonte_treinado.csv (mesmos dias, rede de cada horizonte)
          figuras/faixas_antecedencia.png                     (Figura 11 refeita)

resultados_completos.py também passa a usar este cálculo para as redes.

Como executar:
    python faixas_antecedencia_direto.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402
import metricas  # noqa: E402

RESULTADOS = RAIZ / "resultados"
REDES = RESULTADOS / "redes"
REDES_NEURAIS = ["LSTM", "GRU", "BILSTM", "BIGRU", "NHITS"]
ESTATISTICOS = ["INGENUO", "ARIMA", "PROPHET"]
FAIXAS = {"DIAS_1_7": (0, 7), "DIAS_8_90": (7, 90), "DIAS_91_180": (90, 180)}


def carregar_reais():
    df = pd.read_csv(RAIZ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
    df = df[df["DATA"] >= config.DATA_INICIO_MODELAGEM]
    return {p: g.set_index("DATA")["VOLUME_TOTAL"].sort_index().asfreq("D") for p, g in df.groupby("PRACA")}


def origens_comuns():
    """Origens (datas) que têm os três horizontes: as 62 da Figura 11."""
    n = len(next(iter(carregar_reais().values())))
    idx = next(iter(carregar_reais().values())).index
    todas = metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS, min(config.HORIZONTES.values()))
    h_max = max(config.HORIZONTES.values())
    return {str(idx[o].date()) for o in todas if o + h_max <= n - 1}


def previsoes_rede(modelo, h):
    z = np.load(REDES / f"{modelo}_{h}_previsoes.npz")
    praca = np.array([str(p) for p in z["praca"]], dtype=object)
    origem = np.array([str(o) for o in z["origem"]], dtype=object)
    return praca, origem, z["previsao"].astype(np.float64)   # (S, N, h)


def smape_por_trecho(reais, praca, origem, P, inicio, fim):
    """SMAPE de cada linha no trecho [inicio, fim) do horizonte, média das sementes."""
    saida = np.empty(len(praca))
    for i, (p, o) in enumerate(zip(praca, origem)):
        y = reais[p]
        pos = y.index.get_loc(pd.Timestamp(o))
        real = y.iloc[pos + 1 + inicio: pos + 1 + fim].values
        saida[i] = np.mean([metricas.smape(real, P[s, i, inicio:fim]) for s in range(P.shape[0])])
    return saida


def faixas_redes_direto(reais=None, modelos=REDES_NEURAIS, h_max=180):
    """Faixas de cada rede calculadas direto da previsão diária da rede de 180 dias."""
    reais = reais or carregar_reais()
    comuns = origens_comuns()
    partes = []
    for m in modelos:
        arq = REDES / f"{m}_{h_max}_previsoes.npz"
        if not arq.exists():
            continue
        praca, origem, P = previsoes_rede(m, h_max)
        sel = np.isin(origem, list(comuns))
        praca, origem, P = praca[sel], origem[sel], P[:, sel, :]
        f = pd.DataFrame({"PRACA": praca, "ORIGEM": pd.to_datetime(origem), "MODELO": m})
        for nome, (a, b) in FAIXAS.items():
            f[nome] = smape_por_trecho(reais, praca, origem, P, a, b)
        partes.append(f)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def faixas_equacoes(d):
    """Equações 7 e 8 (válidas só para modelos com uma única trajetória de previsão)."""
    pv = d.pivot_table(index=["PRACA", "ORIGEM", "MODELO"], columns="DIAS", values="SMAPE").dropna()
    return pd.DataFrame({"DIAS_1_7": pv[7],
                         "DIAS_8_90": (90 * pv[90] - 7 * pv[7]) / 83,
                         "DIAS_91_180": (180 * pv[180] - 90 * pv[90]) / 90}).reset_index()


def redes_por_horizonte_treinado(reais=None):
    """Mesmos dias previstos pelas redes treinadas para 7, 90 e 180 dias (62 origens)."""
    reais = reais or carregar_reais()
    comuns = origens_comuns()
    linhas = []
    for m in REDES_NEURAIS:
        for h in sorted(config.HORIZONTES.values()):
            if not (REDES / f"{m}_{h}_previsoes.npz").exists():
                continue
            praca, origem, P = previsoes_rede(m, h)
            sel = np.isin(origem, list(comuns))
            praca, origem, P = praca[sel], origem[sel], P[:, sel, :]
            for nome, (a, b) in FAIXAS.items():
                if b > h:
                    continue
                v = smape_por_trecho(reais, praca, origem, P, a, b)
                linhas.append({"MODELO": m, "REDE_DE": h, "FAIXA": nome, "SMAPE": v.mean(),
                               "ORIGENS": len(set(origem)), "PRACAS": len(set(praca))})
    return pd.DataFrame(linhas)


def conferir(reais, d):
    """Conferências: (1) métricas recalculadas = arquivos oficiais; (2) Equações 7 e 8 = cálculo
    direto no ingênuo; (3) a média ponderada das faixas = SMAPE de 180 dias da rede."""
    ok = True
    for m in REDES_NEURAIS:
        for h in sorted(config.HORIZONTES.values()):
            arq = REDES / f"{m}_{h}_previsoes.npz"
            if not arq.exists():
                continue
            praca, origem, P = previsoes_rede(m, h)
            novo = smape_por_trecho(reais, praca, origem, P, 0, h)
            of = pd.read_csv(REDES / f"{m}_{h}_por_origem.csv")
            of = of.set_index(["PRACA", "ORIGEM"]).SMAPE
            dif = np.max(np.abs(novo - of.loc[list(zip(praca, origem))].values))
            ok &= dif < 1e-3
            print(f"  {m:6s} {h:3d} dias: SMAPE recalculado da previsão diária x arquivo oficial, "
                  f"maior diferença {dif:.1e} p.p.")
    # Ingênuo: previsão diária refeita do zero e faixas calculadas direto.
    comuns = origens_comuns()
    linhas = []
    for p, y in reais.items():
        if p not in config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS:
            continue
        for o in sorted(comuns):
            pos = y.index.get_loc(pd.Timestamp(o))
            semana = y.iloc[:pos + 1].ffill().iloc[-7:].values
            prev = np.tile(semana, 180 // 7 + 1)[:180]
            real = y.iloc[pos + 1: pos + 181].values
            linhas.append({"PRACA": p, "ORIGEM": pd.Timestamp(o),
                           **{k: metricas.smape(real[a:b], prev[a:b]) for k, (a, b) in FAIXAS.items()}})
    direto = pd.DataFrame(linhas).set_index(["PRACA", "ORIGEM"]).sort_index()
    eq = faixas_equacoes(d[d.MODELO == "INGENUO"]).set_index(["PRACA", "ORIGEM"]).sort_index()
    dif = float(np.max(np.abs(direto[list(FAIXAS)].values - eq.loc[direto.index, list(FAIXAS)].values)))
    ok &= dif < 1e-9
    print(f"  Ingênuo: Equações 7 e 8 x cálculo direto da previsão diária, maior diferença {dif:.1e} p.p.")
    return ok


def montar(d):
    """Tabela final por praça e modelo: equações nos estatísticos, cálculo direto nas redes."""
    eq = faixas_equacoes(d[~d.MODELO.isin(REDES_NEURAIS)])
    eq["METODO"] = "trajetoria unica (Equacoes 7 e 8)"
    direto = faixas_redes_direto()
    if len(direto):
        direto["METODO"] = "direto da previsao diaria da rede de 180 dias"
    f = pd.concat([eq, direto], ignore_index=True)
    return f


if __name__ == "__main__":
    import resultados_completos as rc   # mesmas cores, rótulos e estilo das demais figuras

    d, modelos = rc.carregar()
    reais = carregar_reais()
    print("Conferências:")
    tudo_ok = conferir(reais, d)

    f = montar(d)
    por_praca = f.groupby(["PRACA", "MODELO"])[list(FAIXAS)].mean().reset_index()
    por_praca.to_csv(RESULTADOS / "completos_faixas_antecedencia.csv", index=False, encoding="utf-8-sig")
    media = f.groupby("MODELO")[list(FAIXAS)].mean().reindex(modelos)

    antigo = faixas_equacoes(d).groupby("MODELO")[list(FAIXAS)].mean().reindex(modelos)
    comparacao = pd.concat({"pelas equacoes (texto atual)": antigo, "corrigido": media}, axis=1)
    print("\nSMAPE médio das seis praças por faixa (%), 62 origens:")
    print(comparacao.round(2).to_string())

    # Conferência da decomposição: 7/180, 83/180 e 90/180 das faixas = SMAPE de 180 dias.
    pv180 = d[d.DIAS == 180].groupby("MODELO").SMAPE.mean().reindex(modelos)
    recomposto = (7 * media.DIAS_1_7 + 83 * media.DIAS_8_90 + 90 * media.DIAS_91_180) / 180
    dif = float((recomposto - pv180).abs().max())
    tudo_ok &= dif < 1e-3
    print(f"\nMédia ponderada das três faixas x SMAPE de 180 dias: maior diferença {dif:.1e} p.p.")

    cruz = redes_por_horizonte_treinado(reais)
    cruz.to_csv(RESULTADOS / "faixas_redes_por_horizonte_treinado.csv", index=False, encoding="utf-8-sig")
    tab = cruz.pivot_table(index=["MODELO", "REDE_DE"], columns="FAIXA", values="SMAPE")[list(FAIXAS)]
    print("\nMesmos dias previstos pela rede de cada horizonte (SMAPE %, 62 origens, seis praças):")
    print(tab.reindex(REDES_NEURAIS, level=0).round(2).to_string())

    rc.figura11(media, modelos)
    print(f"\nFigura 11 refeita: {rc.FIGURAS / 'faixas_antecedencia.png'}")
    print("\n" + ("TODAS AS CONFERÊNCIAS PASSARAM" if tudo_ok else "ATENÇÃO: alguma conferência falhou"))
    sys.exit(0 if tudo_ok else 1)
