"""
Testes de significância com a MÉDIA DAS PRAÇAS POR ORIGEM como bloco
(ponto 3 da validação de 02/11/2026; complementa a Tabela 10 e a seção 4.4.4).

Por que: as seis praças compartilham as datas e os feriados. Na mesma origem, a
diferença de SMAPE entre dois modelos é muito parecida nas seis praças (correlação
média perto de 0,8), e as seis valem juntas pouco mais de uma observação
independente. Tratar cada praça como um bloco independente (522, 36 e 18 blocos na
Tabela 10) deixa os valores-p otimistas.

O que o script faz, em cada horizonte:
  1. usa os mesmos blocos sem sobreposição da Tabela 10 (uma origem a cada 1, 13 e
     26 origens, para 7, 90 e 180 dias);
  2. em cada origem, tira a média do SMAPE das seis praças: 87, 6 e 3 blocos;
  3. aplica o Friedman e o Wilcoxon com correção de Holm nos 28 pares;
  4. calcula, para cada par, medidas descritivas: diferença mediana e média (p.p.),
     % de blocos em que o primeiro modelo erra menos e em quantas praças ele tem o
     menor SMAPE médio (todas as origens daquele horizonte);
  5. compara com a versão da Tabela 10 e marca os pares que mudam de conclusão;
  6. mede a dependência: correlação média, entre praças, da diferença de SMAPE de
     cada par na mesma origem; número efetivo de praças independentes; e
     autocorrelação entre semanas consecutivas.

Com 6 e 3 blocos, o menor valor-p possível do Wilcoxon bilateral é 2/2^6 = 0,031 e
2/2^3 = 0,25. Em 90 e 180 dias nenhum par pode ser significativo depois de Holm,
e as conclusões desses horizontes ficam descritivas (itens 4 e 5).

Entrada: resultados/*_metricas_por_origem.csv (os oito modelos)
Saídas:  resultados/testes_blocos_por_origem_pares.csv
         resultados/testes_blocos_por_origem_friedman.csv
         resultados/testes_blocos_por_origem_dependencia.csv

Como executar (segundos):
    python testes_blocos_por_origem.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

RESULTADOS = RAIZ / "resultados"
ORDEM_MODELOS = ["INGENUO", "ARIMA", "PROPHET", "LSTM", "GRU", "BILSTM", "BIGRU", "NHITS"]
ALFA = 0.05


def holm(p):
    """Mesma correção de Holm de resultados_completos.py."""
    p = np.asarray(p, dtype=float)
    ajustado, acumulado = np.empty_like(p), 0.0
    for i, j in enumerate(np.argsort(p)):
        acumulado = max(acumulado, (len(p) - i) * p[j])
        ajustado[j] = min(1.0, acumulado)
    return ajustado


def carregar():
    d = pd.concat([pd.read_csv(a) for a in sorted(RESULTADOS.glob("*_metricas_por_origem.csv"))],
                  ignore_index=True)
    d["ORIGEM"] = pd.to_datetime(d["ORIGEM"])
    modelos = [m for m in ORDEM_MODELOS if m in set(d.MODELO)]
    w = d.pivot_table(index=["PRACA", "DIAS", "ORIGEM"], columns="MODELO", values="SMAPE")[modelos]
    return w, modelos


def blocos(w, h):
    """Blocos sem sobreposição da Tabela 10: uma origem a cada ceil(h / 7)."""
    passo = int(np.ceil(h / config.PASSO_ORIGENS_DIAS))
    wh = w.xs(h, level="DIAS")
    origens = sorted(wh.index.get_level_values("ORIGEM").unique())[::passo]
    return wh[wh.index.get_level_values("ORIGEM").isin(origens)]


def testar(tabela, modelos):
    """Friedman e Wilcoxon-Holm nos 28 pares de uma tabela bloco x modelo."""
    pares = list(combinations(modelos, 2))
    p_bruto = [stats.wilcoxon(tabela[a], tabela[b]).pvalue for a, b in pares]
    return (stats.friedmanchisquare(*[tabela[m].values for m in modelos]).pvalue,
            pares, np.array(p_bruto), holm(p_bruto))


def dependencia(w, modelos, h=7):
    """Correlação entre praças da diferença de SMAPE de cada par, na mesma origem."""
    wh = w.xs(h, level="DIAS")
    linhas = []
    for a, b in combinations(modelos, 2):
        dif = (wh[b] - wh[a]).unstack("PRACA")            # origem x praça
        c = dif.corr().values
        rho = c[np.triu_indices_from(c, k=1)].mean()
        k = dif.shape[1]
        media = dif.mean(axis=1)
        linhas.append({"DIAS": h, "A": a, "B": b, "CORRELACAO_MEDIA_ENTRE_PRACAS": rho,
                       "PRACAS_EFETIVAS": k / (1 + (k - 1) * rho),
                       "AUTOCORRELACAO_SEMANAS_CONSECUTIVAS": media.autocorr(lag=1)})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    w, modelos = carregar()
    pares_saida, friedman_saida = [], []
    for h in sorted(w.index.get_level_values("DIAS").unique()):
        sel = blocos(w, h)                                         # praça x origem (Tabela 10)
        media = sel.groupby(level="ORIGEM").mean()                 # média das praças por origem
        f_antes, pares, _, holm_antes = testar(sel, modelos)
        f_agora, _, bruto_agora, holm_agora = testar(media, modelos)
        n = len(media)
        friedman_saida.append({"DIAS": h, "BLOCOS_TABELA10": len(sel), "BLOCOS_POR_ORIGEM": n,
                               "FRIEDMAN_P_TABELA10": f_antes, "FRIEDMAN_P_POR_ORIGEM": f_agora,
                               "MENOR_P_POSSIVEL_WILCOXON": 2 / 2 ** n if n <= 25 else np.nan})
        # Número de praças em que cada modelo tem menor SMAPE médio (todas as origens do horizonte).
        por_praca = w.xs(h, level="DIAS").groupby(level="PRACA").mean()
        # Descritivo com TODAS as origens do horizonte (87, 75 e 62; janelas sobrepostas em 90 e
        # 180 dias, por isso não entram nos testes): % das origens em que A erra menos que B na
        # média das seis praças.
        todas = w.xs(h, level="DIAS").groupby(level="ORIGEM").mean()
        for (a, b), pb, ph, ph_antes in zip(pares, bruto_agora, holm_agora, holm_antes):
            dif = media[b] - media[a]                              # > 0: A erra menos
            pares_saida.append({
                "DIAS": h, "A": a, "B": b, "BLOCOS": n,
                "SMAPE_A": media[a].mean(), "SMAPE_B": media[b].mean(),
                "DIF_MEDIANA_B_MENOS_A_PP": dif.median(), "DIF_MEDIA_B_MENOS_A_PP": dif.mean(),
                "PCT_BLOCOS_A_MENOR": (media[a] < media[b]).mean() * 100,
                "PRACAS_A_MENOR": int((por_praca[a] < por_praca[b]).sum()),
                "ORIGENS_TODAS": len(todas),
                "PCT_TODAS_ORIGENS_A_MENOR": (todas[a] < todas[b]).mean() * 100,
                "P_WILCOXON": pb, "P_HOLM": ph, "P_HOLM_TABELA10": ph_antes,
                "SIGNIFICATIVO_TABELA10": ph_antes < ALFA, "SIGNIFICATIVO_POR_ORIGEM": ph < ALFA,
            })
    pares_df = pd.DataFrame(pares_saida)
    pares_df["MUDOU"] = pares_df.SIGNIFICATIVO_TABELA10 != pares_df.SIGNIFICATIVO_POR_ORIGEM
    friedman_df = pd.DataFrame(friedman_saida)
    dep = dependencia(w, modelos)
    pares_df.to_csv(RESULTADOS / "testes_blocos_por_origem_pares.csv", index=False, encoding="utf-8-sig")
    friedman_df.to_csv(RESULTADOS / "testes_blocos_por_origem_friedman.csv", index=False, encoding="utf-8-sig")
    dep.to_csv(RESULTADOS / "testes_blocos_por_origem_dependencia.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print("=== Friedman ===")
    print(friedman_df.to_string(index=False))
    print("\n=== Dependência entre praças (7 dias), média dos 28 pares ===")
    print(dep[["CORRELACAO_MEDIA_ENTRE_PRACAS", "PRACAS_EFETIVAS",
               "AUTOCORRELACAO_SEMANAS_CONSECUTIVAS"]].describe().loc[["mean", "min", "max"]].round(2).to_string())
    cols = ["A", "B", "SMAPE_A", "SMAPE_B", "DIF_MEDIANA_B_MENOS_A_PP", "PCT_BLOCOS_A_MENOR",
            "PCT_TODAS_ORIGENS_A_MENOR", "PRACAS_A_MENOR", "P_HOLM_TABELA10", "P_HOLM", "MUDOU"]
    for h in sorted(pares_df.DIAS.unique()):
        print(f"\n=== {h} dias: pares (Holm com {pares_df[pares_df.DIAS == h].BLOCOS.iloc[0]} blocos por origem) ===")
        print(pares_df[pares_df.DIAS == h][cols].round(4).to_string(index=False))
    print(f"\nArquivos em {RESULTADOS}")
