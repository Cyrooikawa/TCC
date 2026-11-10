"""
Junta todos os experimentos numa tabela de "antes e depois" (item 7 da seção 6 do guia).

Para cada variante: SMAPE médio das seis praças antes e depois em 7, 90 e 180 dias,
% de origens em que a nova versão erra menos, SMAPE de 7 dias por tipo de semana,
teste de Wilcoxon pareado (7 dias, janelas sem sobreposição) e o critério que
permite decidir SEM olhar o teste (AIC do treino, validação dentro do treino ou
perda de validação das redes).

Uso:  python consolidar.py
Saída: resultados/RESUMO_experimentos.csv  e  resultados/RESUMO_experimentos.md
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import comum
from comum import config

HS = sorted(config.HORIZONTES.values())
REDES = ["LSTM", "GRU", "BILSTM", "BIGRU", "NHITS"]

DESCRICAO = {
    ("INGENUO", "media4"): "Média das últimas 4 semanas",
    ("INGENUO", "ano_anterior"): "Mesma semana do ano anterior, corrigida pelo crescimento",
    ("SARIMA", "menor_aic"): "Ordem de menor AIC do ajuste final",
    ("SARIMA", "feriados"): "Feriado, véspera e dia seguinte como variáveis externas",
    ("SARIMA", "deriva"): "Termo de deriva (crescimento)",
    ("SARIMA", "fourier1"): "Fourier anual, 1 harmônico",
    ("SARIMA", "fourier2"): "Fourier anual, 2 harmônicos",
    ("SARIMA", "fourier3"): "Fourier anual, 3 harmônicos",
    ("SARIMA", "combinado_menor_aic_feriados"): "COMBINADO: menor AIC + feriados",
    ("SARIMA", "combinado_menor_aic_feriados_fourier2"): "COMBINADO: menor AIC + feriados + Fourier (2)",
    ("PROPHET", "janelas_longas"): "Janela própria para os feriados longos",
    ("PROPHET", "multiplicativo"): "Sazonalidade multiplicativa",
    ("PROPHET", "ferias_escolares"): "Regressor de férias escolares",
    ("PROPHET", "cps0.005"): "changepoint_prior_scale = 0,005 (padrão 0,05)",
    ("PROPHET", "cps0.5"): "changepoint_prior_scale = 0,5 (padrão 0,05)",
    ("PROPHET", "hps1"): "holidays_prior_scale = 1 (padrão 10)",
    ("PROPHET", "hps0.1"): "holidays_prior_scale = 0,1 (padrão 10)",
    ("PROPHET", "combinado_longas_mult"): "COMBINADO: janelas longas + multiplicativo",
    "vies_esquecimento": "Viés da porta de esquecimento = 1",
    "paciencia40": "Paciência 40 e até 400 épocas",
    "janela91": "Janela de entrada de 91 dias no horizonte de 7",
    "unid32": "32 unidades (oficial: 64)",
    "unid128": "128 unidades (oficial: 64)",
    "nhits256": "256 unidades por camada (oficial: 512)",
    "calendario": "Calendário (dia da semana e feriados) como entrada",
}


def p_wilcoxon7(base, novo):
    ch = ["PRACA", "DIAS", "ORIGEM"]
    j = base[ch + ["SMAPE"]].merge(novo[ch + ["SMAPE"]], on=ch, suffixes=("_A", "_D"))
    j = j[j.DIAS == 7]
    if len(j) < 10 or np.allclose(j.SMAPE_A, j.SMAPE_D):
        return np.nan
    return wilcoxon(j.SMAPE_A, j.SMAPE_D).pvalue


def linha(modelo, variante, descricao, base, novo, criterio="", extra=None):
    c = comum.comparar(base, novo)
    r = {"MODELO": modelo, "VARIANTE": variante, "MUDANCA": descricao}
    for h in HS:
        if f"ANTES_{h}" in c:
            r[f"ANTES_{h}"] = c[f"ANTES_{h}"]
            r[f"DEPOIS_{h}"] = c[f"DEPOIS_{h}"]
            r[f"DIF_{h}"] = c[f"DEPOIS_{h}"] - c[f"ANTES_{h}"]
            r[f"PCT_ORIGENS_MELHOR_{h}"] = c[f"PCT_MELHOR_{h}"]
            r[f"PRACAS_MELHOR_{h}"] = c[f"PRACAS_MELHOR_{h}"]
    for t in ("comum", "feriado", "pos_feriado"):
        if f"ANTES_7_{t}" in c:
            r[f"ANTES_7_{t}"] = c[f"ANTES_7_{t}"]
            r[f"DEPOIS_7_{t}"] = c[f"DEPOIS_7_{t}"]
    r["P_WILCOXON_7"] = p_wilcoxon7(base, novo)
    r["CRITERIO_SEM_TESTE"] = criterio
    r.update(extra or {})
    return r


def media_val(arq):
    d = comum.ler(arq)
    return d.groupby(["DIAS", "PRACA"]).SMAPE.mean().groupby("DIAS").mean()


def estatisticos():
    linhas = []
    aic = pd.read_csv(comum.SAIDA / "sarima_aic.csv", encoding="utf-8-sig") if (comum.SAIDA / "sarima_aic.csv").exists() else None
    for modelo, prefixo in (("INGENUO", "ingenuo"), ("SARIMA", "sarima"), ("PROPHET", "prophet")):
        arq_base = comum.SAIDA / f"{prefixo}__oficial.csv"
        if not arq_base.exists():
            continue
        base = comum.ler(arq_base)
        for arq in sorted(comum.SAIDA.glob(f"{prefixo}__*.csv")):
            variante = arq.stem.split("__")[1]
            if variante == "oficial":
                continue
            criterio, extra = "", {}
            if modelo == "SARIMA" and aic is not None:
                a0 = aic[aic.VARIANTE == "oficial"].AIC_TREINO.sum()
                a1 = aic[aic.VARIANTE == variante]
                criterio = f"AIC do treino (soma das 6 praças): {a0:.0f} -> {a1.AIC_TREINO.sum():.0f}"
                extra = {"CRIT_ANTES": a0, "CRIT_DEPOIS": a1.AIC_TREINO.sum(),
                         "OBS": "" if a1.CONVERGIU.all() else
                         "otimizador não convergiu em: " + ", ".join(a1[~a1.CONVERGIU].PRACA)}
            v0, v1 = comum.SAIDA / "validacao" / f"{prefixo}__oficial.csv", comum.SAIDA / "validacao" / arq.name
            if modelo != "SARIMA" and v0.exists() and v1.exists():
                m0, m1 = media_val(v0), media_val(v1)
                criterio = "SMAPE na validação (dentro do treino), 7/90 dias: " + \
                           f"{m0[7]:.2f}/{m0[90]:.2f} -> {m1[7]:.2f}/{m1[90]:.2f}"
                extra = {"CRIT_ANTES": (m0[7] + m0[90]) / 2, "CRIT_DEPOIS": (m1[7] + m1[90]) / 2}
            linhas.append(linha(modelo, variante, DESCRICAO.get((modelo, variante), variante),
                                base, comum.ler(arq), criterio, extra))
    return linhas


def redes():
    linhas = []
    pasta = comum.SAIDA / "redes"
    resumos = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(pasta.glob("*_resumo.json"))]
    combos = sorted({(r["variante"], r["modelo"]) for r in resumos if r["variante"] != "base"})
    for variante, modelo in combos:
        feitos = [r for r in resumos if r["variante"] == variante and r["modelo"] == modelo]
        novo = pd.concat([comum.ler(pasta / f"{variante}_{modelo}_{r['horizonte']}_por_origem.csv") for r in feitos])
        base = comum.ler(comum.OFICIAIS / f"{modelo.lower()}_metricas_por_origem.csv")
        partes, v0s, v1s, obs = [], [], [], []
        for r in sorted(feitos, key=lambda r: r["horizonte"]):
            of = json.loads((comum.OFICIAIS / "redes" / f"{modelo}_{r['horizonte']}_resumo.json").read_text(encoding="utf-8"))
            v0, v1 = np.mean(of["melhor_validacao"]), np.mean(r["melhor_validacao"])
            v0s.append(v0); v1s.append(v1)
            partes.append(f"{r['horizonte']}d: {v0:.4f} -> {v1:.4f}")
            obs.append(f"{r['horizonte']}d: {of['tempo_treino_s']/60:.0f} -> {r['tempo_treino_s']/60:.0f} min")
        criterio = "Perda de validação (MAE normalizado), " + "; ".join(partes)
        if variante == "janela91":
            criterio += " (as janelas de validação não são as mesmas: comparação aproximada)"
        linhas.append(linha(modelo, variante, DESCRICAO.get(variante, variante), base, novo, criterio,
                            {"CRIT_ANTES": float(np.mean(v0s)), "CRIT_DEPOIS": float(np.mean(v1s)),
                             "OBS": "tempo de treino " + "; ".join(obs)}))
    return linhas


def br(x, nd=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}".replace(".", ",")


def veredito(r):
    difs = [r.get(f"DIF_{h}") for h in HS if pd.notna(r.get(f"DIF_{h}"))]
    if all(d <= -0.05 for d in difs):
        v = "melhora"
    elif all(d >= 0.05 for d in difs):
        v = "piora"
    elif all(abs(d) < 0.05 for d in difs):
        v = "sem efeito"
    else:
        v = "misto"
    if pd.notna(r.get("CRIT_ANTES")):
        ok = r["CRIT_DEPOIS"] < r["CRIT_ANTES"]
        v += "; critério sem teste " + ("confirma" if ok == (np.mean(difs) < 0) else "NÃO confirma")
    return v


if __name__ == "__main__":
    t = pd.DataFrame(estatisticos() + redes())
    t["VEREDITO"] = t.apply(veredito, axis=1)
    t.to_csv(comum.SAIDA / "RESUMO_experimentos.csv", index=False, encoding="utf-8-sig")

    out = ["| Modelo | O que mudou | SMAPE 7 d | SMAPE 90 d | SMAPE 180 d | Origens melhores (7/90/180) | "
           "Semana comum / com feriado / seguinte (7 d) | p (7 d) | Resultado |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        cel = lambda h: (f"{br(r[f'ANTES_{h}'])} → {br(r[f'DEPOIS_{h}'])} ({r[f'DIF_{h}']:+.2f})".replace(".", ",")
                         if pd.notna(r.get(f"ANTES_{h}")) else "—")
        pct = " / ".join(f"{r[f'PCT_ORIGENS_MELHOR_{h}']:.0f}%" if pd.notna(r.get(f"PCT_ORIGENS_MELHOR_{h}")) else "—" for h in HS)
        tipos = " / ".join(f"{br(r[f'ANTES_7_{x}'], 1)}→{br(r[f'DEPOIS_7_{x}'], 1)}" for x in ("comum", "feriado", "pos_feriado")) \
            if pd.notna(r.get("ANTES_7_comum")) else "—"
        p = "—" if pd.isna(r["P_WILCOXON_7"]) else ("< 0,001" if r["P_WILCOXON_7"] < 0.001 else br(r["P_WILCOXON_7"], 3))
        out.append(f"| {r.MODELO} | {r.MUDANCA} | {cel(7)} | {cel(90)} | {cel(180)} | {pct} | {tipos} | {p} | {r.VEREDITO} |")
    (comum.SAIDA / "RESUMO_experimentos.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print("\nCritérios sem teste:")
    for _, r in t.iterrows():
        print(f"  {r.MODELO} {r.VARIANTE}: {r.CRITERIO_SEM_TESTE} {r.get('OBS', '') or ''}")
