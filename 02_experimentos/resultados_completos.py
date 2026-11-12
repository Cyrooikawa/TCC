"""
Resultados completos e análises das seções 4.3, 4.4 e 4.5 da monografia,
com todos os modelos combinados com o orientador (ingênuo sazonal, SARIMA,
Prophet, LSTM, GRU, Bi-LSTM, Bi-GRU e N-HiTS).

Lê os resultados por origem de todos os modelos (resultados/*_metricas_por_origem.csv),
as análises de sensibilidade e, se existirem, os arquivos das redes
(resultados/redes/*_por_semente.csv e *_resumo.json). Gera:

  resultados/completos_tabela4_smape_praca.csv     SMAPE por praça, horizonte e modelo
  resultados/completos_tabela7_dtw_praca.csv       DTW por praça, horizonte e modelo
  resultados/completos_tabela8_consolidada.csv     métricas consolidadas (média das praças)
  resultados/completos_tabela9_tipo_semana.csv     SMAPE de 7 dias por tipo de semana
  resultados/completos_tabela10_significancia.csv  Friedman e Wilcoxon-Holm
  resultados/completos_pares_efeito.csv            diferença mediana e vitórias por par
  resultados/completos_wilcoxon_por_praca_7d.csv   testes por praça (7 dias)
  resultados/completos_tabela11_sensibilidade.csv  análises de sensibilidade
  resultados/completos_faixas_antecedencia.csv     SMAPE por faixa (dias 1-7, 8-90, 91-180)
  resultados/completos_redes_sementes.csv          dispersão entre sementes e custo das redes
  figuras/smape_praca_modelo.png                   SMAPE por praça, modelo e horizonte
  figuras/dtw_relativo_praca_modelo.png            DTW relativo ao ingênuo
  figuras/smape7_por_origem.png                    SMAPE de 7 dias ao longo do teste
  figuras/faixas_antecedencia.png                  SMAPE por faixa de antecedência
  figuras/curvas_aprendizado.png                   curvas de treino e validação das redes

Como executar (depois de rodar os modelos e analises_sensibilidade.py):
    python resultados_completos.py
"""

import json
from datetime import timedelta
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.easter import easter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy import stats

import config

RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "resultados"
REDES = RESULTADOS / "redes"
FIGURAS = RAIZ / "figuras"
FIGURAS.mkdir(exist_ok=True)

METRICAS = ["MAE", "RMSE", "MAPE", "SMAPE", "DTW"]
ORD = ["ALAMBARI", "BLOQUEIO DE BOITUVA", "BOITUVA", "PORTO FELIZ", "ITU", "SOROCABA"]
assert set(ORD) == set(config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS), "praças do config.py mudaram: revisar ORD"
NM = {"ALAMBARI": "Alambari", "BLOQUEIO DE BOITUVA": "Bloqueio de Boituva", "BOITUVA": "Boituva",
      "PORTO FELIZ": "Porto Feliz", "ITU": "Itu", "SOROCABA": "Sorocaba"}
GRUPO = {p: ("núcleo" if p in config.PRACAS_NUCLEO else "unidirecional") for p in ORD}

ORDEM_MODELOS = ["INGENUO", "ARIMA", "PROPHET", "LSTM", "GRU", "BILSTM", "BIGRU", "NHITS"]
REDES_NEURAIS = ["LSTM", "GRU", "BILSTM", "BIGRU", "NHITS"]
ROTULO = {"INGENUO": "Ingênuo sazonal", "ARIMA": "SARIMA", "PROPHET": "Prophet", "LSTM": "LSTM",
          "GRU": "GRU", "BILSTM": "Bi-LSTM", "BIGRU": "Bi-GRU", "NHITS": "N-HiTS"}
# Cor por família (paleta validada para daltonismo): a versão bidirecional usa a
# cor da unidirecional com linha pontilhada; o ingênuo, referência, fica em cinza.
TRACO, PONTO = (0, (4, 2)), (0, (1.2, 1.6))
ESTILO = {"INGENUO": ("#7a7a7a", TRACO, "s"), "ARIMA": ("#eb6834", "-", "^"), "PROPHET": ("#2a78d6", "-", "o"),
          "LSTM": ("#1baf7a", "-", "D"), "BILSTM": ("#1baf7a", PONTO, "d"), "GRU": ("#6250d6", "-", "v"),
          "BIGRU": ("#6250d6", PONTO, "P"), "NHITS": ("#e87ba4", "-", "X")}
EXTRAS = iter([("#eda100", "-", "h"), ("#008300", "-", "*"), ("#e34948", "-", "p")])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "text.color": "#0b0b0b"})
W = 6.3  # largura em polegadas, a mesma das demais figuras
VIRGULA = FuncFormatter(lambda v, _: (f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}").replace(".", ","))


def br(v, nd=1):
    return f"{v:.{nd}f}".replace(".", ",")


def carregar():
    arquivos = sorted(RESULTADOS.glob("*_metricas_por_origem.csv"))
    d = pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)
    d["ORIGEM"] = pd.to_datetime(d["ORIGEM"])
    presentes = set(d.MODELO.unique())
    modelos = [m for m in ORDEM_MODELOS if m in presentes] + sorted(presentes - set(ORDEM_MODELOS))
    for m in modelos:
        if m not in ESTILO:
            ESTILO[m] = next(EXTRAS)
            ROTULO.setdefault(m, m)
    # Todos os modelos precisam ter as mesmas praças e origens (comparação pareada).
    contagem = d.groupby(["MODELO", "DIAS"]).size().unstack()
    assert (contagem.nunique() == 1).all(), f"modelos com origens diferentes:\n{contagem}"
    return d, modelos


def linha(ax, x, y, m, **kw):
    cor, ls, mk = ESTILO[m]
    base = dict(color=cor, ls=ls, marker=mk, label=ROTULO[m], ms=4.2, mec="white", mew=0.6,
                lw=1.5 if m == "PROPHET" else 1.15, zorder=3 if m == "PROPHET" else 2)
    base.update(kw)
    return ax.plot(x, y, **base)


def limpar(ax):
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def tabela_por_praca(d, modelos, metrica):
    t = d.groupby(["PRACA", "DIAS", "MODELO"])[metrica].mean().unstack()[modelos]
    return t.reindex(pd.MultiIndex.from_product([ORD, sorted(d.DIAS.unique())], names=["PRACA", "DIAS"]))


def tabela8(d, modelos):
    """Médias das praças (SMAPE, MAPE) e erros relativos ao ingênuo (média geométrica)."""
    med = d.groupby(["PRACA", "DIAS", "MODELO"])[METRICAS].mean()
    w = d.pivot_table(index=["PRACA", "DIAS", "ORIGEM"], columns="MODELO", values="SMAPE")[modelos]
    postos = w.rank(axis=1)
    linhas = []
    for h in sorted(d.DIAS.unique()):
        wh, ph = w.xs(h, level="DIAS"), postos.xs(h, level="DIAS")
        for m in modelos:
            sm = [med.loc[(p, h, m), "SMAPE"] for p in ORD]
            rel = {k: np.exp(np.mean([np.log(med.loc[(p, h, m), k] / med.loc[(p, h, "INGENUO"), k]) for p in ORD]))
                   for k in ["MAE", "RMSE", "DTW"]}
            linhas.append({
                "DIAS": h, "MODELO": m,
                "SMAPE": np.mean(sm), "SMAPE_MIN": min(sm), "SMAPE_MAX": max(sm),
                "MAPE": np.mean([med.loc[(p, h, m), "MAPE"] for p in ORD]),
                "MAE_REL": rel["MAE"], "RMSE_REL": rel["RMSE"], "DTW_REL": rel["DTW"],
                "PCT_VENCE_INGENUO": np.nan if m == "INGENUO" else (wh[m] < wh["INGENUO"]).mean() * 100,
                "POSTO_MEDIO": ph[m].mean(), "PCT_MELHOR": (ph[m] == 1).mean() * 100,
                "SMAPE_MEDIANA": wh[m].median(), "SMAPE_P90": wh[m].quantile(0.9),
                "RMSE_SOBRE_MAE": np.mean([med.loc[(p, h, m), "RMSE"] / med.loc[(p, h, m), "MAE"] for p in ORD]),
                "N_PRACAS_MELHOR_SMAPE": sum(
                    med.loc[(p, h, m), "SMAPE"] <= min(med.loc[(p, h, k), "SMAPE"] for k in modelos) for p in ORD),
            })
    return pd.DataFrame(linhas)


def datas_de_feriado(inicio, fim):
    """Mesmo calendário do Prophet: feriados nacionais e de SP, Carnaval e Corpus Christi."""
    anos = range(inicio.year, fim.year + 1)
    try:
        import holidays
        datas = [pd.Timestamp(x) for x in holidays.Brazil(subdiv="SP", years=anos)]
    except ImportError:
        print("Aviso: biblioteca holidays não encontrada; usando a lista fixa de 2024 a 2026.")
        datas = list(pd.to_datetime([
            "2024-11-15", "2024-11-20", "2024-12-25", "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01",
            "2025-07-09", "2025-09-07", "2025-10-12", "2025-11-02", "2025-11-15", "2025-11-20", "2025-12-25",
            "2026-01-01", "2026-04-03", "2026-04-21", "2026-05-01", "2026-07-09"]))
    for ano in anos:
        pascoa = pd.Timestamp(easter(ano))
        datas += [pascoa - timedelta(days=48), pascoa - timedelta(days=47), pascoa + timedelta(days=60)]
    # Dias afetados: 2 antes e 1 depois, a mesma janela usada no Prophet.
    return {f + timedelta(days=k) for f in datas for k in range(-2, 2)}


def tabela9(d, modelos):
    """SMAPE de 7 dias por tipo de semana: comum, com feriado e seguinte a feriado."""
    d7 = d[d.DIAS == 7].copy()
    afetados = datas_de_feriado(d7.ORIGEM.min() - timedelta(days=7), d7.ORIGEM.max() + timedelta(days=8))
    tem = lambda ini, n: any(ini + timedelta(days=i) in afetados for i in range(n))
    cat = {o: ("feriado no período previsto" if tem(o + timedelta(days=1), 7)
               else "semana seguinte a feriado" if tem(o - timedelta(days=6), 7) else "semana comum")
           for o in d7.ORIGEM.unique()}
    d7["TIPO"] = d7.ORIGEM.map(cat)
    w = d7.pivot_table(index=["PRACA", "ORIGEM", "TIPO"], columns="MODELO", values="SMAPE").reset_index()
    linhas = []
    for tipo, g in list(w.groupby("TIPO")) + [("todas", w)]:
        linha_ = {"TIPO": tipo, "JANELAS": len(g), "ORIGENS": g.ORIGEM.nunique()}
        for m in modelos:
            linha_[f"SMAPE_{m}"] = g[m].mean()
            if m != "INGENUO":
                linha_[f"PCT_{m}_VENCE_INGENUO"] = (g[m] < g.INGENUO).mean() * 100
            linha_[f"PARTICIPACAO_ERRO_{m}"] = g[m].sum() / w[m].sum() * 100
        linhas.append(linha_)
    return pd.DataFrame(linhas), d7


def holm(p):
    p = np.asarray(p, dtype=float)
    ajustado, acumulado = np.empty_like(p), 0.0
    for i, j in enumerate(np.argsort(p)):
        acumulado = max(acumulado, (len(p) - i) * p[j])
        ajustado[j] = min(1.0, acumulado)
    return ajustado


def blocos_sem_sobreposicao(w, h):
    passo = int(np.ceil(h / config.PASSO_ORIGENS_DIAS))  # 1, 13 e 26 origens
    return pd.concat([w.xs((p, h), level=["PRACA", "DIAS"]).sort_index().iloc[::passo].assign(PRACA=p)
                      for p in ORD])


def tabela10(d, modelos):
    """Friedman e Wilcoxon-Holm sobre o SMAPE, só com janelas sem sobreposição."""
    w = d.pivot_table(index=["PRACA", "DIAS", "ORIGEM"], columns="MODELO", values="SMAPE")[modelos]
    linhas, efeitos = [], []
    for h in sorted(d.DIAS.unique()):
        sel = blocos_sem_sobreposicao(w, h)
        linha_ = {"DIAS": h, "BLOCOS": len(sel),
                  "FRIEDMAN_P": stats.friedmanchisquare(*[sel[m].values for m in modelos]).pvalue}
        pares = list(combinations(modelos, 2))
        p_bruto = [stats.wilcoxon(sel[a], sel[b]).pvalue for a, b in pares]
        p_holm = holm(p_bruto)
        linha_.update({f"WILCOXON_HOLM_{a}_x_{b}": p for (a, b), p in zip(pares, p_holm)})
        linha_.update({f"POSTO_{m}": v for m, v in sel[modelos].rank(axis=1).mean().items()})
        media = sel.reset_index().groupby("ORIGEM")[modelos].mean()
        linha_["FRIEDMAN_P_MEDIA_DAS_PRACAS"] = stats.friedmanchisquare(*[media[m].values for m in modelos]).pvalue
        linhas.append(linha_)
        for (a, b), pb, ph in zip(pares, p_bruto, p_holm):
            dif = sel[b] - sel[a]
            efeitos.append({"DIAS": h, "A": a, "B": b, "BLOCOS": len(sel),
                            "DIF_MEDIANA_B_MENOS_A_PP": dif.median(), "DIF_MEDIA_B_MENOS_A_PP": dif.mean(),
                            "PCT_BLOCOS_A_MENOR": (sel[a] < sel[b]).mean() * 100,
                            "P_WILCOXON": pb, "P_HOLM": ph})
    por_praca = []
    for p in ORD:
        wp = w.xs((p, 7), level=["PRACA", "DIAS"])
        for a, b in combinations(modelos, 2):
            por_praca.append({"PRACA": p, "PAR": f"{a} x {b}", "DIF_MEDIA_SMAPE": (wp[b] - wp[a]).mean(),
                              "P": stats.wilcoxon(wp[a], wp[b]).pvalue})
    por_praca = pd.DataFrame(por_praca)
    por_praca["P_HOLM"] = holm(por_praca.P.values)
    return pd.DataFrame(linhas), pd.DataFrame(efeitos), por_praca


def faixas(d, modelos):
    """SMAPE por faixa de antecedência, nas origens com os três horizontes.

    Estatísticos: Equações 7 e 8, válidas porque as janelas de 7, 90 e 180 dias saem
    da mesma trajetória de previsão. Redes: cálculo direto da previsão diária da rede
    de 180 dias (faixas_antecedencia_direto.py), porque cada horizonte tem a sua rede
    e as equações misturariam três modelos (validação de 02/11/2026, ponto 1).
    """
    from faixas_antecedencia_direto import FAIXAS, montar
    f = montar(d)
    colunas = list(FAIXAS)
    media = f.groupby("MODELO")[colunas].mean()
    return (f.groupby(["PRACA", "MODELO"])[colunas].mean().reset_index(),
            media.reindex([m for m in modelos if m in media.index]))


def tabela11(d):
    extras = [RESULTADOS / "sensibilidade_arima_sem_sazonalidade.csv",
              RESULTADOS / "sensibilidade_prophet_ajuste_unico.csv",
              RESULTADOS / "sensibilidade_prophet_sem_feriados.csv",
              RESULTADOS / "sensibilidade_prophet_sem_feriados_sem_anual.csv"]
    partes = [d] + [pd.read_csv(a) for a in extras if a.exists()]
    todos = pd.concat(partes, ignore_index=True)
    return todos.groupby(["DIAS", "MODELO"]).SMAPE.mean().unstack()


def redes_sementes():
    """Dispersão entre as cinco sementes e custo de treino de cada rede."""
    if not REDES.exists():
        return None, None
    linhas, curvas = [], {}
    for arq in sorted(REDES.glob("*_resumo.json")):
        r = json.loads(arq.read_text(encoding="utf-8"))
        m, h = r["modelo"], r["horizonte"]
        sem = pd.read_csv(REDES / f"{m}_{h}_por_semente.csv")
        por_semente = sem.groupby(["SEMENTE", "PRACA"]).SMAPE.mean().groupby("SEMENTE").mean()
        linhas.append({"MODELO": m, "DIAS": h, "SMAPE_MEDIA_SEMENTES": por_semente.mean(),
                       "SMAPE_DP_SEMENTES": por_semente.std(ddof=1), "SMAPE_MIN_SEMENTE": por_semente.min(),
                       "SMAPE_MAX_SEMENTE": por_semente.max(), "EPOCAS_EXECUTADAS": r["epocas_executadas"],
                       "MELHOR_EPOCA_MEDIA": np.mean(r["melhor_epoca"]), "MELHOR_EPOCA_MIN": min(r["melhor_epoca"]),
                       "MELHOR_EPOCA_MAX": max(r["melhor_epoca"]), "TEMPO_TREINO_MIN": r["tempo_treino_s"] / 60,
                       "PARAMETROS_POR_SEMENTE": r["n_parametros_por_semente"],
                       "JANELAS_TREINO": sum(j["treino"] for j in r["dados"]["janelas"]),
                       "JANELAS_VALIDACAO": sum(j["validacao"] for j in r["dados"]["janelas"]),
                       "VALIDACAO_MELHOR_MEDIA": np.mean(r["melhor_validacao"])})
        hist = r["historico"]
        curvas[(m, h)] = (np.array([e["epoca"] for e in hist]), np.array([e["treino"] for e in hist]),
                          np.array([e["validacao"] for e in hist]), np.array(r["melhor_epoca"]))
    t = pd.DataFrame(linhas)
    t["ORD"] = t.MODELO.map({m: i for i, m in enumerate(ORDEM_MODELOS)})
    return t.sort_values(["DIAS", "ORD"]).drop(columns="ORD"), curvas


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def figura_mapas(t4, modelos, nome, rotulo_barra, relativo=False):
    """Mapas de calor praça x modelo, um por horizonte, com o valor escrito em cada célula."""
    if relativo:
        modelos = [m for m in modelos if m != "INGENUO"]  # sempre 1
    horizontes = sorted(t4.index.get_level_values("DIAS").unique())
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.9), sharey=True)
    if relativo:
        cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f4f3f0", "#eb6834"])
        norm = TwoSlopeNorm(vcenter=1.0, vmin=0.3, vmax=1.7)
    else:
        cmap = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#9cc0ec", "#2a78d6", "#123e78"])
        vals = t4.values[np.isfinite(t4.values)]
        norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
    for ax, h in zip(axes, horizontes):
        M = t4.xs(h, level="DIAS").loc[ORD, modelos].values
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            linha_min = np.nanmin(M[i]) if not relativo else None
            for j in range(M.shape[1]):
                v = M[i, j]
                fundo = cmap(norm(v))
                escuro = (0.299 * fundo[0] + 0.587 * fundo[1] + 0.114 * fundo[2]) < 0.55
                txt = br(v, 2 if relativo else 1)
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.4,
                        color="white" if escuro else "#0b0b0b",
                        fontweight="bold" if (not relativo and v == linha_min) else "normal")
        ax.set_xticks(range(len(modelos)))
        ax.set_xticklabels([ROTULO[m] for m in modelos], rotation=60, ha="right", fontsize=6.8)
        ax.set_yticks(range(len(ORD)))
        ax.set_yticklabels([NM[p] for p in ORD], fontsize=7)
        ax.set_title(f"{h} dias", fontsize=8.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.5, 0.012, rotulo_barra, ha="center", fontsize=7, color="#52514e")
    fig.subplots_adjust(left=0.165, right=0.995, top=0.92, bottom=0.30, wspace=0.06)
    fig.savefig(FIGURAS / nome, dpi=220)
    plt.close(fig)


def figura10(d7, modelos, cor=None):
    p = d7.groupby(["MODELO", "ORIGEM"]).SMAPE.mean().reset_index()
    estat = [m for m in modelos if m not in REDES_NEURAIS]
    redes = [m for m in modelos if m in REDES_NEURAIS]
    paineis = [("Modelos estatísticos", estat)] + ([("Redes neurais", ["PROPHET"] + redes)] if redes else [])
    fig, axes = plt.subplots(len(paineis), 1, figsize=(W, 2.25 * len(paineis) + 0.55), sharex=True, squeeze=False)
    periodos = [("2024-12-22", "2025-01-02", "Natal e\nAno-Novo"), ("2025-03-01", "2025-03-05", "Carnaval"),
                ("2025-04-16", "2025-04-22", "Páscoa e\nTiradentes"), ("2025-12-23", "2026-01-02", "Natal e\nAno-Novo"),
                ("2026-02-14", "2026-02-18", "Carnaval"), ("2026-04-01", "2026-04-05", "Páscoa")]
    ymax = max(27.5, np.ceil(p.SMAPE.max() / 5) * 5 + 2.5)
    for k, ((titulo, grupo), ax) in enumerate(zip(paineis, axes[:, 0])):
        for a, b, txt in periodos:
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            ax.axvspan(a, b + timedelta(days=1), color="#efeee9", lw=0, zorder=0)
            if k == 0:
                ax.text(a + (b - a) / 2, ymax * (0.86 if txt == "Carnaval" else 0.995), txt, ha="center", va="top",
                        fontsize=6.4, color="#52514e", linespacing=1.05)
        for m in grupo:
            x = p[p.MODELO == m].sort_values("ORIGEM")
            cor_, ls, _ = ESTILO[m]
            ref = (m == "PROPHET" and k > 0)
            ax.plot(x.ORIGEM + timedelta(days=1), x.SMAPE, color=cor_, ls=ls, label=ROTULO[m],
                    lw=1.0 if ref else (1.5 if m == "PROPHET" else 1.05), alpha=0.55 if ref else 1,
                    zorder=3 if m == "PROPHET" and not ref else 2, solid_joinstyle="round")
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_formatter(VIRGULA)
        ax.set_title(titulo, loc="left", fontsize=8)
        ax.set_ylabel("SMAPE (%)")
        limpar(ax)
        ax.legend(loc="lower right", ncol=len(grupo), frameon=False, fontsize=6.8, bbox_to_anchor=(1.0, 0.99),
                  handlelength=1.9, columnspacing=1.0, borderaxespad=0.1)
    ax = axes[-1, 0]
    ax.set_xlim(pd.Timestamp("2024-11-20"), pd.Timestamp("2026-08-01"))
    ax.set_xlabel("Primeiro dia da semana prevista")
    meses = {1: "jan", 4: "abr", 7: "jul", 10: "out"}
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{meses[mdates.num2date(v).month]}/{mdates.num2date(v).year}"))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09 if len(paineis) > 1 else 0.17, hspace=0.26)
    fig.savefig(FIGURAS / "smape7_por_origem.png", dpi=220)
    plt.close(fig)


def figura11(fx_media, modelos):
    colunas, xs = ["DIAS_1_7", "DIAS_8_90", "DIAS_91_180"], np.arange(3)
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for m in modelos:
        linha(ax, xs, fx_media.loc[m, colunas].values, m)
    # Rótulos à direita, afastados para não se sobreporem (mínimo de 4% da faixa do eixo).
    ys = fx_media.loc[modelos, "DIAS_91_180"].astype(float)
    faixa = float(fx_media[colunas].values.max() - fx_media[colunas].values.min())
    gap, pos = 0.045 * faixa, {}
    for m in ys.sort_values().index:
        alvo = ys[m]
        if pos:
            alvo = max(alvo, max(pos.values()) + gap)
        pos[m] = alvo
    excesso = max(0.0, max(pos.values()) - ys.max())
    for m in pos:
        y = pos[m] - excesso / 2 if excesso else pos[m]
        ax.annotate(ROTULO[m], (2.0, ys[m]), xytext=(2.1, y), va="center", fontsize=6.8, color="#0b0b0b",
                    arrowprops=dict(arrowstyle="-", color="#b3b2ad", lw=0.5, shrinkA=0, shrinkB=3))
    ax.set_xticks(xs)
    ax.set_xticklabels(["1 a 7", "8 a 90", "91 a 180"])
    ax.set_xlim(-0.15, 2.55)
    ax.set_xlabel("Dias à frente da origem")
    ax.set_ylabel("SMAPE médio das seis praças (%)")
    ax.yaxis.set_major_formatter(VIRGULA)
    limpar(ax)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.15)
    fig.savefig(FIGURAS / "faixas_antecedencia.png", dpi=220)
    plt.close(fig)


def figura_curvas(curvas):
    if not curvas:
        return
    redes = [m for m in ORDEM_MODELOS if any(k[0] == m for k in curvas)]
    horizontes = sorted({k[1] for k in curvas})
    fig, axes = plt.subplots(len(redes), len(horizontes), figsize=(W, 1.25 * len(redes) + 0.6),
                             sharex=False, squeeze=False)
    for i, m in enumerate(redes):
        for j, h in enumerate(horizontes):
            ax = axes[i, j]
            if (m, h) not in curvas:
                ax.axis("off")
                continue
            ep, tr, va, melhor = curvas[(m, h)]
            cor_ = ESTILO[m][0]
            ax.plot(ep, tr.mean(axis=1), color="#52514e", lw=0.9, ls=TRACO, label="treino (média das sementes)")
            for s_ in range(va.shape[1]):
                ax.plot(ep, va[:, s_], color=cor_, lw=0.8, alpha=0.75,
                        label="validação (uma linha por semente)" if s_ == 0 else None)
                k = int(melhor[s_]) - 1
                ax.plot(ep[k], va[k, s_], "o", ms=3.2, color=cor_, mec="white", mew=0.5, zorder=4,
                        label="época escolhida" if s_ == 0 else None)
            ax.set_ylim(0.03, 0.12)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
            ax.tick_params(labelsize=6.3)
            limpar(ax)
            if i == 0:
                ax.set_title(f"{h} dias", fontsize=8)
            if j == 0:
                ax.set_ylabel(ROTULO[m], fontsize=7.5)
            if i == len(redes) - 1:
                ax.set_xlabel("Época", fontsize=7)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper right", ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.99, 1.0))
    fig.text(0.01, 0.004, "Perda MAE na escala normalizada; o ponto marca, em cada semente, a época de menor "
             "erro de validação antes da parada antecipada.", fontsize=6.4, va="bottom", color="#52514e")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.085, hspace=0.45, wspace=0.22)
    fig.savefig(FIGURAS / "curvas_aprendizado.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    d, modelos = carregar()
    print("Modelos encontrados:", ", ".join(modelos))

    t4 = tabela_por_praca(d, modelos, "SMAPE")
    t4.to_csv(RESULTADOS / "completos_tabela4_smape_praca.csv", encoding="utf-8-sig")
    t7 = tabela_por_praca(d, modelos, "DTW")
    t7.to_csv(RESULTADOS / "completos_tabela7_dtw_praca.csv", encoding="utf-8-sig")
    t7_rel = t7.div(t7["INGENUO"], axis=0)

    t8 = tabela8(d, modelos)
    t8.to_csv(RESULTADOS / "completos_tabela8_consolidada.csv", index=False, encoding="utf-8-sig")
    print("\nSMAPE médio das seis praças:")
    print(t8.pivot(index="MODELO", columns="DIAS", values="SMAPE").reindex(modelos).round(2).to_string())

    t9, d7 = tabela9(d, modelos)
    t9.to_csv(RESULTADOS / "completos_tabela9_tipo_semana.csv", index=False, encoding="utf-8-sig")

    t10, efeitos, por_praca = tabela10(d, modelos)
    t10.to_csv(RESULTADOS / "completos_tabela10_significancia.csv", index=False, encoding="utf-8-sig")
    efeitos.to_csv(RESULTADOS / "completos_pares_efeito.csv", index=False, encoding="utf-8-sig")
    por_praca.to_csv(RESULTADOS / "completos_wilcoxon_por_praca_7d.csv", index=False, encoding="utf-8-sig")

    t11 = tabela11(d)
    t11.to_csv(RESULTADOS / "completos_tabela11_sensibilidade.csv", encoding="utf-8-sig")

    fx, fx_media = faixas(d, modelos)
    fx.to_csv(RESULTADOS / "completos_faixas_antecedencia.csv", index=False, encoding="utf-8-sig")
    print("\nFaixas de antecedência (média das praças):")
    print(fx_media.round(2).to_string())

    sementes, curvas = redes_sementes()
    if sementes is not None:
        sementes.to_csv(RESULTADOS / "completos_redes_sementes.csv", index=False, encoding="utf-8-sig")

    figura_mapas(t4, modelos, "smape_praca_modelo.png", "SMAPE médio (%); em negrito, o menor de cada praça")
    figura_mapas(t7_rel, modelos, "dtw_relativo_praca_modelo.png",
                 "DTW dividido pelo DTW do ingênuo sazonal (abaixo de 1 = forma mais próxima da real)", relativo=True)
    figura10(d7, modelos)
    figura11(fx_media, list(fx_media.index))
    figura_curvas(curvas)
    print(f"\nTabelas em {RESULTADOS} e figuras em {FIGURAS}")
