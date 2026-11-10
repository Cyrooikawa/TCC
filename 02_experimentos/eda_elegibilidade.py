"""
Análise exploratória de elegibilidade das praças (complementa a seção 4.2).

Aplica os critérios de config.py a TODAS as praças físicas da RMS encontradas nos
arquivos brutos (auditar_dados_brutos.py), mais Morro I Norte e Morro II Sul
(SPVias, SP-127, entre Tatuí e Itapetininga), cujo nome não contém o município e
que por isso a identificação automática não reconhece. Verifica também as
anomalias declaradas no trabalho.

Uso (depois de auditar_dados_brutos.py):
    python eda_elegibilidade.py <pasta_com_os_arquivos_brutos>
Saídas (pasta auditoria/):
    elegibilidade_pracas.csv, eda_elegibilidade.txt,
    figura_cobertura_pracas.png, figura_anomalias.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402
from tratamento import normalizar  # noqa: E402

AUD = RAIZ / "auditoria"
LOG = []
INICIO, FIM_TREINO = pd.Timestamp(config.DATA_INICIO_MODELAGEM), pd.Timestamp("2024-11-26")


def log(*a):
    t = " ".join(str(x) for x in a)
    print(t)
    LOG.append(t)


def ler_morro(pasta):
    partes = []
    for a in sorted(pasta.rglob("VOLUME_PEDAGIADO-L20-DIARIO-*.csv")):
        df = pd.read_csv(a, dtype=str, low_memory=False)
        df["PRACA"] = df["PRACA"].map(normalizar)
        df = df[df.PRACA.str.startswith("MORRO")]
        eixos = [c for c in df.columns if "EIXOS" in c.upper() and int(c.split(" ")[-1]) <= 10]
        df["VOL"] = df[eixos].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        df["DATA"] = pd.to_datetime(df["DATA"])
        ano = int(a.stem.split("-")[-1])
        partes.append(df[df.DATA.dt.year == ano][["PRACA", "DATA", "SENTIDO", "VOL"]])
    m = pd.concat(partes)
    return m.groupby(["PRACA", "DATA"]).VOL.sum().reset_index(), m.groupby("PRACA").SENTIDO.agg(lambda s: "/".join(sorted(set(s))))


def quebra_de_nivel(s):
    """Maior razão entre a mediana de 8 semanas depois e antes de cada data, corrigida pela mesma
    razão um ano antes (tira a sazonalidade anual). Valores longe de 1 indicam quebra."""
    semanal = s.resample("W").mean().dropna()
    razoes = []
    for i in range(60, len(semanal) - 8):
        antes, depois = semanal.iloc[i - 8:i].median(), semanal.iloc[i:i + 8].median()
        a1, d1 = semanal.iloc[i - 60:i - 52].median(), semanal.iloc[i - 52:i - 44].median()
        if min(antes, depois, a1, d1) > 0:
            razoes.append((semanal.index[i], (depois / antes) / (d1 / a1)))
    if not razoes:
        return np.nan, None
    r = pd.Series(dict(razoes))
    k = (np.log(r)).abs().idxmax()
    return float(r[k]), k


if __name__ == "__main__":
    pasta = Path(sys.argv[1])
    inv = pd.read_csv(AUD / "inventario_pracas_rms.csv")
    diario = pd.read_csv(AUD / "diario_pracas_rms.csv", parse_dates=["DATA"])
    morro, sent_morro = ler_morro(pasta)
    diario = pd.concat([diario, morro], ignore_index=True)
    fisicas = inv[inv.REGIME == "PRACA FISICA"].copy()
    fisicas = fisicas[~fisicas.PRACA.str.match(r"^(SAO (CARLOS|SIMAO|VICENTE|JOAO|JOSE)|P0[78] - SAO PEDRO)")]  # falsos positivos
    candidatos = list(fisicas.PRACA) + ["MORRO I NORTE", "MORRO II SUL"]
    sentidos = dict(zip(fisicas.PRACA, fisicas.SENTIDOS)) | sent_morro.to_dict()
    lote = dict(zip(fisicas.PRACA, fisicas.LOTES)) | {"MORRO I NORTE": "L20-SPVIAS", "MORRO II SUL": "L20-SPVIAS"}

    linhas = []
    for p in candidatos:
        s = diario[diario.PRACA == p].set_index("DATA").VOL.sort_index()
        s = s[s.index >= INICIO]
        cal = pd.date_range(INICIO, max(s.index.max(), INICIO))
        fim = s.index.max()
        ausentes = len(cal) - len(s)
        invalid = sum(((s.index >= i) & (s.index <= f)).sum() for i, f in config.PERIODOS_INVALIDOS.get(p, []))
        teste = s[s.index > FIM_TREINO]
        dias_teste_esp = len(pd.date_range(FIM_TREINO + pd.Timedelta(days=1), fim)) if fim > FIM_TREINO else 0
        r, quando = quebra_de_nivel(s.drop(s.index[(s.index >= "2022-10-01") & (s.index <= "2022-11-30")], errors="ignore"))
        semana = s.groupby(s.index.dayofweek).mean()
        linhas.append({
            "PRACA": p, "LOTE": lote[p], "SENTIDOS": sentidos.get(p, ""),
            "DIRECIONALIDADE": "BIDIRECIONAL" if "/" in sentidos.get(p, "") else "UNIDIRECIONAL",
            "FIM_DA_SERIE": fim.date(), "COBRE_ATE_2026": fim >= pd.Timestamp("2026-05-31"),
            "DIAS_AUSENTES_DESDE_2021": ausentes, "DIAS_INVALIDADOS": invalid,
            "LACUNAS_NO_TESTE": dias_teste_esp - len(teste),
            "MEDIA_DIARIA_2024": s[s.index.year == 2024].mean(),
            "CRESC_ANUAL_2021_2025_PCT": ((s[s.index.year == 2025].mean() / s[s.index.year == 2021].mean()) ** 0.25 - 1) * 100
            if (s.index.year == 2025).any() else np.nan,
            "AMPLITUDE_SEMANAL_PCT": (semana.max() - semana.min()) / semana.mean() * 100,
            "MAIOR_QUEBRA_DE_NIVEL": r, "DATA_DA_QUEBRA": quando.date() if quando is not None else None,
        })
    el = pd.DataFrame(linhas)

    def decisao(r):
        if not r.COBRE_ATE_2026:
            return "fora: série termina em " + str(r.FIM_DA_SERIE)
        if abs(np.log(r.MAIOR_QUEBRA_DE_NIVEL)) > np.log(1.25):
            return f"verificar: possível quebra de nível em {r.DATA_DA_QUEBRA}"
        if r.LACUNAS_NO_TESTE > 0:
            return "fora: lacunas no teste"
        return ("elegível, grupo principal (bidirecional)" if r.DIRECIONALIDADE == "BIDIRECIONAL"
                else "elegível, grupo unidirecional")
    el["AVALIACAO"] = el.apply(decisao, axis=1)
    el["NO_TCC"] = el.PRACA.isin(config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS)
    el.to_csv(AUD / "elegibilidade_pracas.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    log("Elegibilidade das praças físicas da RMS (desde 2021; brutos até 31/05/2026)")
    log(el[["PRACA", "LOTE", "SENTIDOS", "FIM_DA_SERIE", "DIAS_AUSENTES_DESDE_2021", "DIAS_INVALIDADOS", "LACUNAS_NO_TESTE",
            "MEDIA_DIARIA_2024", "CRESC_ANUAL_2021_2025_PCT", "AMPLITUDE_SEMANAL_PCT", "MAIOR_QUEBRA_DE_NIVEL",
            "DATA_DA_QUEBRA", "NO_TCC", "AVALIACAO"]].round(2).to_string(index=False))

    # Anomalias declaradas
    log("\nOutubro de 2022 (lote L13): média diária de outubro / média de setembro e dezembro de 2022")
    for p in ["BOITUVA", "BLOQUEIO DE BOITUVA", "PORTO FELIZ", "ALAMBARI"]:
        s = diario[diario.PRACA == p].set_index("DATA").VOL
        out = s["2022-10-01":"2022-10-31"].mean()
        ref = pd.concat([s["2022-09-01":"2022-09-30"], s["2022-12-01":"2022-12-31"]]).mean()
        nov = len(s["2022-11-01":"2022-11-30"])
        out21 = s["2021-10-01":"2021-10-31"].mean() / pd.concat([s["2021-09-01":"2021-09-30"], s["2021-12-01":"2021-12-31"]]).mean()
        log(f"  {p:20s} razão out/2022 = {out / ref:.2f} (em 2021: {out21:.2f}); dias com dado em nov/2022: {nov}")
    log("\nPandemia (2020): queda de abril de 2020 em relação a abril de 2021")
    for p in config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS:
        s = diario[diario.PRACA == p].set_index("DATA").VOL
        log(f"  {p:20s} abr/2020 ÷ abr/2021 = {s['2020-04'].mean() / s['2021-04'].mean():.2f}")
    log("\nTroca de lote em 2025 (L12-ViaOeste para L33-Rota Sorocabana): nível nas 8 semanas antes e depois")
    brutos = pd.read_pickle(AUD / "cache_linhas_rms.pkl")
    for p in ["ITU", "SOROCABA"]:
        b = brutos[brutos.PRACA == p]
        troca = b[b.LOTE.str.contains("L33")].DATA.min()
        s = diario[diario.PRACA == p].set_index("DATA").VOL
        antes = s[troca - pd.Timedelta(days=56): troca - pd.Timedelta(days=1)].mean()
        depois = s[troca: troca + pd.Timedelta(days=55)].mean()
        a1 = s[troca - pd.Timedelta(days=56 + 365): troca - pd.Timedelta(days=366)].mean()
        d1 = s[troca - pd.Timedelta(days=365): troca + pd.Timedelta(days=55 - 365)].mean()
        log(f"  {p:9s} primeiro dia no L33: {troca.date()}; depois ÷ antes = {depois / antes:.3f} "
            f"(mesmas datas um ano antes: {d1 / a1:.3f})")
    log("\nSentido de cobrança das praças modeladas, por ano")
    for p in config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS:
        b = brutos[brutos.PRACA == p]
        log(f"  {p:20s} " + "; ".join(f"{a}: {'/'.join(sorted(g.SENTIDO.dropna().unique()))}"
                                      for a, g in b.groupby(b.DATA.dt.year)))
    log("\nDias atípicos nas praças modeladas (volume < 40% da mediana do mesmo dia da semana no mês)")
    for p in config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS:
        s = diario[diario.PRACA == p].set_index("DATA").VOL
        s = s[s.index >= INICIO]
        ref = s.groupby([s.index.to_period("M"), s.index.dayofweek]).transform("median")
        baixos = s[s < 0.4 * ref]
        log(f"  {p:20s} {len(baixos)} dia(s)" + (": " + ", ".join(f"{d.date()} ({v:.0f})" for d, v in baixos.items()) if len(baixos) else ""))

    # Figura 1: cobertura e nível mensal das praças físicas candidatas
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cores = plt.cm.tab20(np.linspace(0, 1, len(candidatos)))
    for c, p in zip(cores, candidatos):
        s = diario[diario.PRACA == p].set_index("DATA").VOL.resample("MS").mean()
        s = s[s.index >= "2020-01-01"]
        no = p in config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS
        ax.plot(s.index, s / 1000, color=c, lw=1.8 if no else 1.0, ls="-" if no else "--",
                label=p.title() + (" (no TCC)" if no else ""))
    ax.axvspan(pd.Timestamp("2020-01-01"), INICIO, color="#eeeeee", lw=0)
    ax.axvline(FIM_TREINO, color="#52514e", ls=":", lw=0.8)
    ax.text(FIM_TREINO, ax.get_ylim()[1] * 0.97, " fim do treino", fontsize=7, va="top")
    ax.set_ylabel("Média diária no mês (mil veículos)")
    ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.set_title("Praças físicas da RMS: cobertura e nível (sombreado: 2020, fora da modelagem)", fontsize=9)
    fig.tight_layout()
    fig.savefig(AUD / "figura_cobertura_pracas.png", dpi=200)

    # Figura 2: as duas anomalias tratadas (outubro de 2022 e troca de lote)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for p in ["BOITUVA", "PORTO FELIZ", "BLOQUEIO DE BOITUVA", "ALAMBARI"]:
        s = diario[diario.PRACA == p].set_index("DATA").VOL.asfreq("D")["2022-07-01":"2023-02-28"]
        axes[0].plot(s.index, s / s["2022-07-01":"2022-09-30"].mean(), lw=0.8, label=p.title())
    axes[0].axvspan(pd.Timestamp("2022-10-01"), pd.Timestamp("2022-10-31"), color="#fbeaea")
    axes[0].set_title("Out/2022 no lote L13 (volume ÷ média jul–set)", fontsize=8.5)
    axes[0].legend(fontsize=6.5, frameon=False)
    for p in ["ITU", "SOROCABA"]:
        b = brutos[brutos.PRACA == p]
        troca = b[b.LOTE.str.contains("L33")].DATA.min()
        s = diario[diario.PRACA == p].set_index("DATA").VOL[troca - pd.Timedelta(days=120): troca + pd.Timedelta(days=120)]
        axes[1].plot(s.index, s / 1000, lw=0.8, label=p.title())
        axes[1].axvline(troca, color="#52514e", ls=":", lw=0.8)
    axes[1].set_title("Troca de lote L12 → L33 (linha pontilhada)", fontsize=8.5)
    axes[1].set_ylabel("Mil veículos/dia", fontsize=8)
    axes[1].legend(fontsize=6.5, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(AUD / "figura_anomalias.png", dpi=200)
    (AUD / "eda_elegibilidade.txt").write_text("\n".join(LOG), encoding="utf-8")
