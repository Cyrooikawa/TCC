"""
Auditoria independente dos dados brutos da ARTESP e da série diária do TCC.

Não usa extracao.py nem preparar_series.py: relê TODOS os arquivos brutos, de
TODOS os lotes, com código próprio, e confere:

  1. inventário de todas as praças da RMS em qualquer lote (não só nos 5 lotes
     usados pelo pipeline), com período, cobertura, volume e sentidos;
  2. linhas com DATA fora do ano do arquivo;
  3. a mesma praça e data em mais de um arquivo ou lote (dupla contagem);
  4. volume nas colunas de mais de 10 eixos (que o pipeline descarta);
  5. a série de cada praça modelada, recalculada do zero, contra series_diarias.csv.

Uso:
    python auditar_dados_brutos.py <pasta_com_os_arquivos_brutos>
Saídas (pasta auditoria/):
    inventario_pracas_rms.csv, diario_pracas_rms.csv (volume diário por praça),
    auditoria_resumo.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402
from tratamento import MUNICIPIOS_RMS_NORM, extrair_municipio, normalizar  # noqa: E402

SAIDA = RAIZ / "auditoria"
SAIDA.mkdir(exist_ok=True)
LOG = []


def log(*a):
    t = " ".join(str(x) for x in a)
    print(t)
    LOG.append(t)


def municipio(praca):
    praca = normalizar(praca)
    partes = praca.split(" - ", 1)
    return extrair_municipio(partes[1] if len(partes) > 1 else praca, partes[0])


def ler(arquivo):
    df = pd.read_csv(arquivo, dtype=str, low_memory=False)
    df["PRACA"] = df["PRACA"].map(normalizar)
    eixos = [c for c in df.columns if "EIXOS" in c.upper()]
    for c in eixos:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    ate10 = [c for c in eixos if int(c.split(" ")[-1]) <= 10]
    acima = [c for c in eixos if int(c.split(" ")[-1]) > 10]
    df["VOL"] = df[ate10].sum(axis=1)
    df["VOL_ACIMA_10"] = df[acima].sum(axis=1) if acima else 0
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
    ano = int(arquivo.stem.split("-")[-1])
    df["FORA_DO_ANO"] = df["DATA"].dt.year != ano
    df["ARQUIVO"] = arquivo.name
    return df[["DATA", "LOTE", "PRACA", "SENTIDO", "VOL", "VOL_ACIMA_10", "FORA_DO_ANO", "ARQUIVO"]]


if __name__ == "__main__":
    pasta = Path(sys.argv[1])
    arquivos = sorted(pasta.rglob("VOLUME_PEDAGIADO-*-DIARIO-*.csv"))
    log(f"Arquivos brutos: {len(arquivos)}")

    # nomes de praça de todos os arquivos, para achar as da RMS (leitura guardada em cache)
    cache = SAIDA / "cache_linhas_rms.pkl"
    if cache.exists() and "--refazer" not in sys.argv:
        d = pd.read_pickle(cache)
    else:
        partes = []
        for a in arquivos:
            df = ler(a)
            df["MUNICIPIO"] = df["PRACA"].map({p: municipio(p) for p in df["PRACA"].unique()})
            rms = df[df["MUNICIPIO"].isin(MUNICIPIOS_RMS_NORM)]
            if len(rms):
                partes.append(rms)
        d = pd.concat(partes, ignore_index=True)
        d.to_pickle(cache)
    log(f"Linhas de praças da RMS: {len(d)}; lotes com praça da RMS: {sorted(d.LOTE.unique())}")

    # 2. linhas fora do ano do arquivo
    fora = d[d.FORA_DO_ANO]
    log(f"\n[2] Linhas com DATA fora do ano do arquivo: {len(fora)}")
    if len(fora):
        log(fora.groupby(["ARQUIVO"]).DATA.agg(["count", "min", "max"]).to_string())
    d = d[~d.FORA_DO_ANO]

    # 3. dupla contagem: mesma praça e data vinda de mais de um arquivo
    por_arq = d.groupby(["PRACA", "DATA"]).ARQUIVO.nunique()
    dup = por_arq[por_arq > 1]
    log(f"\n[3] Praça e data presentes em mais de um arquivo: {len(dup)}")
    if len(dup):
        log(dup.reset_index().groupby("PRACA").DATA.agg(["count", "min", "max"]).to_string())

    # 4. volume acima de 10 eixos
    a10 = d.groupby("PRACA")[["VOL", "VOL_ACIMA_10"]].sum()
    a10["PCT_ACIMA_10"] = a10.VOL_ACIMA_10 / (a10.VOL + a10.VOL_ACIMA_10) * 100
    log(f"\n[4] Volume em colunas de mais de 10 eixos (descartado): no máximo "
        f"{a10.PCT_ACIMA_10.max():.4f}% do total de uma praça")

    # 1. inventário
    diario = d.groupby(["PRACA", "DATA"]).VOL.sum().reset_index()
    diario.to_csv(SAIDA / "diario_pracas_rms.csv", index=False, encoding="utf-8-sig")
    inv = []
    for p, g in d.groupby("PRACA"):
        dd = diario[diario.PRACA == p].set_index("DATA").VOL
        cal = pd.date_range(dd.index.min(), dd.index.max())
        inv.append({
            "PRACA": p, "MUNICIPIO": g.MUNICIPIO.iloc[0], "LOTES": "/".join(sorted(g.LOTE.unique())),
            "REGIME": "PORTICO" if p.startswith("PORTICO") else "PRACA FISICA",
            "SENTIDOS": "/".join(sorted(g.SENTIDO.dropna().unique())), "N_SENTIDOS": g.SENTIDO.nunique(),
            "INICIO": dd.index.min().date(), "FIM": dd.index.max().date(), "DIAS_COM_DADO": len(dd),
            "DIAS_AUSENTES": len(cal) - len(dd), "MEDIA_DIARIA_2024": dd[dd.index.year == 2024].mean(),
            "MEDIA_DIARIA_ULTIMOS_90": dd.iloc[-90:].mean(),
        })
    inv = pd.DataFrame(inv).sort_values(["MUNICIPIO", "PRACA"])
    inv.to_csv(SAIDA / "inventario_pracas_rms.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    log("\n[1] Inventário das praças da RMS em todos os lotes")
    log(inv.round(0).to_string(index=False))

    # 5. série recalculada x series_diarias.csv
    of = pd.read_csv(RAIZ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
    fim_brutos = diario.DATA.max()
    log(f"\n[5] Série recalculada do zero x series_diarias.csv (até {fim_brutos.date()}, último dia dos brutos)")
    for p in config.PRACAS_NUCLEO + config.PRACAS_UNIDIRECIONAIS:
        a = diario[diario.PRACA == p].set_index("DATA").VOL
        b = of[(of.PRACA == p) & (of.STATUS == "OK") & (of.DATA <= fim_brutos)].set_index("DATA").VOLUME_TOTAL
        inv_ = [(i, f) for i, f in config.PERIODOS_INVALIDOS.get(p, [])]
        for i, f in inv_:
            a = a[(a.index < i) | (a.index > f)]
        comum = a.index.intersection(b.index)
        dif = (a.loc[comum] - b.loc[comum]).abs()
        ate_jan = dif[dif.index < "2026-02-01"]
        log(f"  {p:20s} dias em comum {len(comum)}; só nos brutos {len(a.index.difference(b.index))}; "
            f"só na série {len(b.index.difference(a.index))}; diferentes até jan/2026: {int((ate_jan > 0).sum())}; "
            f"de fev a mai/2026: {int((dif[dif.index >= '2026-02-01'] > 0).sum())}")
    (SAIDA / "auditoria_resumo.txt").write_text("\n".join(LOG), encoding="utf-8")
