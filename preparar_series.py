"""
Gera as séries diárias prontas para modelagem, a partir da saída do
tratamento.py.

Entrada:
    dados/processados/nucleo_rms_final.csv
    dados/processados/estudo_caso_rms_final.csv
Saída:
    dados/processados/series_diarias.csv

Uma linha por praça por dia, com calendário diário completo (dias sem registro
aparecem como linha, com volume vazio, em vez de simplesmente sumirem). A
coluna STATUS explica cada dia:
    OK          -> dado presente e válido
    AUSENTE     -> dia sem nenhum registro no arquivo da ARTESP
    INVALIDADO  -> dado existe, mas foi descartado (ver PERIODOS_INVALIDOS
                   em config.py)

Como executar (depois de extracao.py e tratamento.py):
    python preparar_series.py
"""

import pandas as pd
from pathlib import Path

import config

RAIZ = Path(__file__).resolve().parent
PROCESSADOS = RAIZ / "dados" / "processados"

COLUNAS_VOLUME = ["VOLUME_TOTAL", "VOLUME_LEVE", "VOLUME_MOTO", "VOLUME_PESADO"]


def montar_serie(df: pd.DataFrame, praca: str, grupo: str) -> pd.DataFrame:
    sub = df[df["PRACA"] == praca]
    if sub.empty:
        print(f"  [ALERTA] praça {praca} não encontrada nos dados tratados.")
        return pd.DataFrame()

    diario = sub.groupby("DATA")[COLUNAS_VOLUME].sum()

    calendario = pd.date_range(diario.index.min(), diario.index.max(), freq="D")
    diario = diario.reindex(calendario)
    diario.index.name = "DATA"

    diario["STATUS"] = "OK"
    diario.loc[diario["VOLUME_TOTAL"].isna(), "STATUS"] = "AUSENTE"

    for inicio, fim in config.PERIODOS_INVALIDOS.get(praca, []):
        mascara = (diario.index >= inicio) & (diario.index <= fim)
        diario.loc[mascara, COLUNAS_VOLUME] = float("nan")
        diario.loc[mascara, "STATUS"] = "INVALIDADO"

    diario = diario.reset_index()
    diario.insert(1, "PRACA", praca)
    diario.insert(2, "GRUPO", grupo)
    diario.insert(3, "DIRECIONALIDADE", sub["DIRECIONALIDADE"].iloc[0])
    return diario


if __name__ == "__main__":
    nucleo = pd.read_csv(PROCESSADOS / "nucleo_rms_final.csv", low_memory=False)
    estudo = pd.read_csv(PROCESSADOS / "estudo_caso_rms_final.csv", low_memory=False)
    for df in (nucleo, estudo):
        df["DATA"] = pd.to_datetime(df["DATA"])

    partes = []
    for praca in config.PRACAS_NUCLEO:
        partes.append(montar_serie(nucleo, praca, "NUCLEO"))
    for praca in config.PRACAS_CASO_SOROCABA:
        partes.append(montar_serie(estudo, praca, "CASO_SOROCABA"))

    series = pd.concat([p for p in partes if not p.empty], ignore_index=True)

    caminho = PROCESSADOS / "series_diarias.csv"
    series.to_csv(caminho, index=False, encoding="utf-8-sig")

    print("Resumo por praça:")
    resumo = series.groupby(["GRUPO", "PRACA", "DIRECIONALIDADE"]).agg(
        inicio=("DATA", "min"),
        fim=("DATA", "max"),
        dias_ok=("STATUS", lambda s: (s == "OK").sum()),
        dias_ausentes=("STATUS", lambda s: (s == "AUSENTE").sum()),
        dias_invalidados=("STATUS", lambda s: (s == "INVALIDADO").sum()),
        media_diaria=("VOLUME_TOTAL", "mean"),
    )
    resumo["media_diaria"] = resumo["media_diaria"].round(0)
    print(resumo.to_string())
    print(f"\nSalvo em: {caminho}")
