"""
Mostra na tela (e grava em CSV para o Excel) as métricas de cada modelo.

Lê os arquivos que os modelos já geraram em resultados/:
    ingenuo_metricas_por_origem.csv
    arima_metricas_por_origem.csv
    prophet_metricas_por_origem.csv

Cada linha desses arquivos é UMA previsão: uma praça, um horizonte, uma data de
partida (origem) e as cinco métricas daquela previsão. Este script calcula a
média das origens, que é o valor das tabelas da monografia.

Como executar (a partir da pasta do projeto):

    python ver_metricas.py                       SMAPE de todas as praças
    python ver_metricas.py DTW                   outra métrica (MAE, RMSE, MAPE, SMAPE ou DTW)
    python ver_metricas.py SMAPE "PORTO FELIZ"   só uma praça
    python ver_metricas.py SMAPE "PORTO FELIZ" --origens 7
                                                 as previsões de 7 dias, uma por data de partida

O asterisco (*) marca o modelo de menor valor em cada linha.
A tabela também é gravada em resultados/tabela_<METRICA>.csv, no formato que o
Excel em português abre direto (separador ponto e vírgula, vírgula decimal).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parent / "resultados"
ARQUIVOS = {"INGENUO": "ingenuo", "ARIMA": "arima", "PROPHET": "prophet"}
ROTULOS = {"INGENUO": "Ingenuo", "ARIMA": "SARIMA", "PROPHET": "Prophet"}
METRICAS = ["MAE", "RMSE", "MAPE", "SMAPE", "DTW"]
ORDEM_PRACAS = ["ALAMBARI", "BLOQUEIO DE BOITUVA", "BOITUVA", "PORTO FELIZ", "ITU", "SOROCABA"]


def carregar() -> pd.DataFrame:
    partes = []
    for modelo, nome in ARQUIVOS.items():
        caminho = PASTA / f"{nome}_metricas_por_origem.csv"
        if caminho.exists():
            partes.append(pd.read_csv(caminho))
        else:
            print(f"[aviso] {caminho.name} não encontrado: rode o script do modelo {ROTULOS[modelo]} antes.")
    if not partes:
        sys.exit("Nenhum resultado encontrado em resultados/.")
    return pd.concat(partes, ignore_index=True)


def tabela_media(dados, metrica, praca):
    if praca:
        dados = dados[dados["PRACA"] == praca.upper()]
        if dados.empty:
            sys.exit(f"Praça '{praca}' não encontrada. Opções: {ORDEM_PRACAS}")
    medias = dados.groupby(["PRACA", "DIAS", "MODELO"])[metrica].mean().unstack("MODELO")
    medias = medias[[m for m in ARQUIVOS if m in medias.columns]].reset_index()
    ordem = {p: i for i, p in enumerate(ORDEM_PRACAS)}
    medias["_o"] = medias["PRACA"].map(ordem).fillna(99)
    return medias.sort_values(["_o", "DIAS"]).drop(columns="_o").set_index(["PRACA", "DIAS"])


def mostrar(medias, metrica):
    unidade = "%" if metrica in ("MAPE", "SMAPE") else "veículos/dia"
    print(f"\n{metrica} médio por praça, horizonte (dias) e modelo  [{unidade}]\n")
    cab = f"{'praça':22s} {'dias':>5s} " + " ".join(f"{ROTULOS[m]:>10s}" for m in medias.columns)
    print(cab); print("-" * len(cab))
    ultima = None
    for (praca, dias), linha in medias.iterrows():
        menor = linha.idxmin()
        valores = " ".join(f"{linha[m]:>9.2f}{'*' if m == menor else ' '}" for m in medias.columns)
        print(f"{(praca if praca != ultima else ''):22s} {dias:>5d} {valores}")
        ultima = praca
    print("\n* = menor valor da linha (melhor modelo)")


def mostrar_origens(dados, metrica, praca, horizonte):
    x = dados[(dados["PRACA"] == praca.upper()) & (dados["DIAS"] == horizonte)]
    if x.empty:
        sys.exit("Nenhuma previsão com essa praça e esse horizonte.")
    t = x.pivot_table(index="ORIGEM", columns="MODELO", values=metrica)
    t = t[[m for m in ARQUIVOS if m in t.columns]]
    print(f"\n{metrica} de cada previsão de {horizonte} dias em {praca.upper()}, por data de partida (origem)\n")
    print(f"{'origem':12s} " + " ".join(f"{ROTULOS[m]:>10s}" for m in t.columns)); print("-" * (13 + 11 * len(t.columns)))
    for origem, linha in t.iterrows():
        print(f"{origem:12s} " + " ".join(f"{linha[m]:>10.2f}" for m in t.columns))
    print(f"\nMédia das {len(t)} origens: " + ", ".join(f"{ROTULOS[m]} {t[m].mean():.2f}" for m in t.columns))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Consulta as métricas dos modelos.")
    ap.add_argument("metrica", nargs="?", default="SMAPE", choices=METRICAS)
    ap.add_argument("praca", nargs="?", default=None)
    ap.add_argument("--origens", type=int, metavar="HORIZONTE", help="mostra cada previsão (7, 90 ou 180 dias)")
    a = ap.parse_args()
    dados = carregar()
    if a.origens:
        if not a.praca:
            sys.exit('Para --origens, informe a praça. Exemplo: python ver_metricas.py SMAPE "PORTO FELIZ" --origens 7')
        mostrar_origens(dados, a.metrica, a.praca, a.origens)
    else:
        medias = tabela_media(dados, a.metrica, a.praca)
        mostrar(medias, a.metrica)
        saida = PASTA / f"tabela_{a.metrica}.csv"
        medias.round(2).rename(columns=ROTULOS).to_csv(saida, sep=";", decimal=",", encoding="utf-8-sig")
        print(f"Tabela gravada em: {saida.name}")
