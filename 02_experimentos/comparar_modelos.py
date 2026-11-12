"""
Junta os resultados de todos os modelos já rodados numa tabela comparativa.

Lê todos os arquivos resultados/*_metricas_por_origem.csv (ARIMA, ingênuo e,
no futuro, Prophet e redes neurais -- entram automaticamente, sem alterar este
script) e calcula, para cada praça e horizonte:
  - a média de cada métrica por modelo;
  - em quantas datas de partida (%) cada modelo teve SMAPE menor que o ingênuo (o SMAPE é a métrica
    principal de comparação do trabalho).

O núcleo e o grupo unidirecional (Sorocaba e Itu) aparecem separados: as
praças unidirecionais medem só um sentido e não devem ser comparadas
diretamente com as praças do núcleo.

Como executar (depois de rodar os modelos):
    python comparar_modelos.py
Saída:
    resultados/comparacao_modelos.csv
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
PASTA_RESULTADOS = RAIZ / "resultados"
CHAVES = ["GRUPO", "PRACA", "HORIZONTE", "DIAS"]
METRICAS = ["MAE", "RMSE", "MAPE", "SMAPE", "DTW"]

if __name__ == "__main__":
    arquivos = sorted(PASTA_RESULTADOS.glob("*_metricas_por_origem.csv"))
    if not arquivos:
        raise SystemExit("Nenhum resultado encontrado em resultados/. Rode os modelos antes.")

    dados = pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)
    modelos = sorted(dados["MODELO"].unique())
    print("Modelos encontrados:", ", ".join(modelos))

    # As médias ficam SEM arredondar até a hora de gravar e de imprimir. Arredondar
    # duas vezes (2 casas e depois 1) alterava o último dígito de algumas células.
    medias = dados.groupby(CHAVES + ["MODELO"])[METRICAS].mean().reset_index()

    if "INGENUO" in modelos:
        ingenuo = dados[dados["MODELO"] == "INGENUO"][CHAVES + ["ORIGEM", "SMAPE"]]
        ingenuo = ingenuo.rename(columns={"SMAPE": "SMAPE_INGENUO"})
        pareado = dados.merge(ingenuo, on=CHAVES + ["ORIGEM"], how="left")
        pareado["VENCE_INGENUO"] = pareado["SMAPE"] < pareado["SMAPE_INGENUO"]
        vitorias = (pareado.groupby(CHAVES + ["MODELO"])["VENCE_INGENUO"]
                    .mean().mul(100).round(1).rename("PCT_ORIGENS_VENCE_INGENUO").reset_index())
        medias = medias.merge(vitorias, on=CHAVES + ["MODELO"], how="left")
        medias.loc[medias["MODELO"] == "INGENUO", "PCT_ORIGENS_VENCE_INGENUO"] = None

    ordem_grupo = {"NUCLEO": 0, "UNIDIRECIONAL": 1}
    medias["_ORDEM"] = medias["GRUPO"].map(ordem_grupo)
    medias = medias.sort_values(["_ORDEM", "PRACA", "DIAS", "MODELO"]).drop(columns="_ORDEM")
    medias.assign(**{m: medias[m].round(2) for m in METRICAS}).to_csv(
        PASTA_RESULTADOS / "comparacao_modelos.csv", index=False, encoding="utf-8-sig")

    tabela = medias.pivot_table(index=["GRUPO", "PRACA", "DIAS"], columns="MODELO",
                                values="SMAPE").round(1)
    tabela = tabela.reindex(sorted(tabela.index, key=lambda i: (ordem_grupo.get(i[0], 9), i[1], i[2])))
    print("\n=== SMAPE médio (%), por modelo ===")
    print(tabela.to_string())
    print(f"\nTabela completa (todas as métricas): {PASTA_RESULTADOS / 'comparacao_modelos.csv'}")
