"""
Junta os resultados de todos os modelos já rodados numa tabela comparativa.

Lê todos os arquivos resultados/*_metricas_por_origem.csv (ARIMA, ingênuo e,
no futuro, Prophet e redes neurais -- entram automaticamente, sem alterar este
script) e calcula, para cada praça e horizonte:
  - a média de cada métrica por modelo;
  - em quantas datas de partida (%) cada modelo errou menos que o ingênuo.

O núcleo e o caso Sorocaba aparecem separados: Sorocaba é unidirecional e não
deve ser comparada diretamente com as praças do núcleo.

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

    medias = dados.groupby(CHAVES + ["MODELO"])[METRICAS].mean().round(2).reset_index()

    if "INGENUO" in modelos:
        ingenuo = dados[dados["MODELO"] == "INGENUO"][CHAVES + ["ORIGEM", "MAPE"]]
        ingenuo = ingenuo.rename(columns={"MAPE": "MAPE_INGENUO"})
        pareado = dados.merge(ingenuo, on=CHAVES + ["ORIGEM"], how="left")
        pareado["VENCE_INGENUO"] = pareado["MAPE"] < pareado["MAPE_INGENUO"]
        vitorias = (pareado.groupby(CHAVES + ["MODELO"])["VENCE_INGENUO"]
                    .mean().mul(100).round(1).rename("PCT_ORIGENS_VENCE_INGENUO").reset_index())
        medias = medias.merge(vitorias, on=CHAVES + ["MODELO"], how="left")
        medias.loc[medias["MODELO"] == "INGENUO", "PCT_ORIGENS_VENCE_INGENUO"] = None

    medias = medias.sort_values(["GRUPO", "PRACA", "DIAS", "MODELO"],
                                ascending=[False, True, True, True])
    medias.to_csv(PASTA_RESULTADOS / "comparacao_modelos.csv", index=False, encoding="utf-8-sig")

    tabela = medias.pivot_table(index=["GRUPO", "PRACA", "DIAS"], columns="MODELO",
                                values="MAPE").round(1)
    tabela = tabela.sort_index(level=["GRUPO", "PRACA", "DIAS"], ascending=[False, True, True])
    print("\n=== MAPE médio (%), por modelo ===")
    print(tabela.to_string())
    print(f"\nTabela completa (todas as métricas): {PASTA_RESULTADOS / 'comparacao_modelos.csv'}")
