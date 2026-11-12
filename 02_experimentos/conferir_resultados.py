"""
Confere os resultados gerados no seu computador com os valores esperados.

Lê resultados/*_metricas_por_origem.csv (gerados por modelo_ingenuo.py, arima_baseline.py,
prophet_modelo.py e redes_neurais.py) e compara o SMAPE médio de cada modelo e horizonte
com o arquivo de valores esperados da configuração em uso (config.CONFIGURACAO):

    esperado_configuracao_ajustada.json   (padrão; a configuração do capítulo 4)
    esperado_configuracao_original.json   (set TCC_CONFIGURACAO=original)

Resultado de cada linha:
    OK         diferença dentro da tolerância
    ATENÇÃO    diferença até o dobro da tolerância (em geral, versão de biblioteca)
    DIFERENTE  diferença maior: revisar antes de usar os números
    FALTA      o modelo ainda não foi rodado nesta pasta

Uso:  python conferir_resultados.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

import config

RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "resultados"
NOMES = {"INGENUO": "Ingênuo sazonal", "ARIMA": "SARIMA", "PROPHET": "Prophet", "LSTM": "LSTM", "GRU": "GRU",
         "BILSTM": "Bi-LSTM", "BIGRU": "Bi-GRU", "NHITS": "N-HiTS"}
# Tolerância em pontos percentuais de SMAPE (média das seis praças | pior praça).
# Ingênuo e Prophet são praticamente determinísticos. O SARIMA, com raiz sazonal quase
# unitária, para em pontos um pouco diferentes do otimizador conforme o computador.
# As redes variam com a biblioteca de álgebra linear e a ordem das operações.
TOLERANCIA = {"INGENUO": (0.01, 0.01), "ARIMA": (0.10, 0.20), "PROPHET": (0.05, 0.10)}
TOLERANCIA_REDES = (0.30, 0.60)


def situacao(dif, tol):
    if dif <= tol:
        return "OK"
    return "ATENÇÃO" if dif <= 2 * tol else "DIFERENTE"


def main():
    arq = RAIZ / f"esperado_configuracao_{config.CONFIGURACAO}.json"
    esperado = json.loads(arq.read_text(encoding="utf-8"))
    print(f"Configuração em uso: {config.CONFIGURACAO.upper()}  (valores esperados: {arq.name})")
    print(f"Origem dos valores esperados: {esperado['origem_dos_valores']}\n")

    arquivos = sorted(RESULTADOS.glob("*_metricas_por_origem.csv"))
    if not arquivos and not list((RESULTADOS / "redes").glob("*_por_origem.csv")):
        sys.exit("Nenhum resultado encontrado em resultados/. Rode antes os scripts dos modelos.")
    arquivos = arquivos or []
    d = (pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True) if arquivos
         else pd.DataFrame(columns=["MODELO", "DIAS", "PRACA", "SMAPE"]))
    # Redes rodadas uma combinação por vez (python redes_neurais.py LSTM 180) ficam em
    # resultados/redes/<MODELO>_<h>_por_origem.csv até que os três horizontes existam.
    for a in sorted((RESULTADOS / "redes").glob("*_por_origem.csv")):
        m, h = a.stem.split("_")[:2]
        if m in NOMES and h.isdigit() and not ((d.MODELO == m) & (d.DIAS == int(h))).any():
            d = pd.concat([d, pd.read_csv(a)], ignore_index=True)
    obtido_praca = d.groupby(["MODELO", "DIAS", "PRACA"]).SMAPE.mean()
    esp_praca = pd.DataFrame(esperado["por_praca"]).set_index(["modelo", "dias", "praca"]).smape

    contagem = {"OK": 0, "ATENÇÃO": 0, "DIFERENTE": 0, "FALTA": 0}
    print(f"{'Modelo':16s}{'Dias':>5s}{'Obtido':>9s}{'Esperado':>10s}{'Dif.':>7s}{'Pior praça (dif.)':>30s}   Resultado")
    print("-" * 92)
    for m in NOMES:
        for h in (7, 90, 180):
            esp = esp_praca.xs((m, h), level=["modelo", "dias"])
            if (m, h) not in obtido_praca.index.droplevel(2):
                print(f"{NOMES[m]:16s}{h:5d}{'—':>9s}{esp.mean():10.2f}{'':>7s}{'':>30s}   FALTA")
                contagem["FALTA"] += 1
                continue
            obt = obtido_praca.loc[(m, h)]
            difs = (obt - esp).abs().dropna()
            dif_media = abs(obt.mean() - esp.mean())
            tol_m, tol_p = TOLERANCIA.get(m, TOLERANCIA_REDES)
            res = max([situacao(dif_media, tol_m), situacao(difs.max(), tol_p)],
                      key=["OK", "ATENÇÃO", "DIFERENTE"].index)
            pior = f"{difs.idxmax().title()} ({difs.max():.2f})"
            print(f"{NOMES[m]:16s}{h:5d}{obt.mean():9.2f}{esp.mean():10.2f}{dif_media:7.2f}{pior:>30s}   {res}")
            contagem[res] += 1

    medias = obtido_praca.groupby(["MODELO", "DIAS"]).mean()
    print()
    if all((m, h) in medias.index for m in ("PROPHET", "ARIMA") for h in (7, 90, 180)):
        prophet_1o = all(medias.xs(h, level="DIAS").idxmin() == "PROPHET" for h in (7, 90, 180))
        print(f"Prophet com o menor erro nos três horizontes: {'SIM' if prophet_1o else 'NÃO'}")
        if config.CONFIGURACAO == "ajustada" and len(medias.index.get_level_values(0).unique()) == 8:
            segundo = all(medias.xs(h, level="DIAS").sort_values().index[1] == "ARIMA" for h in (7, 90, 180))
            print(f"SARIMA em segundo lugar nos três horizontes:   {'SIM' if segundo else 'NÃO'}")
    print(f"\nResumo: {contagem['OK']} OK, {contagem['ATENÇÃO']} ATENÇÃO, {contagem['DIFERENTE']} DIFERENTE, "
          f"{contagem['FALTA']} FALTA (de 24 combinações de modelo e horizonte)")
    if contagem["DIFERENTE"]:
        print("Há diferenças acima do esperado. Confira as versões das bibliotecas (versoes_bibliotecas.txt) "
              "e se a configuração em uso é a certa.")
    elif contagem["ATENÇÃO"]:
        print("Diferenças pequenas, típicas de outra versão de biblioteca ou de outro processador. "
              "As conclusões não mudam, mas use nos textos os números da sua execução.")
    elif not contagem["FALTA"]:
        print("Tudo confere com os resultados do capítulo 4.")


if __name__ == "__main__":
    main()
