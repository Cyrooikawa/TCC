"""
Padronização de praça/município da base unificada.

Lista de municípios da RMS corrigida contra fonte oficial (27 municípios em
3 sub-regiões), fornecida pelo autor em 12/09. Duas correções em relação à
lista anterior, montada apenas a partir do que aparecia nos dados:
  - REMOVIDOS: QUADRA, VARGEM GRANDE PAULISTA (não constam na lista oficial)
  - ADICIONADOS: IPERÓ, SÃO MIGUEL ARCANJO (constavam nos dados sob o nome
    'S M ARCANJO' e eram descartados por engano como não identificados)

ATENÇÃO: QUADRA era uma das cinco praças do núcleo de modelagem usado desde
o início deste trabalho. Removê-la da lista de municípios reconhecidos
significa que ela passará a cair em 'fora_da_rms' na saída do pipeline, a
menos que o autor decida mantê-la no núcleo por outro critério (ver
observação no README). Esta decisão NÃO foi tomada automaticamente aqui.
"""

import re
import unicodedata
import pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PROCESSADOS = RAIZ / "dados" / "processados"

# Lista oficial dos 27 municípios da Região Metropolitana de Sorocaba,
# organizados em 3 sub-regiões conforme fornecido pelo autor.
SUBREGIOES_RMS = {
    1: ["ALAMBARI", "BOITUVA", "CAPELA DO ALTO", "CERQUILHO", "CESARIO LANGE",
        "ITAPETININGA", "JUMIRIM", "SARAPUI", "TATUI", "TIETE"],
    2: ["ALUMINIO", "ARACARIGUAMA", "IBIUNA", "ITU", "MAIRINQUE",
        "PORTO FELIZ", "SALTO", "SAO ROQUE"],
    3: ["ARACOIABA DA SERRA", "IPERO", "PIEDADE", "PILAR DO SUL",
        "SALTO DE PIRAPORA", "SAO MIGUEL ARCANJO", "SOROCABA", "TAPIRAI",
        "VOTORANTIM"],
}

MUNICIPIO_PARA_SUBREGIAO = {
    m: sub for sub, lista in SUBREGIOES_RMS.items() for m in lista
}

MUNICIPIOS_RMS = [m for lista in SUBREGIOES_RMS.values() for m in lista]
assert len(MUNICIPIOS_RMS) == 27, f"esperado 27 municípios, encontrado {len(MUNICIPIOS_RMS)}"


def normalizar(texto: str) -> str:
    """Maiúsculas, sem acento, sem espaço nas pontas."""
    if texto is None:
        return ""
    texto = str(texto).strip().upper()
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ASCII", "ignore").decode("ASCII")
    texto = re.sub(r"\s+", " ", texto)
    return texto


MUNICIPIOS_RMS_NORM = {normalizar(m) for m in MUNICIPIOS_RMS}


def classificar_regime(nome_praca: str) -> str:
    """Distingue praça física de pórtico free-flow, sem misturar no nome."""
    nome = normalizar(nome_praca)
    return "PORTICO_FREE_FLOW" if nome.startswith("PORTICO") else "PRACA_FISICA"


def extrair_municipio(nome_praca: str, ponto_id: str) -> str:
    """
    Compara por PALAVRA INTEIRA (token) contra a lista fechada de municípios,
    nunca por substring livre — é a substring livre que causava BOITUVA virar
    ITU (porque 'ITU' é substring de 'BOITUVA', mas não é um token de
    'BOITUVA': os tokens de 'BOITUVA' são só ['BOITUVA']).

    Isso também resolve nomes compostos legítimos, como 'BLOQUEIO DE BOITUVA'
    (token 'BOITUVA' presente) e 'ARAÇOIABA LESTE' (token 'ARAÇOIABA'
    presente), sem reabrir a possibilidade de falso positivo por substring.
    """
    candidatos = [normalizar(nome_praca), normalizar(ponto_id)]

    for candidato in candidatos:
        if candidato in MUNICIPIOS_RMS_NORM:
            return candidato

    municipios_primeiro_token = {m.split(" ")[0]: m for m in MUNICIPIOS_RMS_NORM}
    # Municípios compostos (3+ palavras) também são buscados pela ÚLTIMA
    # palavra, para cobrir abreviações por iniciais (ex.: 'S M ARCANJO' para
    # 'SAO MIGUEL ARCANJO'). Só aplicado a nomes com 3+ palavras, para não
    # arriscar colisão com municípios de duas palavras (ex.: 'SAO ROQUE').
    municipios_ultimo_token = {
        m.split(" ")[-1]: m for m in MUNICIPIOS_RMS_NORM if len(m.split(" ")) >= 3
    }
    for candidato in candidatos:
        for token in candidato.split(" "):
            if token in MUNICIPIOS_RMS_NORM:
                return token
            if token in municipios_primeiro_token:
                return municipios_primeiro_token[token]
            if token in municipios_ultimo_token:
                return municipios_ultimo_token[token]

    if " - " in str(nome_praca):
        _, resto = str(nome_praca).split(" - ", 1)
        resto_norm = normalizar(resto)
        for token in resto_norm.split(" "):
            if token in MUNICIPIOS_RMS_NORM:
                return token

    return "NAO_IDENTIFICADO"


def tratar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PRACA"] = df["PRACA"].apply(normalizar)

    df["REGIME_COBRANCA"] = df["PRACA"].apply(classificar_regime)

    df_separado = df["PRACA"].str.split(" - ", n=1, expand=True)
    df["PONTO_ID"] = df_separado[0].str.strip()
    df["CIDADE_NOME"] = df_separado.get(1, pd.Series(dtype=str))
    df["CIDADE_NOME"] = df["CIDADE_NOME"].fillna(df["PRACA"])

    df["MUNICIPIO"] = df.apply(
        lambda row: extrair_municipio(row["CIDADE_NOME"], row["PONTO_ID"]),
        axis=1,
    )
    df["SUBREGIAO_RMS"] = df["MUNICIPIO"].map(MUNICIPIO_PARA_SUBREGIAO)

    nao_identificados = df[df["MUNICIPIO"] == "NAO_IDENTIFICADO"]["PRACA"].unique()
    if len(nao_identificados) > 0:
        print(f"[ALERTA] {len(nao_identificados)} praça(s) não identificada(s) "
              f"contra a lista de municípios da RMS:")
        for p in nao_identificados:
            print(f"    - {p}")
        print("Revisar a lista MUNICIPIOS_RMS ou o dado de origem antes de prosseguir.\n")

    return df


if __name__ == "__main__":
    for nome_arquivo in ["nucleo_bruto_unificado.csv", "estudo_caso_bruto_unificado.csv"]:
        caminho_entrada = PROCESSADOS / nome_arquivo
        if not caminho_entrada.exists():
            print(f"[aviso] {caminho_entrada} não encontrado, pulando.")
            continue

        print(f"=== Tratando {nome_arquivo} ===")
        df = pd.read_csv(caminho_entrada, dtype=str, low_memory=False)
        df_tratado = tratar(df)

        print(df_tratado[["PRACA", "PONTO_ID", "MUNICIPIO", "SUBREGIAO_RMS", "REGIME_COBRANCA"]]
              .drop_duplicates()
              .sort_values(["MUNICIPIO", "PRACA"])
              .to_string(index=False))

        caminho_saida = PROCESSADOS / nome_arquivo.replace("_bruto_unificado", "_tratado")
        df_tratado.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
        print(f"\nSalvo em: {caminho_saida}")

        prefixo = nome_arquivo.replace("_bruto_unificado.csv", "")
        df_rms = df_tratado[df_tratado["MUNICIPIO"].isin(MUNICIPIOS_RMS_NORM)]
        df_fora = df_tratado[~df_tratado["MUNICIPIO"].isin(MUNICIPIOS_RMS_NORM)]

        caminho_rms = PROCESSADOS / f"{prefixo}_rms_final.csv"
        caminho_fora = PROCESSADOS / f"{prefixo}_fora_da_rms.csv"
        df_rms.to_csv(caminho_rms, index=False, encoding="utf-8-sig")
        df_fora.to_csv(caminho_fora, index=False, encoding="utf-8-sig")

        print(f"\n{len(df_rms)} linhas dentro da RMS -> {caminho_rms}")
        print(f"{len(df_fora)} linhas fora da RMS -> {caminho_fora}\n")
