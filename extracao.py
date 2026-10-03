"""
Extração e unificação dos arquivos brutos da ARTESP.

Estrutura de pastas esperada:
    dados/brutos/volume_pedagiado/nucleo/{LOTE}/*.csv
    dados/brutos/volume_pedagiado/estudo_caso/{LOTE}/*.csv

Núcleo: praças sem quebra estrutural (L13-Colinas, L20-SPVias) — usadas na
comparação principal entre modelos estatísticos e arquiteturas de aprendizado
profundo.

Estudo de caso: praças atravessadas por mudança de instrumento de medição
(L12-ViaOeste, L33-Rota Sorocabana, L34-Raposo Castello) — usadas só na
discussão sobre robustez frente a descontinuidade estrutural.
"""

import pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
BRUTOS = RAIZ / "dados" / "brutos" / "volume_pedagiado"
SAIDA = RAIZ / "dados" / "processados"
SAIDA.mkdir(parents=True, exist_ok=True)


def carregar_lote(pasta_lote: Path) -> pd.DataFrame:
    """Lê todos os CSVs de um lote, concatena, marca origem.

    Cada arquivo é nomeado com um ano (ex.: '2023.csv'), mas o portal da
    ARTESP não garante que o conteúdo do arquivo esteja contido nesse ano —
    já foi encontrado arquivo de 2023 que estende até setembro de 2024. Sem
    filtrar isso, a concatenação com o arquivo de 2024 duplica esses meses.
    Por isso, linhas cuja DATA não pertence ao ano indicado pelo nome do
    arquivo são descartadas aqui, e reportadas.
    """
    arquivos = sorted(pasta_lote.glob("*.csv"))
    if not arquivos:
        print(f"  [aviso] nenhum CSV encontrado em {pasta_lote}")
        return pd.DataFrame()

    partes = []
    colunas_referencia = None
    for arquivo in arquivos:
        df = pd.read_csv(arquivo, dtype=str, low_memory=False)

        if colunas_referencia is None:
            colunas_referencia = tuple(df.columns)
        elif tuple(df.columns) != colunas_referencia:
            print(f"  [ALERTA] {arquivo.name} tem colunas diferentes dos demais "
                  f"arquivos deste lote — revisar antes de prosseguir.")

        ano_esperado = arquivo.stem
        if ano_esperado.isdigit() and "DATA" in df.columns:
            datas = pd.to_datetime(df["DATA"], errors="coerce")
            fora_do_ano = datas.dt.year.astype("Int64") != int(ano_esperado)
            n_fora = int(fora_do_ano.sum())
            if n_fora > 0:
                anos_encontrados = sorted(datas[fora_do_ano].dt.year.dropna().unique().tolist())
                print(f"  [ALERTA] {arquivo.name}: {n_fora} linha(s) com DATA fora "
                      f"de {ano_esperado} (anos encontrados: {anos_encontrados}). "
                      f"Descartadas para evitar duplicação com o arquivo do ano correto.")
                df = df[~fora_do_ano]

        df["ARQUIVO_ORIGEM"] = arquivo.name
        df["LOTE_PASTA"] = pasta_lote.name
        partes.append(df)

    return pd.concat(partes, ignore_index=True)


def carregar_grupo(pasta_grupo: Path, nome_grupo: str) -> pd.DataFrame:
    if not pasta_grupo.exists():
        print(f"[aviso] pasta do grupo '{nome_grupo}' não existe: {pasta_grupo}")
        return pd.DataFrame()

    partes = []
    for pasta_lote in sorted(p for p in pasta_grupo.iterdir() if p.is_dir()):
        print(f"Carregando lote: {pasta_lote.name}")
        df_lote = carregar_lote(pasta_lote)
        if not df_lote.empty:
            partes.append(df_lote)

    if not partes:
        return pd.DataFrame()

    df_grupo = pd.concat(partes, ignore_index=True)
    print(f"Grupo '{nome_grupo}': {len(df_grupo)} linhas, "
          f"{df_grupo['LOTE_PASTA'].nunique()} lote(s).")
    return df_grupo


if __name__ == "__main__":
    print("=== NÚCLEO (comparação principal) ===")
    df_nucleo = carregar_grupo(BRUTOS / "nucleo", "núcleo")
    if not df_nucleo.empty:
        caminho = SAIDA / "nucleo_bruto_unificado.csv"
        df_nucleo.to_csv(caminho, index=False, encoding="utf-8-sig")
        print(f"Salvo em: {caminho}\n")

    print("=== ESTUDO DE CASO (descontinuidade estrutural) ===")
    df_estudo = carregar_grupo(BRUTOS / "estudo_caso", "estudo_caso")
    if not df_estudo.empty:
        caminho = SAIDA / "estudo_caso_bruto_unificado.csv"
        df_estudo.to_csv(caminho, index=False, encoding="utf-8-sig")
        print(f"Salvo em: {caminho}")
