"""
Organiza arquivos brutos da ARTESP a partir de uma estrutura "por ano",
como baixada diretamente do portal:

    <raiz>/volume_trafego_diario_2020/VOLUME_PEDAGIADO-L12-DIARIO-2020.csv
    <raiz>/volume_trafego_diario_2020/VOLUME_PEDAGIADO-L13-DIARIO-2020.csv
    <raiz>/volume_trafego_diario_2021/VOLUME_PEDAGIADO-L13-DIARIO-2021.csv
    ...

E copia cada arquivo relevante (só os lotes usados neste trabalho) para a
estrutura que extracao.py espera:

    dados/brutos/volume_pedagiado/nucleo/{LOTE}/{ano}.csv
    dados/brutos/volume_pedagiado/estudo_caso/{LOTE}/{ano}.csv

Uso (na raiz do repositório TCC):
    python organizar_por_ano.py "C:\\Users\\cyro.oikawa\\OneDrive - Vida Forte Nutrientes Ind E Com De Prod Nat Ltda\\Documents\\TCC\\dpc"

O caminho entre aspas é obrigatório no Windows por causa dos espaços no nome
da pasta do OneDrive.
"""

import re
import shutil
import sys
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parent
DESTINO = RAIZ_REPO / "dados" / "brutos" / "volume_pedagiado"

# Mapeamento lote (como aparece no nome do arquivo) -> (grupo, nome da pasta do lote)
LOTES = {
    "L12": ("estudo_caso", "L12-VIAOESTE"),
    "L13": ("nucleo",      "L13-COLINAS"),
    "L20": ("nucleo",      "L20-SPVIAS"),
    "L33": ("estudo_caso", "L33-ROTA_SOROCABANA"),
    "L34": ("estudo_caso", "L34-RAPOSO_CASTELLO"),
}

PADRAO_NOME = re.compile(r"VOLUME_PEDAGIADO-(L\d+)-DIARIO-(\d{4})\.csv", re.IGNORECASE)


def organizar(raiz_dados: Path) -> None:
    if not raiz_dados.exists():
        print(f"[erro] pasta não encontrada: {raiz_dados}")
        sys.exit(1)

    arquivos_encontrados = list(raiz_dados.rglob("*.csv"))
    print(f"Encontrados {len(arquivos_encontrados)} arquivos .csv em {raiz_dados}\n")

    copiados = 0
    ignorados = []

    for arquivo in arquivos_encontrados:
        m = PADRAO_NOME.match(arquivo.name)
        if not m:
            ignorados.append(arquivo.name)
            continue

        lote, ano = m.group(1).upper(), m.group(2)
        if lote not in LOTES:
            ignorados.append(arquivo.name)
            continue

        grupo, pasta_lote = LOTES[lote]
        pasta_destino = DESTINO / grupo / pasta_lote
        pasta_destino.mkdir(parents=True, exist_ok=True)

        destino_final = pasta_destino / f"{ano}.csv"
        shutil.copy2(arquivo, destino_final)
        print(f"  {arquivo.name}  ->  {destino_final.relative_to(RAIZ_REPO)}")
        copiados += 1

    print(f"\n{copiados} arquivo(s) copiado(s) para dados/brutos/volume_pedagiado/.")
    if ignorados:
        print(f"{len(ignorados)} arquivo(s) ignorado(s) (lote fora do escopo deste "
              f"trabalho, ou nome fora do padrão esperado):")
        for nome in ignorados[:15]:
            print(f"    - {nome}")
        if len(ignorados) > 15:
            print(f"    ... e mais {len(ignorados) - 15}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python organizar_por_ano.py <caminho_da_pasta_com_os_anos>")
        sys.exit(1)

    organizar(Path(sys.argv[1]))
