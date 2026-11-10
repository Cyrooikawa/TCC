"""
Funções comuns aos experimentos de refinamento (seção 6 do Guia dos modelos).

Regra de ouro do guia: nada aqui grava em resultados/ (os resultados oficiais).
Tudo o que os experimentos produzem vai para experimentos/resultados/.

Este arquivo não é executado sozinho.
"""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
PROJ = AQUI.parent
sys.path.insert(0, str(PROJ))
import config  # noqa: E402
import metricas  # noqa: E402

OFICIAIS = PROJ / "resultados"
SAIDA = AQUI / "resultados"
SAIDA.mkdir(exist_ok=True)

ALVOS = [(p, "NUCLEO") for p in config.PRACAS_NUCLEO] + \
        [(p, "UNIDIRECIONAL") for p in config.PRACAS_UNIDIRECIONAIS]
NOME_HORIZONTE = {v: k for k, v in config.HORIZONTES.items()}
H_MAX = max(config.HORIZONTES.values())
H_MIN = min(config.HORIZONTES.values())
FRACAO_VALIDACAO = 0.20   # mesma das redes: 20% finais do treino


def carregar_series() -> dict:
    df = pd.read_csv(PROJ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
    series = {}
    for praca, _ in ALVOS:
        sub = df[(df["PRACA"] == praca) & (df["DATA"] >= config.DATA_INICIO_MODELAGEM)]
        series[praca] = sub.set_index("DATA")["VOLUME_TOTAL"].sort_index().asfreq("D")
    return series


def divisao(n: int):
    """n_treino e início da validação (mesma conta de redes_neurais.py)."""
    n_treino = int(n * config.PROPORCAO_TREINO)
    inicio_val = n_treino - int(round(FRACAO_VALIDACAO * n_treino))
    return n_treino, inicio_val


def origens_teste(n: int) -> list:
    return metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS, H_MIN)


def origens_validacao(n: int) -> list:
    """Origens semanais DENTRO do treino (últimos 20%), para escolher sem olhar o teste.
    Mesma fase semanal das origens do teste (terça-feira)."""
    n_treino, inicio_val = divisao(n)
    todas = list(range(n_treino - 1, inicio_val - 2, -config.PASSO_ORIGENS_DIAS))[::-1]
    return [o for o in todas if o + H_MIN <= n_treino - 1]


def avaliar(y: pd.Series, praca: str, grupo: str, modelo: str, previsoes: dict, limite: int = None) -> list:
    """previsoes: {índice da origem: vetor com a previsão dos dias origem+1, origem+2, ...}.
    limite: último índice que pode ser usado como real (n-1 no teste; n_treino-1 na validação)."""
    n = len(y)
    limite = n - 1 if limite is None else limite
    linhas = []
    for origem, prev in previsoes.items():
        prev = np.asarray(prev, dtype=float)
        for nome_h, h in config.HORIZONTES.items():
            if origem + h > limite or len(prev) < h:
                continue
            real = y.iloc[origem + 1: origem + 1 + h].values
            r = metricas.todas(real, prev[:h], config.JANELA_DTW)
            r.update({"MODELO": modelo, "GRUPO": grupo, "PRACA": praca, "HORIZONTE": nome_h,
                      "DIAS": h, "ORIGEM": y.index[origem].date()})
            linhas.append(r)
    return linhas


def salvar(linhas: list, variante: str, pasta: Path = SAIDA) -> pd.DataFrame:
    d = pd.DataFrame(linhas)
    d.to_csv(pasta / f"{variante}.csv", index=False, encoding="utf-8-sig")
    return d


# ---------------------------------------------------------------------------
# Calendário (o mesmo do Prophet oficial) e tipo de semana (Tabela 9)
# ---------------------------------------------------------------------------

def datas_feriado(ano_inicio=2020, ano_fim=2027) -> set:
    import holidays
    from dateutil.easter import easter
    anos = range(ano_inicio, ano_fim + 1)
    datas = set(holidays.Brazil(subdiv="SP", years=anos).keys())
    for ano in anos:
        p = easter(ano)
        datas |= {p - timedelta(days=48), p - timedelta(days=47), p + timedelta(days=60)}
    return datas


def tipo_semana(origens) -> dict:
    """Mesma regra de resultados_completos.tabela9: janela de efeito = feriado, 2 dias antes e 1 depois."""
    afetados = set()
    for f in datas_feriado():
        for k in range(-2, 2):
            afetados.add(f + timedelta(days=k))
    tem = lambda ini, dias: any((ini + timedelta(days=k)) in afetados for k in range(dias))
    return {o: ("feriado" if tem(o + timedelta(days=1), 7)
                else "pos_feriado" if tem(o - timedelta(days=6), 7) else "comum")
            for o in origens}


# ---------------------------------------------------------------------------
# Comparação pareada, nas mesmas origens
# ---------------------------------------------------------------------------

def ler(caminho) -> pd.DataFrame:
    d = pd.read_csv(caminho, encoding="utf-8-sig")
    d["ORIGEM"] = pd.to_datetime(d["ORIGEM"]).dt.date
    return d


def comparar(base: pd.DataFrame, novo: pd.DataFrame) -> dict:
    """SMAPE médio antes e depois (média das praças, como na Tabela 8), % de origens em que
    a nova versão erra menos e SMAPE de 7 dias por tipo de semana."""
    chaves = ["PRACA", "DIAS", "ORIGEM"]
    j = base[chaves + ["SMAPE"]].merge(novo[chaves + ["SMAPE"]], on=chaves, suffixes=("_A", "_D"))
    saida = {}
    for h in sorted(config.HORIZONTES.values()):
        s = j[j.DIAS == h]
        if s.empty:
            continue
        por_praca = s.groupby("PRACA")[["SMAPE_A", "SMAPE_D"]].mean()
        saida[f"ANTES_{h}"] = por_praca.SMAPE_A.mean()
        saida[f"DEPOIS_{h}"] = por_praca.SMAPE_D.mean()
        saida[f"PCT_MELHOR_{h}"] = 100 * (s.SMAPE_D < s.SMAPE_A).mean()
        saida[f"PRACAS_MELHOR_{h}"] = int((por_praca.SMAPE_D < por_praca.SMAPE_A).sum())
        saida[f"N_{h}"] = len(s)
    s7 = j[j.DIAS == 7].copy()
    if not s7.empty:
        tipos = tipo_semana(sorted(s7.ORIGEM.unique()))
        s7["TIPO"] = s7.ORIGEM.map(tipos)
        for t, g in s7.groupby("TIPO"):
            saida[f"ANTES_7_{t}"] = g.SMAPE_A.mean()
            saida[f"DEPOIS_7_{t}"] = g.SMAPE_D.mean()
    return saida
