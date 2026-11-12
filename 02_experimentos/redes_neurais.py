"""
Treina e avalia as redes neurais do TCC (LSTM, GRU, Bi-LSTM, Bi-GRU e N-HiTS) nos
três horizontes, com o mesmo desenho dos modelos estatísticos:

- série diária de 2021-01-01 a 2026-07-31, 70% iniciais para treino (1.426 dias);
- log(1 + x) e Min-Max com mínimo e máximo do treino de cada praça;
- janelas de treino sem lacunas (dias ausentes ou invalidados descartam a janela);
- validação: os 20% finais do treino (285 dias), só para a parada antecipada;
- saída direta dos h dias; parâmetros estimados uma vez no treino e mantidos;
  em cada origem do teste muda apenas a janela de entrada (dados até a origem);
- mesmas origens (87, 75 e 62) e mesmas métricas (metricas.py) dos demais modelos;
- cinco inicializações (sementes); a métrica de cada origem é a média das cinco.

Coloque este arquivo e redes_numpy.py na pasta 02_codigo_e_resultados.

Uso:
    python redes_neurais.py                 roda as 15 combinações (5 redes x 3 horizontes), uma por vez
    python redes_neurais.py --processos 2   roda duas combinações ao mesmo tempo (use até o nº de núcleos)
    python redes_neurais.py GRU 7           roda só uma combinação
    python redes_neurais.py --juntar        só junta os arquivos já gerados

Saídas: resultados/redes/<MODELO>_<h>_* (por origem, por semente, previsões, resumo com
histórico de treino) e, ao final, resultados/<modelo>_metricas_por_origem.csv, no mesmo
formato dos demais modelos (lido por resultados_completos.py).
Tempo: cerca de 4 horas e 15 minutos de processamento no total (as 15 combinações);
com --processos 2, pouco mais de 2 horas. Não usa placa de vídeo.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")        # uma combinação por núcleo
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
import config  # noqa: E402
import metricas  # noqa: E402
from redes_numpy import NHITS, RecorrenteDireta, prever, treinar  # noqa: E402

SAIDA = PROJ / "resultados" / "redes"
SAIDA.mkdir(parents=True, exist_ok=True)
MODELOS = ["LSTM", "GRU", "BILSTM", "BIGRU", "NHITS"]

# Hiperparâmetros (fixos, sem busca, como no Prophet)
JANELA_ENTRADA = {7: 28, 90: 91, 180: 182}   # 4, 13 e 26 semanas
FRACAO_VALIDACAO = 0.20
UNIDADES_RECORRENTES = 64
UNIDADES_NHITS = 512
# Lote: 32 janelas nas recorrentes (padrão batch_size=32 da NeuralForecast);
# 256 no N-HiTS, que é totalmente conectado (a NeuralForecast usa 1.024 janelas).
LOTE = {"RECORRENTES": 32, "NHITS": 256}
MAX_EPOCAS = 200
PACIENCIA = 20
TAXA_APRENDIZADO = 1e-3
RECORTE_GRADIENTE = 1.0
SEMENTES = [1, 2, 3, 4, 5]
NOME_HORIZONTE = {v: k for k, v in config.HORIZONTES.items()}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def carregar():
    df = pd.read_csv(PROJ / "dados" / "processados" / "series_diarias.csv", parse_dates=["DATA"])
    alvos = [(p, "NUCLEO") for p in config.PRACAS_NUCLEO] + \
            [(p, "UNIDIRECIONAL") for p in config.PRACAS_UNIDIRECIONAIS]
    series = []
    for praca, grupo in alvos:
        sub = df[(df["PRACA"] == praca) & (df["DATA"] >= config.DATA_INICIO_MODELAGEM)]
        y = sub.set_index("DATA")["VOLUME_TOTAL"].sort_index().asfreq("D")
        series.append((praca, grupo, y))
    return series


def montar_dados(series, L, h):
    n = len(series[0][2])
    n_treino = int(n * config.PROPORCAO_TREINO)
    inicio_val = n_treino - int(round(FRACAO_VALIDACAO * n_treino))
    Xtr, Ytr, Xva, Yva, teste, escalas = [], [], [], [], [], {}
    contagem = []
    for praca, grupo, y in series:
        assert len(y) == n
        v = y.values.astype(float)
        lv = np.log1p(v)
        mn, mx = np.nanmin(lv[:n_treino]), np.nanmax(lv[:n_treino])
        z = (lv - mn) / (mx - mn)
        escalas[praca] = (mn, mx)
        nan_acum = np.concatenate([[0], np.cumsum(np.isnan(z))])
        descartadas = 0
        n_tr = n_va = 0
        for t in range(L - 1, n_treino - h):
            if nan_acum[t + h + 1] - nan_acum[t - L + 1] > 0:
                descartadas += 1
                continue
            entrada, alvo = z[t - L + 1:t + 1], z[t + 1:t + h + 1]
            if t + h < inicio_val:
                Xtr.append(entrada); Ytr.append(alvo); n_tr += 1
            elif t + 1 >= inicio_val:
                Xva.append(entrada); Yva.append(alvo); n_va += 1
        for o in metricas.origens(n, config.PROPORCAO_TREINO, config.PASSO_ORIGENS_DIAS,
                                  min(config.HORIZONTES.values())):
            if o + h > n - 1:
                continue
            entrada = z[o - L + 1:o + 1]
            assert not np.isnan(entrada).any(), (praca, o)
            teste.append({"praca": praca, "grupo": grupo, "origem": o,
                          "data_origem": y.index[o].date(), "entrada": entrada,
                          "real": v[o + 1:o + 1 + h]})
        contagem.append({"praca": praca, "treino": n_tr, "validacao": n_va, "descartadas": descartadas})
    f = lambda a: np.asarray(a, dtype=np.float32)
    info = {"n": n, "n_treino": n_treino, "inicio_validacao": inicio_val,
            "data_inicio_validacao": str(series[0][2].index[inicio_val].date()),
            "janelas": contagem}
    return f(Xtr), f(Ytr), f(Xva), f(Yva), teste, escalas, info


def criar(modelo, L, h):
    if modelo == "NHITS":
        return NHITS(L, h, SEMENTES, unidades=UNIDADES_NHITS)
    celula = "lstm" if "LSTM" in modelo else "gru"
    D = 2 if modelo.startswith("BI") else 1
    return RecorrenteDireta(celula, UNIDADES_RECORRENTES, h, SEMENTES, D=D)


def main(modelo, h):
    L = JANELA_ENTRADA[h]
    series = carregar()
    Xtr, Ytr, Xva, Yva, teste, escalas, info = montar_dados(series, L, h)
    log(f"{modelo} h={h} L={L}: {len(Xtr)} janelas de treino, {len(Xva)} de validação, {len(teste)} previsões de teste")
    rede = criar(modelo, L, h)
    paciencia, max_epocas = config.PARADA_ESPECIFICA.get((modelo, h), (PACIENCIA, MAX_EPOCAS))
    t0 = time.time()
    historico, melhor_epoca, melhor_val = treinar(
        rede, Xtr, Ytr, Xva, Yva, SEMENTES, lote=LOTE["NHITS" if modelo == "NHITS" else "RECORRENTES"], max_epocas=max_epocas,
        paciencia=paciencia, lr=TAXA_APRENDIZADO, recorte=RECORTE_GRADIENTE, log=log)
    t_treino = time.time() - t0
    t1 = time.time()
    Xte = np.asarray([r["entrada"] for r in teste], dtype=np.float32)
    P = prever(rede, Xte).astype(np.float64)                      # (S, N, h), escala normalizada
    t_previsao = time.time() - t1
    linhas, linhas_sem = [], []
    prev_orig = np.empty_like(P)
    for i, r in enumerate(teste):
        mn, mx = escalas[r["praca"]]
        prev = np.expm1(P[:, i, :] * (mx - mn) + mn)               # volta à escala original
        prev_orig[:, i, :] = prev
        por_semente = [metricas.todas(r["real"], prev[s], config.JANELA_DTW) for s in range(len(SEMENTES))]
        for s, m in zip(SEMENTES, por_semente):
            linhas_sem.append({**m, "SEMENTE": s, "MODELO": modelo, "GRUPO": r["grupo"], "PRACA": r["praca"],
                               "HORIZONTE": NOME_HORIZONTE[h], "DIAS": h, "ORIGEM": r["data_origem"]})
        media = {k: float(np.mean([m[k] for m in por_semente])) for k in por_semente[0]}
        linhas.append({**media, "MODELO": modelo, "GRUPO": r["grupo"], "PRACA": r["praca"],
                       "HORIZONTE": NOME_HORIZONTE[h], "DIAS": h, "ORIGEM": r["data_origem"]})
    base = SAIDA / f"{modelo}_{h}"
    pd.DataFrame(linhas).to_csv(f"{base}_por_origem.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(linhas_sem).to_csv(f"{base}_por_semente.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(f"{base}_previsoes.npz", previsao=prev_orig.astype(np.float32),
                        praca=np.array([r["praca"] for r in teste]),
                        origem=np.array([str(r["data_origem"]) for r in teste]))
    resumo = {
        "modelo": modelo, "horizonte": h, "janela_entrada": L, "sementes": SEMENTES,
        "unidades": UNIDADES_NHITS if modelo == "NHITS" else UNIDADES_RECORRENTES,
        "lote": LOTE["NHITS" if modelo == "NHITS" else "RECORRENTES"], "max_epocas": max_epocas, "paciencia": paciencia, "taxa_aprendizado": TAXA_APRENDIZADO,
        "recorte_gradiente": RECORTE_GRADIENTE, "dados": info,
        "melhor_epoca": melhor_epoca.tolist(), "melhor_validacao": melhor_val.tolist(),
        "epocas_executadas": len(historico),
        "tempo_treino_s": round(t_treino, 1), "tempo_previsao_s": round(t_previsao, 2),
        "n_parametros_por_semente": int(sum(v[0].size for v in rede.p.values())),
        "historico": [{"epoca": e["epoca"], "treino": e["treino"].tolist(), "validacao": e["validacao"].tolist()}
                      for e in historico],
    }
    (Path(f"{base}_resumo.json")).write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    smape = pd.DataFrame(linhas).SMAPE.mean()
    log(f"{modelo} h={h}: treino {t_treino/60:.1f} min, épocas {len(historico)}, melhor época {melhor_epoca.tolist()}, SMAPE médio {smape:.2f}%")


def juntar():
    """Junta as combinações já rodadas em resultados/<modelo>_metricas_por_origem.csv."""
    for modelo in MODELOS:
        partes = [SAIDA / f"{modelo}_{h}_por_origem.csv" for h in sorted(JANELA_ENTRADA)]
        if not all(p.exists() for p in partes):
            faltam = [p.name for p in partes if not p.exists()]
            print(f"{modelo}: ainda faltam {', '.join(faltam)}")
            continue
        d = pd.concat([pd.read_csv(p) for p in partes], ignore_index=True)
        destino = PROJ / "resultados" / f"{modelo.lower()}_metricas_por_origem.csv"
        d.to_csv(destino, index=False, encoding="utf-8-sig")
        print(f"{modelo}: {len(d)} linhas em {destino.name}")


def _rodar(args):
    modelo, h = args
    main(modelo, h)


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    if argumentos == ["--juntar"]:
        juntar()
    elif len(argumentos) == 2 and not argumentos[0].startswith("--"):
        main(argumentos[0].upper(), int(argumentos[1]))
        juntar()
    else:
        processos = int(argumentos[argumentos.index("--processos") + 1]) if "--processos" in argumentos else 1
        # Das combinações mais longas para as mais curtas (equilibra os processos).
        fila = [(m, h) for h in (90, 180, 7) for m in ("BILSTM", "BIGRU", "LSTM", "GRU", "NHITS")]
        if processos > 1:
            from multiprocessing import Pool
            with Pool(processos) as pool:
                pool.map(_rodar, fila, chunksize=1)
        else:
            for item in fila:
                _rodar(item)
        juntar()
