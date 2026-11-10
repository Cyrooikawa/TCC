"""
Experimentos de refinamento das redes neurais (seções 4.2 a 4.6 do Guia dos modelos).

Cada variante muda UMA coisa em relação à configuração oficial de redes_neurais.py.
O treino (redes_numpy.treinar), as origens, as métricas e as cinco sementes são os
oficiais. Nada é gravado em resultados/: tudo vai para experimentos/resultados/redes/.

Variantes:
    base               configuração oficial (só para conferir a reprodução neste ambiente)
    vies_esquecimento  LSTM com o viés da porta de esquecimento iniciado em 1 (Jozefowicz et al., 2015)
    paciencia40        paciência 40 e até 400 épocas
    janela91           janela de entrada de 91 dias também no horizonte de 7 dias
    unid32, unid128    32 ou 128 unidades nas recorrentes
    nhits256           N-HiTS com 256 unidades por camada
    calendario         dia da semana e feriados (janela e dias previstos) como entrada

Uso:
    python exp_redes.py calendario GRU 7      uma combinação
    python exp_redes.py --fila 3              toda a fila de experimentos, 3 por vez
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import comum
from comum import config, metricas
import redes_neurais as oficial
from redes_numpy import NHITS, RecorrenteDireta, prever, treinar
from redes_calendario import NHITSCalendario, RecorrenteCalendario

SAIDA = comum.SAIDA / "redes"
SAIDA.mkdir(exist_ok=True)

VARIANTES = {
    "base": {},
    "vies_esquecimento": {"vies_esquecimento": 1.0},
    "paciencia40": {"paciencia": 40, "max_epocas": 400},
    "janela91": {"janela": {7: 91}},
    "unid32": {"unidades": 32},
    "unid128": {"unidades": 128},
    "nhits256": {"unidades_nhits": 256},
    "calendario": {"calendario": True},
}

# Fila, da maior prioridade para a menor (roteiro da seção 7 do guia).
FILA = (
    [("vies_esquecimento", "LSTM", 180), ("paciencia40", "LSTM", 180)]
    + [("calendario", m, 7) for m in ("NHITS", "GRU", "LSTM", "BIGRU", "BILSTM")]
    + [("calendario", "NHITS", 90), ("calendario", "NHITS", 180), ("calendario", "GRU", 90), ("calendario", "GRU", 180)]
    + [("janela91", m, 7) for m in ("NHITS", "GRU", "LSTM", "BIGRU", "BILSTM")]
    + [("nhits256", "NHITS", h) for h in (7, 90, 180)]
    + [(v, "GRU", h) for h in (7, 90, 180) for v in ("unid32", "unid128")]
)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def matriz_calendario(datas: pd.DatetimeIndex) -> np.ndarray:
    """10 colunas 0/1 por dia: 7 do dia da semana, feriado, véspera de feriado e dia seguinte a feriado.
    Mesmo calendário do Prophet oficial (nacionais, SP, Carnaval e Corpus Christi)."""
    fer = comum.datas_feriado()
    d = [x.date() for x in datas]
    M = np.zeros((len(d), 10), np.float32)
    for i, x in enumerate(d):
        M[i, x.weekday()] = 1
        M[i, 7] = x in fer
        M[i, 8] = (x + timedelta(days=1)) in fer
        M[i, 9] = (x - timedelta(days=1)) in fer
    return M


def montar_dados(series, L, h, modelo, calendario):
    """Igual a redes_neurais.montar_dados; com calendario=True, acrescenta as colunas do calendário."""
    n = len(series[0][2])
    n_treino = int(n * config.PROPORCAO_TREINO)
    inicio_val = n_treino - int(round(oficial.FRACAO_VALIDACAO * n_treino))
    # O calendário precisa cobrir os h dias depois da última origem de treino/teste (todos dentro da série).
    cal = matriz_calendario(series[0][2].index) if calendario else None

    def entrada(z, t):
        vol = z[t - L + 1:t + 1]
        if not calendario:
            return vol
        hist, fut = cal[t - L + 1:t + 1], cal[t + 1:t + h + 1]
        if modelo == "NHITS":
            # feriados da janela, dia da semana do 1º dia previsto, feriados dos dias previstos
            return np.concatenate([vol, hist[:, 7:].ravel(), cal[t + 1, :7], fut[:, 7:].ravel()])
        return np.concatenate([vol, hist.ravel(), fut.ravel()])

    Xtr, Ytr, Xva, Yva, teste, escalas, contagem = [], [], [], [], [], {}, []
    for praca, grupo, y in series:
        v = y.values.astype(float)
        lv = np.log1p(v)
        mn, mx = np.nanmin(lv[:n_treino]), np.nanmax(lv[:n_treino])
        z = (lv - mn) / (mx - mn)
        escalas[praca] = (mn, mx)
        nan_acum = np.concatenate([[0], np.cumsum(np.isnan(z))])
        n_tr = n_va = desc = 0
        for t in range(L - 1, n_treino - h):
            if nan_acum[t + h + 1] - nan_acum[t - L + 1] > 0:
                desc += 1
                continue
            if t + h < inicio_val:
                Xtr.append(entrada(z, t)); Ytr.append(z[t + 1:t + h + 1]); n_tr += 1
            elif t + 1 >= inicio_val:
                Xva.append(entrada(z, t)); Yva.append(z[t + 1:t + h + 1]); n_va += 1
        for o in comum.origens_teste(n):
            if o + h > n - 1:
                continue
            assert not np.isnan(z[o - L + 1:o + 1]).any(), (praca, o)
            teste.append({"praca": praca, "grupo": grupo, "origem": o, "data_origem": y.index[o].date(),
                          "entrada": entrada(z, o), "real": v[o + 1:o + 1 + h]})
        contagem.append({"praca": praca, "treino": n_tr, "validacao": n_va, "descartadas": desc})
    f = lambda a: np.asarray(a, dtype=np.float32)
    return f(Xtr), f(Ytr), f(Xva), f(Yva), teste, escalas, contagem


def rodar(variante, modelo, h):
    opc = VARIANTES[variante]
    L = opc.get("janela", {}).get(h, oficial.JANELA_ENTRADA[h])
    unidades = opc.get("unidades", oficial.UNIDADES_RECORRENTES)
    unidades_nhits = opc.get("unidades_nhits", oficial.UNIDADES_NHITS)
    paciencia = opc.get("paciencia", oficial.PACIENCIA)
    max_epocas = opc.get("max_epocas", oficial.MAX_EPOCAS)
    calendario = opc.get("calendario", False)
    sementes = oficial.SEMENTES
    rotulo = f"{variante}_{modelo}_{h}"

    series = oficial.carregar()
    Xtr, Ytr, Xva, Yva, teste, escalas, contagem = montar_dados(series, L, h, modelo, calendario)
    log(f"[{rotulo}] L={L}: {len(Xtr)} janelas de treino, {len(Xva)} de validação, {len(teste)} previsões de teste, "
        f"{Xtr.shape[1]} entradas por exemplo")

    celula = "lstm" if "LSTM" in modelo else "gru"
    D = 2 if modelo.startswith("BI") else 1
    if modelo == "NHITS":
        rede = (NHITSCalendario(L, h, Xtr.shape[1] - L, sementes, unidades=unidades_nhits) if calendario
                else NHITS(L, h, sementes, unidades=unidades_nhits))
    elif calendario:
        rede = RecorrenteCalendario(celula, unidades, h, L, 10, sementes, D=D)
    else:
        rede = RecorrenteDireta(celula, unidades, h, sementes, D=D)
    if "vies_esquecimento" in opc:
        assert celula == "lstm"
        H = unidades
        rede.p["b"][..., H:2 * H] = opc["vies_esquecimento"]

    lote = oficial.LOTE["NHITS" if modelo == "NHITS" else "RECORRENTES"]
    t0 = time.time()
    historico, melhor_epoca, melhor_val = treinar(
        rede, Xtr, Ytr, Xva, Yva, sementes, lote=lote, max_epocas=max_epocas, paciencia=paciencia,
        lr=oficial.TAXA_APRENDIZADO, recorte=oficial.RECORTE_GRADIENTE,
        log=lambda m: log(f"[{rotulo}]{m}"))
    t_treino = time.time() - t0

    Xte = np.asarray([r["entrada"] for r in teste], dtype=np.float32)
    P = prever(rede, Xte).astype(np.float64)
    linhas = []
    prev_orig = np.empty_like(P)
    for i, r in enumerate(teste):
        mn, mx = escalas[r["praca"]]
        prev = np.expm1(P[:, i, :] * (mx - mn) + mn)
        prev_orig[:, i, :] = prev
        por_semente = [metricas.todas(r["real"], prev[s], config.JANELA_DTW) for s in range(len(sementes))]
        media = {k: float(np.mean([m[k] for m in por_semente])) for k in por_semente[0]}
        media.update({f"SMAPE_S{s}": m["SMAPE"] for s, m in zip(sementes, por_semente)})
        linhas.append({**media, "MODELO": modelo, "GRUPO": r["grupo"], "PRACA": r["praca"],
                       "HORIZONTE": comum.NOME_HORIZONTE[h], "DIAS": h, "ORIGEM": r["data_origem"]})
    d = pd.DataFrame(linhas)
    d.to_csv(SAIDA / f"{rotulo}_por_origem.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(SAIDA / f"{rotulo}_previsoes.npz", previsao=prev_orig.astype(np.float32),
                        praca=np.array([r["praca"] for r in teste]),
                        origem=np.array([str(r["data_origem"]) for r in teste]))
    resumo = {
        "variante": variante, "modelo": modelo, "horizonte": h, "janela_entrada": L, "opcoes": opc,
        "sementes": sementes, "lote": lote, "max_epocas": max_epocas, "paciencia": paciencia,
        "janelas": contagem, "melhor_epoca": melhor_epoca.tolist(), "melhor_validacao": melhor_val.tolist(),
        "epocas_executadas": len(historico), "tempo_treino_s": round(t_treino, 1),
        "n_parametros_por_semente": int(sum(v[0].size for v in rede.p.values())),
        "smape_por_semente": [float(d[f"SMAPE_S{s}"].mean()) for s in sementes],
        "smape_medio": float(d.SMAPE.mean()),
        "historico": [{"epoca": e["epoca"], "treino": e["treino"].tolist(), "validacao": e["validacao"].tolist()}
                      for e in historico],
    }
    (SAIDA / f"{rotulo}_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[{rotulo}] FIM: treino {t_treino/60:.1f} min, épocas {len(historico)}, melhor época {melhor_epoca.tolist()}, "
        f"validação {melhor_val.mean():.4f}, SMAPE médio {d.SMAPE.mean():.2f}%")


def _rodar(item):
    if (SAIDA / f"{item[0]}_{item[1]}_{item[2]}_resumo.json").exists():
        log(f"[{item[0]}_{item[1]}_{item[2]}] já existe, pulando")
        return
    rodar(*item)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--fila":
        from multiprocessing import Pool
        fila = FILA
        if len(a) > 2:                       # --fila 3 calendario  -> só uma variante
            fila = [f for f in FILA if f[0] in a[2:]]
        with Pool(int(a[1])) as pool:
            pool.map(_rodar, fila, chunksize=1)
    else:
        rodar(a[0], a[1].upper(), int(a[2]))
