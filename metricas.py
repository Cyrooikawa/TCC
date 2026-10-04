"""
Métricas de avaliação, compartilhadas por todos os modelos (seção 3.2.6 e
3.2.7 da monografia). Usar sempre este arquivo garante que ARIMA, Prophet,
LSTM, GRU e N-HiTS sejam avaliados exatamente da mesma forma.

Todas as funções ignoram posições em que o valor real está ausente (NaN).

Este arquivo não é executado sozinho.
"""

import numpy as np


def _alinhar(real, previsto):
    real = np.asarray(real, dtype=float)
    previsto = np.asarray(previsto, dtype=float)
    validos = ~np.isnan(real) & ~np.isnan(previsto)
    return real[validos], previsto[validos]


def mae(real, previsto):
    r, p = _alinhar(real, previsto)
    return float(np.mean(np.abs(r - p))) if len(r) else np.nan


def rmse(real, previsto):
    r, p = _alinhar(real, previsto)
    return float(np.sqrt(np.mean((r - p) ** 2))) if len(r) else np.nan


def mape(real, previsto):
    r, p = _alinhar(real, previsto)
    validos = r != 0
    r, p = r[validos], p[validos]
    return float(np.mean(np.abs((r - p) / r)) * 100) if len(r) else np.nan


def smape(real, previsto):
    r, p = _alinhar(real, previsto)
    denominador = (np.abs(r) + np.abs(p)) / 2
    validos = denominador != 0
    r, p, denominador = r[validos], p[validos], denominador[validos]
    return float(np.mean(np.abs(r - p) / denominador) * 100) if len(r) else np.nan


def dtw(real, previsto, janela: int = 7):
    """
    Dynamic Time Warping com restrição de banda de Sakoe-Chiba: um ponto só
    pode ser alinhado com pontos até `janela` posições de distância. Mede
    semelhança de FORMA entre as curvas, tolerando pequenos deslocamentos no
    tempo. Retorna a distância acumulada dividida pelo tamanho da série, para
    ficar comparável entre horizontes de tamanhos diferentes.
    """
    r, p = _alinhar(real, previsto)
    n = len(r)
    if n == 0:
        return np.nan
    custo = np.full((n + 1, n + 1), np.inf)
    custo[0, 0] = 0.0
    for i in range(1, n + 1):
        inicio = max(1, i - janela)
        fim = min(n, i + janela)
        for j in range(inicio, fim + 1):
            d = abs(r[i - 1] - p[j - 1])
            custo[i, j] = d + min(custo[i - 1, j], custo[i, j - 1], custo[i - 1, j - 1])
    return float(custo[n, n] / n)


def todas(real, previsto, janela_dtw: int = 7) -> dict:
    return {
        "MAE": mae(real, previsto),
        "RMSE": rmse(real, previsto),
        "MAPE": mape(real, previsto),
        "SMAPE": smape(real, previsto),
        "DTW": dtw(real, previsto, janela_dtw),
    }
