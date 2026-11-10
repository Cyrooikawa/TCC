"""
Redes com o calendário como entrada (seção 4.6 do Guia dos modelos).

As classes abaixo são as MESMAS redes de redes_numpy.py, com um único acréscimo:
além dos volumes passados, a rede recebe o calendário (dia da semana e feriados)
dos dias da janela e dos dias a serem previstos. O calendário futuro é conhecido
de antemão, então nenhuma informação posterior à origem é usada.

Para aproveitar sem alteração o treino oficial (redes_numpy.treinar e prever),
cada exemplo chega como um vetor único, que a rede desempacota:

  recorrentes:  [volume (L) | calendário da janela (L x C) | calendário futuro (h x C)]
  N-HiTS:       [volume (L) | extras (E)]

Recorrentes: em cada dia da janela a célula lê [volume, calendário do dia]
(1 + C entradas em vez de 1). O calendário de cada dia futuro passa por uma
pequena MLP compartilhada (C -> M -> 1) cujo resultado é somado à previsão
daquele dia (efeito aditivo na escala log, isto é, proporcional ao volume).

N-HiTS: os extras são concatenados à entrada da MLP de cada bloco, como a
NeuralForecast faz com as variáveis exógenas (hist_exog / futr_exog).

Os gradientes foram conferidos por diferenças finitas (python redes_calendario.py).
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from redes_numpy import DT, NHITS, _sig, _uniforme  # noqa: E402


class RecorrenteCalendario:
    def __init__(self, celula, H, h, L, C, sementes, D=1, M=16, dtype=DT):
        self.celula, self.H, self.h, self.L, self.C, self.D, self.M, self.dtype = celula, H, h, L, C, D, M, dtype
        self.S = len(sementes)
        self.G = 4 if celula == "lstm" else 3
        self.F = 1 + C
        rngs = [np.random.default_rng(s) for s in sementes]
        k = 1.0 / math.sqrt(H)
        G, F = self.G, self.F
        p = {}
        p["Wx"] = _uniforme(rngs, k, (D, F, G * H), dtype)
        p["Wh"] = _uniforme(rngs, k, (D, H, G * H), dtype)
        if celula == "lstm":
            p["b"] = _uniforme(rngs, k, (D, 1, G * H), dtype) + _uniforme(rngs, k, (D, 1, G * H), dtype)
        else:
            p["bx"] = _uniforme(rngs, k, (D, 1, G * H), dtype)
            p["bh"] = _uniforme(rngs, k, (D, 1, G * H), dtype)
        kf = 1.0 / math.sqrt(D * H)
        p["Wo"] = _uniforme(rngs, kf, (D * H, h), dtype)
        p["bo"] = _uniforme(rngs, kf, (1, h), dtype)
        # MLP do calendário futuro (inicialização padrão de camada densa do PyTorch)
        p["V1"] = _uniforme(rngs, 1.0 / math.sqrt(C), (C, M), dtype)
        p["c1"] = _uniforme(rngs, 1.0 / math.sqrt(C), (1, M), dtype)
        p["V2"] = _uniforme(rngs, 1.0 / math.sqrt(M), (M, 1), dtype)
        self.p = p
        self._cache = None

    def _desempacotar(self, X):
        S, B, _ = X.shape
        L, C, h = self.L, self.C, self.h
        vol = X[:, :, :L, None]
        cal = X[:, :, L:L + L * C].reshape(S, B, L, C)
        fut = X[:, :, L + L * C:].reshape(S, B * h, C)
        return np.concatenate([vol, cal], axis=3), fut

    def forward(self, X, guardar=False):
        p, H, G, D, F = self.p, self.H, self.G, self.D, self.F
        S, B, _ = X.shape
        L, h = self.L, self.h
        K = S * D
        Xs, fut = self._desempacotar(X)                               # (S,B,L,F), (S,B*h,C)
        if D == 2:
            Xk = np.stack([Xs, Xs[:, :, ::-1]], axis=1)
        else:
            Xk = Xs[:, None]
        Xk = np.ascontiguousarray(Xk.reshape(K, B, L, F))
        Wx = p["Wx"].reshape(K, F, G * H)
        Wh = p["Wh"].reshape(K, H, G * H)
        hk = np.zeros((K, B, H), self.dtype)
        if self.celula == "lstm":
            b = p["b"].reshape(K, 1, G * H)
            c = np.zeros((K, B, H), self.dtype)
            if guardar:
                portas = np.empty((L, K, B, G * H), self.dtype)
                Cs = np.empty((L + 1, K, B, H), self.dtype)
                TCs = np.empty((L, K, B, H), self.dtype)
                Hs = np.empty((L + 1, K, B, H), self.dtype)
                Cs[0] = 0
                Hs[0] = 0
            for t in range(L):
                a = Xk[:, :, t] @ Wx + hk @ Wh + b
                a[..., :2 * H] = _sig(a[..., :2 * H])
                a[..., 2 * H:3 * H] = np.tanh(a[..., 2 * H:3 * H])
                a[..., 3 * H:] = _sig(a[..., 3 * H:])
                c = a[..., H:2 * H] * c + a[..., :H] * a[..., 2 * H:3 * H]
                tc = np.tanh(c)
                hk = a[..., 3 * H:] * tc
                if guardar:
                    portas[t], Cs[t + 1], TCs[t], Hs[t + 1] = a, c, tc, hk
            if guardar:
                self._cache = ("lstm", Xk, portas, Cs, TCs, Hs)
        else:
            bx = p["bx"].reshape(K, 1, G * H)
            bh = p["bh"].reshape(K, 1, G * H)
            if guardar:
                R = np.empty((L, K, B, H), self.dtype)
                Z = np.empty((L, K, B, H), self.dtype)
                N = np.empty((L, K, B, H), self.dtype)
                AHN = np.empty((L, K, B, H), self.dtype)
                Hs = np.empty((L + 1, K, B, H), self.dtype)
                Hs[0] = 0
            for t in range(L):
                ax = Xk[:, :, t] @ Wx + bx
                ah = hk @ Wh + bh
                r = _sig(ax[..., :H] + ah[..., :H])
                z = _sig(ax[..., H:2 * H] + ah[..., H:2 * H])
                n = np.tanh(ax[..., 2 * H:] + r * ah[..., 2 * H:])
                hk = (1 - z) * n + z * hk
                if guardar:
                    R[t], Z[t], N[t], AHN[t] = r, z, n, ah[..., 2 * H:]
                    Hs[t + 1] = hk
            if guardar:
                self._cache = ("gru", Xk, R, Z, N, AHN, Hs)
        hfin = hk.reshape(S, D, B, H).transpose(0, 2, 1, 3).reshape(S, B, D * H)
        z1 = fut @ p["V1"] + p["c1"]                                  # (S, B*h, M)
        a1 = np.maximum(z1, 0)
        efeito = (a1 @ p["V2"]).reshape(S, B, h)
        if guardar:
            self._hfin, self._cal = hfin, (fut, z1, a1)
        return hfin @ p["Wo"] + p["bo"] + efeito

    def backward(self, dY):
        p, H, G, D, F = self.p, self.H, self.G, self.D, self.F
        S, B, h = dY.shape
        K = S * D
        g = {}
        fut, z1, a1 = self._cal
        dE = dY.reshape(S, B * h, 1)
        g["V2"] = a1.transpose(0, 2, 1) @ dE
        dz1 = (dE @ p["V2"].transpose(0, 2, 1)) * (z1 > 0)
        g["V1"] = fut.transpose(0, 2, 1) @ dz1
        g["c1"] = dz1.sum(axis=1, keepdims=True)
        g["Wo"] = self._hfin.transpose(0, 2, 1) @ dY
        g["bo"] = dY.sum(axis=1, keepdims=True)
        dhfin = dY @ p["Wo"].transpose(0, 2, 1)
        dh = np.ascontiguousarray(dhfin.reshape(S, B, D, H).transpose(0, 2, 1, 3)).reshape(K, B, H)
        Wh = p["Wh"].reshape(K, H, G * H)
        WhT = Wh.transpose(0, 2, 1)
        dWx = np.zeros((K, F, G * H), self.dtype)
        dWh = np.zeros((K, H, G * H), self.dtype)
        if self.celula == "lstm":
            _, Xk, portas, Cs, TCs, Hs = self._cache
            L = portas.shape[0]
            db = np.zeros((K, 1, G * H), self.dtype)
            dc = np.zeros((K, B, H), self.dtype)
            da = np.empty((K, B, G * H), self.dtype)
            for t in range(L - 1, -1, -1):
                a = portas[t]
                i, f, gg, o = a[..., :H], a[..., H:2 * H], a[..., 2 * H:3 * H], a[..., 3 * H:]
                tc = TCs[t]
                dc = dc + dh * o * (1 - tc * tc)
                da[..., :H] = dc * gg * i * (1 - i)
                da[..., H:2 * H] = dc * Cs[t] * f * (1 - f)
                da[..., 2 * H:3 * H] = dc * i * (1 - gg * gg)
                da[..., 3 * H:] = dh * tc * o * (1 - o)
                dWx += Xk[:, :, t].transpose(0, 2, 1) @ da
                dWh += Hs[t].transpose(0, 2, 1) @ da
                db += da.sum(axis=1, keepdims=True)
                dh = da @ WhT
                dc = dc * f
            g["b"] = db.reshape(p["b"].shape)
        else:
            _, Xk, R, Z, N, AHN, Hs = self._cache
            L = R.shape[0]
            dbx = np.zeros((K, 1, G * H), self.dtype)
            dbh = np.zeros((K, 1, G * H), self.dtype)
            dax = np.empty((K, B, G * H), self.dtype)
            dah = np.empty((K, B, G * H), self.dtype)
            for t in range(L - 1, -1, -1):
                r, z, n, ahn, hprev = R[t], Z[t], N[t], AHN[t], Hs[t]
                dz = dh * (hprev - n)
                dnp = dh * (1 - z) * (1 - n * n)
                drp = dnp * ahn * r * (1 - r)
                dzp = dz * z * (1 - z)
                dax[..., :H], dax[..., H:2 * H], dax[..., 2 * H:] = drp, dzp, dnp
                dah[..., :H], dah[..., H:2 * H], dah[..., 2 * H:] = drp, dzp, dnp * r
                dWx += Xk[:, :, t].transpose(0, 2, 1) @ dax
                dbx += dax.sum(axis=1, keepdims=True)
                dWh += hprev.transpose(0, 2, 1) @ dah
                dbh += dah.sum(axis=1, keepdims=True)
                dh = dh * z + dah @ WhT
            g["bx"] = dbx.reshape(p["bx"].shape)
            g["bh"] = dbh.reshape(p["bh"].shape)
        g["Wx"] = dWx.reshape(p["Wx"].shape)
        g["Wh"] = dWh.reshape(p["Wh"].shape)
        self._cache = self._cal = None
        return g


class NHITSCalendario(NHITS):
    """N-HiTS oficial com E variáveis extras concatenadas à entrada da MLP de cada bloco."""

    def __init__(self, L, h, E, sementes, unidades=512, pools=(2, 2, 1), reducoes=(4, 2, 1), dtype=DT):
        self.L, self.h, self.E, self.dtype = L, h, E, dtype
        self.S = len(sementes)
        rngs = [np.random.default_rng(s) for s in sementes]
        self.blocos = []
        p = {}
        for j, (k, r) in enumerate(zip(pools, reducoes)):
            Lp = math.ceil(L / k)
            n_theta = max(h // r, 1)
            for nome, (fi, fo) in {"W1": (Lp + E, unidades), "W2": (unidades, unidades),
                                   "W3": (unidades, L + n_theta)}.items():
                lim = 1.0 / math.sqrt(fi)
                p[f"{nome}_{j}"] = _uniforme(rngs, lim, (fi, fo), dtype)
                p[f"b{nome[1]}_{j}"] = _uniforme(rngs, lim, (1, fo), dtype)
            from redes_numpy import matriz_interpolacao
            self.blocos.append({"k": k, "Lp": Lp, "n_theta": n_theta,
                                "M": matriz_interpolacao(n_theta, h, dtype)})
        self.p = p
        self._cache = None

    def forward(self, X, guardar=False):
        p, L, h = self.p, self.L, self.h
        vol, extras = X[:, :, :L], X[:, :, L:]
        res = np.ascontiguousarray(vol[:, :, ::-1])
        prev = np.repeat(vol[:, :, -1:], h, axis=2)
        caches = []
        for j, bl in enumerate(self.blocos):
            pooled, arg = self._pool(res, bl["k"])
            ent = np.concatenate([pooled, extras], axis=2)
            z1 = ent @ p[f"W1_{j}"] + p[f"b1_{j}"]
            a1 = np.maximum(z1, 0)
            z2 = a1 @ p[f"W2_{j}"] + p[f"b2_{j}"]
            a2 = np.maximum(z2, 0)
            th = a2 @ p[f"W3_{j}"] + p[f"b3_{j}"]
            back, nos = th[..., :L], th[..., L:]
            prev = prev + nos @ bl["M"]
            res = res - back
            if guardar:
                caches.append((ent, arg, z1, a1, z2, a2))
        if guardar:
            self._cache = caches
        return prev

    def backward(self, dY):
        p, L = self.p, self.L
        g = {}
        dres = np.zeros(dY.shape[:2] + (L,), self.dtype)
        for j in range(len(self.blocos) - 1, -1, -1):
            bl = self.blocos[j]
            ent, arg, z1, a1, z2, a2 = self._cache[j]
            dnos = dY @ bl["M"].T
            dth = np.concatenate([-dres, dnos], axis=2)
            g[f"W3_{j}"] = (a2.transpose(0, 2, 1) @ dth)
            g[f"b3_{j}"] = dth.sum(axis=1, keepdims=True)
            dz2 = (dth @ p[f"W3_{j}"].transpose(0, 2, 1)) * (z2 > 0)
            g[f"W2_{j}"] = a1.transpose(0, 2, 1) @ dz2
            g[f"b2_{j}"] = dz2.sum(axis=1, keepdims=True)
            dz1 = (dz2 @ p[f"W2_{j}"].transpose(0, 2, 1)) * (z1 > 0)
            g[f"W1_{j}"] = ent.transpose(0, 2, 1) @ dz1
            g[f"b1_{j}"] = dz1.sum(axis=1, keepdims=True)
            dpooled = (dz1 @ p[f"W1_{j}"].transpose(0, 2, 1))[..., :bl["Lp"]]
            dres = dres + self._pool_back(dpooled, arg, bl["k"], L)
        self._cache = None
        return g


# ---------------------------------------------------------------------------
# Conferência dos gradientes por diferenças finitas (em float64)
# ---------------------------------------------------------------------------

def _conferir(modelo, X, Y, n_amostras=12, eps=1e-6):
    rng = np.random.default_rng(0)

    def perda():
        return 0.5 * ((modelo.forward(X) - Y) ** 2).sum()

    Yh = modelo.forward(X, guardar=True)
    g = modelo.backward(Yh - Y)
    pior = 0.0
    for nome, v in modelo.p.items():
        for _ in range(n_amostras):
            i = tuple(rng.integers(0, d) for d in v.shape)
            antigo = v[i]
            v[i] = antigo + eps
            mais = perda()
            v[i] = antigo - eps
            menos = perda()
            v[i] = antigo
            num = (mais - menos) / (2 * eps)
            erro = abs(num - g[nome][i]) / max(1e-8, abs(num) + abs(g[nome][i]))
            if abs(num) > 1e-7:
                pior = max(pior, erro)
    return pior


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    S, B, L, h, C = 2, 3, 9, 5, 4
    ok = True
    for celula in ("lstm", "gru"):
        for D in (1, 2):
            m = RecorrenteCalendario(celula, 6, h, L, C, [1, 2], D=D, M=5, dtype=np.float64)
            X = rng.normal(size=(S, B, L + L * C + h * C))
            Y = rng.normal(size=(S, B, h))
            e = _conferir(m, X, Y)
            ok &= e < 1e-5
            print(f"{celula} D={D}: maior erro relativo do gradiente = {e:.2e}")
    E = 11
    m = NHITSCalendario(L, h, E, [1, 2], unidades=8, dtype=np.float64)
    X = rng.normal(size=(S, B, L + E))
    Y = rng.normal(size=(S, B, h))
    e = _conferir(m, X, Y)
    ok &= e < 1e-5
    print(f"nhits: maior erro relativo do gradiente = {e:.2e}")
    # Com as colunas de calendário zeradas, a parte recorrente tem de coincidir com a rede oficial.
    from redes_numpy import RecorrenteDireta
    a = RecorrenteDireta("gru", 6, h, [1, 2], D=2, dtype=np.float64)
    b = RecorrenteCalendario("gru", 6, h, L, C, [1, 2], D=2, M=5, dtype=np.float64)
    for k in ("Wh", "bx", "bh", "Wo", "bo"):
        b.p[k] = a.p[k].copy()
    b.p["Wx"][:, :, :1] = a.p["Wx"]
    b.p["V2"][:] = 0
    vol = rng.normal(size=(S, B, L))
    Xc = np.concatenate([vol, np.zeros((S, B, L * C + h * C))], axis=2)
    dif = np.abs(a.forward(vol) - b.forward(Xc)).max()
    ok &= dif < 1e-12
    print(f"equivalência com a rede oficial (calendário zerado): diferença máxima = {dif:.1e}")
    print("GRADIENTES CONFERIDOS" if ok else "FALHA NA CONFERÊNCIA")
