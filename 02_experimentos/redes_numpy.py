"""
Redes neurais do TCC (LSTM, GRU, Bi-LSTM, Bi-GRU e N-HiTS) implementadas em NumPy.

Por que NumPy: o ambiente em que estes resultados foram gerados não tem PyTorch
nem NeuralForecast. As arquiteturas seguem as definições dessas bibliotecas
(equações das células LSTM e GRU do PyTorch, inicialização padrão do PyTorch,
bloco do N-HiTS com MaxPool, MLP e interpolação linear como na NeuralForecast).

Todas as sementes de uma mesma configuração são treinadas em paralelo: cada
parâmetro tem um eixo inicial S (uma fatia por semente). Isso não mistura as
sementes: cada uma tem sua inicialização, sua ordem de lotes, seu recorte de
gradiente e sua parada antecipada.
"""

import math
import numpy as np

DT = np.float32


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _uniforme(rngs, limite, forma, dtype):
    """Uma fatia por semente, cada uma com seu gerador (reprodutível)."""
    return np.stack([r.uniform(-limite, limite, size=forma) for r in rngs]).astype(dtype)


# ---------------------------------------------------------------------------
# Redes recorrentes (uma camada, uni ou bidirecional, saída direta de h passos)
# ---------------------------------------------------------------------------

class RecorrenteDireta:
    """
    celula: "lstm" ou "gru". D = 1 (unidirecional) ou 2 (bidirecional).
    A rede lê a janela de entrada (L dias) e a camada de saída densa produz os
    h dias do horizonte de uma só vez (estratégia direta). Na versão
    bidirecional, a segunda direção lê a MESMA janela de trás para frente; a
    previsão nunca usa dado posterior à origem.
    """

    def __init__(self, celula, H, h, sementes, D=1, dtype=DT):
        self.celula, self.H, self.h, self.D, self.dtype = celula, H, h, D, dtype
        self.S = len(sementes)
        self.G = 4 if celula == "lstm" else 3
        rngs = [np.random.default_rng(s) for s in sementes]
        k = 1.0 / math.sqrt(H)
        G, S = self.G, self.S
        p = {}
        p["Wx"] = _uniforme(rngs, k, (D, 1, G * H), dtype)
        p["Wh"] = _uniforme(rngs, k, (D, H, G * H), dtype)
        if celula == "lstm":
            # No PyTorch há b_ih + b_hh; a soma é um único viés efetivo.
            p["b"] = _uniforme(rngs, k, (D, 1, G * H), dtype) + _uniforme(rngs, k, (D, 1, G * H), dtype)
        else:
            p["bx"] = _uniforme(rngs, k, (D, 1, G * H), dtype)
            p["bh"] = _uniforme(rngs, k, (D, 1, G * H), dtype)
        kf = 1.0 / math.sqrt(D * H)
        p["Wo"] = _uniforme(rngs, kf, (D * H, h), dtype)
        p["bo"] = _uniforme(rngs, kf, (1, h), dtype)
        self.p = p
        self._cache = None

    # -- auxiliares -------------------------------------------------------
    def _entrada_por_direcao(self, X):
        S, B, L = X.shape
        if self.D == 2:
            Xk = np.stack([X, X[:, :, ::-1]], axis=1)
        else:
            Xk = X[:, None]
        return np.ascontiguousarray(Xk.reshape(S * self.D, B, L))

    def forward(self, X, guardar=False):
        """X: (S, B, L) -> previsão (S, B, h), na escala normalizada."""
        p, H, G, D = self.p, self.H, self.G, self.D
        S, B, L = X.shape
        K = S * D
        Xk = self._entrada_por_direcao(X)
        Wx = p["Wx"].reshape(K, 1, G * H)
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
                a = Xk[:, :, t:t + 1] * Wx + hk @ Wh + b
                a[..., :2 * H] = _sig(a[..., :2 * H])
                a[..., 2 * H:3 * H] = np.tanh(a[..., 2 * H:3 * H])
                a[..., 3 * H:] = _sig(a[..., 3 * H:])
                c = a[..., H:2 * H] * c + a[..., :H] * a[..., 2 * H:3 * H]
                tc = np.tanh(c)
                hk = a[..., 3 * H:] * tc
                if guardar:
                    portas[t] = a
                    Cs[t + 1] = c
                    TCs[t] = tc
                    Hs[t + 1] = hk
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
                ax = Xk[:, :, t:t + 1] * Wx + bx
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
        if guardar:
            self._hfin = hfin
        return hfin @ p["Wo"] + p["bo"]

    def backward(self, dY):
        """dY: (S, B, h) = derivada da perda em relação à previsão."""
        p, H, G, D = self.p, self.H, self.G, self.D
        S, B, _ = dY.shape
        K = S * D
        g = {}
        g["Wo"] = self._hfin.transpose(0, 2, 1) @ dY
        g["bo"] = dY.sum(axis=1, keepdims=True)
        dhfin = dY @ p["Wo"].transpose(0, 2, 1)
        dh = np.ascontiguousarray(dhfin.reshape(S, B, D, H).transpose(0, 2, 1, 3)).reshape(K, B, H)
        Wh = p["Wh"].reshape(K, H, G * H)
        WhT = Wh.transpose(0, 2, 1)
        dWx = np.zeros((K, 1, G * H), self.dtype)
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
                dWx += Xk[:, :, t:t + 1].transpose(0, 2, 1) @ da
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
                dax[..., :H] = drp
                dax[..., H:2 * H] = dzp
                dax[..., 2 * H:] = dnp
                dah[..., :H] = drp
                dah[..., H:2 * H] = dzp
                dah[..., 2 * H:] = dnp * r
                dWx += Xk[:, :, t:t + 1].transpose(0, 2, 1) @ dax
                dbx += dax.sum(axis=1, keepdims=True)
                dWh += hprev.transpose(0, 2, 1) @ dah
                dbh += dah.sum(axis=1, keepdims=True)
                dh = dh * z + dah @ WhT
            g["bx"] = dbx.reshape(p["bx"].shape)
            g["bh"] = dbh.reshape(p["bh"].shape)
        g["Wx"] = dWx.reshape(p["Wx"].shape)
        g["Wh"] = dWh.reshape(p["Wh"].shape)
        self._cache = None
        return g


# ---------------------------------------------------------------------------
# N-HiTS (pilhas de blocos MLP com MaxPool e interpolação linear)
# ---------------------------------------------------------------------------

def matriz_interpolacao(n_entrada, n_saida, dtype=DT):
    """Mesma interpolação linear de torch.nn.functional.interpolate
    (mode='linear', align_corners=False), escrita como matriz."""
    M = np.zeros((n_entrada, n_saida), dtype=np.float64)
    escala = n_entrada / n_saida
    for j in range(n_saida):
        src = max((j + 0.5) * escala - 0.5, 0.0)
        i0 = int(math.floor(src))
        i0 = min(i0, n_entrada - 1)
        i1 = min(i0 + 1, n_entrada - 1)
        l1 = src - i0
        M[i0, j] += 1 - l1
        M[i1, j] += l1
    return M.astype(dtype)


class NHITS:
    """
    Três pilhas de um bloco cada (padrão da NeuralForecast): MaxPool com
    núcleos (2, 2, 1), MLP com duas camadas ocultas ReLU e coeficientes de
    previsão em resolução reduzida (h//4, h//2, h), interpolados linearmente
    para os h dias. A previsão parte do último valor observado (como na
    NeuralForecast) e cada bloco soma a sua parte; o resíduo da janela de
    entrada passa de um bloco ao seguinte.
    """

    def __init__(self, L, h, sementes, unidades=512, pools=(2, 2, 1), reducoes=(4, 2, 1), dtype=DT):
        self.L, self.h, self.dtype = L, h, dtype
        self.S = len(sementes)
        rngs = [np.random.default_rng(s) for s in sementes]
        self.blocos = []
        p = {}
        for j, (k, r) in enumerate(zip(pools, reducoes)):
            Lp = math.ceil(L / k)
            n_theta = max(h // r, 1)
            for nome, (fi, fo) in {"W1": (Lp, unidades), "W2": (unidades, unidades),
                                   "W3": (unidades, L + n_theta)}.items():
                lim = 1.0 / math.sqrt(fi)
                p[f"{nome}_{j}"] = _uniforme(rngs, lim, (fi, fo), dtype)
                p[f"b{nome[1]}_{j}"] = _uniforme(rngs, lim, (1, fo), dtype)
            self.blocos.append({"k": k, "Lp": Lp, "n_theta": n_theta,
                                "M": matriz_interpolacao(n_theta, h, dtype)})
        self.p = p
        self._cache = None

    @staticmethod
    def _pool(x, k):
        S, B, L = x.shape
        if k == 1:
            return x, None
        Lp = math.ceil(L / k)
        pad = Lp * k - L
        if pad:
            x = np.concatenate([x, np.full((S, B, pad), -np.inf, x.dtype)], axis=2)
        xr = x.reshape(S, B, Lp, k)
        arg = xr.argmax(axis=3)
        return np.take_along_axis(xr, arg[..., None], axis=3)[..., 0], arg

    @staticmethod
    def _pool_back(dp, arg, k, L):
        if k == 1:
            return dp
        S, B, Lp = dp.shape
        dx = np.zeros((S, B, Lp, k), dp.dtype)
        np.put_along_axis(dx, arg[..., None], dp[..., None], axis=3)
        return dx.reshape(S, B, Lp * k)[:, :, :L]

    def forward(self, X, guardar=False):
        p, L, h = self.p, self.L, self.h
        S, B, _ = X.shape
        res = np.ascontiguousarray(X[:, :, ::-1])
        prev = np.repeat(X[:, :, -1:], h, axis=2)
        caches = []
        for j, bl in enumerate(self.blocos):
            pooled, arg = self._pool(res, bl["k"])
            z1 = pooled @ p[f"W1_{j}"] + p[f"b1_{j}"]
            a1 = np.maximum(z1, 0)
            z2 = a1 @ p[f"W2_{j}"] + p[f"b2_{j}"]
            a2 = np.maximum(z2, 0)
            th = a2 @ p[f"W3_{j}"] + p[f"b3_{j}"]
            back, nos = th[..., :L], th[..., L:]
            prev = prev + nos @ bl["M"]
            res = res - back
            if guardar:
                caches.append((pooled, arg, z1, a1, z2, a2))
        if guardar:
            self._cache = caches
        return prev

    def backward(self, dY):
        p, L = self.p, self.L
        g = {}
        dres = np.zeros(dY.shape[:2] + (L,), self.dtype)
        for j in range(len(self.blocos) - 1, -1, -1):
            bl = self.blocos[j]
            pooled, arg, z1, a1, z2, a2 = self._cache[j]
            dnos = dY @ bl["M"].T
            dth = np.concatenate([-dres, dnos], axis=2)
            g[f"W3_{j}"] = (a2.transpose(0, 2, 1) @ dth)
            g[f"b3_{j}"] = dth.sum(axis=1, keepdims=True)
            dz2 = (dth @ p[f"W3_{j}"].transpose(0, 2, 1)) * (z2 > 0)
            g[f"W2_{j}"] = a1.transpose(0, 2, 1) @ dz2
            g[f"b2_{j}"] = dz2.sum(axis=1, keepdims=True)
            dz1 = (dz2 @ p[f"W2_{j}"].transpose(0, 2, 1)) * (z1 > 0)
            g[f"W1_{j}"] = pooled.transpose(0, 2, 1) @ dz1
            g[f"b1_{j}"] = dz1.sum(axis=1, keepdims=True)
            dpooled = dz1 @ p[f"W1_{j}"].transpose(0, 2, 1)
            dres = dres + self._pool_back(dpooled, arg, bl["k"], L)
        self._cache = None
        return g


# ---------------------------------------------------------------------------
# Otimizador e treino com parada antecipada (por semente)
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def passo(self, grads):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for k, gk in grads.items():
            m, v = self.m[k], self.v[k]
            m *= self.b1
            m += (1 - self.b1) * gk
            v *= self.b2
            v += (1 - self.b2) * gk * gk
            self.p[k] -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def recortar_por_semente(grads, limite):
    """Recorte do gradiente pela norma global, separado para cada semente."""
    S = next(iter(grads.values())).shape[0]
    quad = np.zeros(S, np.float64)
    for gk in grads.values():
        quad += (gk.astype(np.float64) ** 2).reshape(S, -1).sum(axis=1)
    fator = np.minimum(1.0, limite / (np.sqrt(quad) + 1e-6)).astype(next(iter(grads.values())).dtype)
    for k in grads:
        grads[k] *= fator.reshape((S,) + (1,) * (grads[k].ndim - 1))
    return np.sqrt(quad)


def prever(modelo, X, lote=1024):
    """X: (N, L) comum a todas as sementes -> (S, N, h)."""
    saidas = []
    for i in range(0, len(X), lote):
        xb = np.broadcast_to(X[i:i + lote], (modelo.S,) + X[i:i + lote].shape)
        saidas.append(modelo.forward(np.ascontiguousarray(xb), guardar=False))
    return np.concatenate(saidas, axis=1)


def treinar(modelo, Xtr, Ytr, Xva, Yva, sementes, lote=256, max_epocas=200,
            paciencia=20, lr=1e-3, recorte=1.0, log=None):
    """
    Perda MAE na escala normalizada (padrão da NeuralForecast). Cada semente
    embaralha as janelas de treino com seu próprio gerador. Ao fim, cada
    semente volta aos pesos da época de menor erro de validação.
    """
    S = modelo.S
    opt = Adam(modelo.p, lr=lr)
    rngs = [np.random.default_rng(10_000 + s) for s in sementes]
    n = len(Xtr)
    lote = min(lote, n)
    n_lotes = n // lote
    melhor_val = np.full(S, np.inf)
    melhor_epoca = np.zeros(S, int)
    sem_melhora = np.zeros(S, int)
    ativo = np.ones(S, bool)
    melhores = {k: v.copy() for k, v in modelo.p.items()}
    historico = []
    for ep in range(1, max_epocas + 1):
        perms = np.stack([r.permutation(n) for r in rngs])
        perda_tr = np.zeros(S)
        for b in range(n_lotes):
            idx = perms[:, b * lote:(b + 1) * lote]
            Xb, Yb = Xtr[idx], Ytr[idx]
            Yh = modelo.forward(Xb, guardar=True)
            erro = Yh - Yb
            perda_tr += np.abs(erro).mean(axis=(1, 2))
            dY = (np.sign(erro) / (lote * erro.shape[2])).astype(modelo.dtype)
            grads = modelo.backward(dY)
            recortar_por_semente(grads, recorte)
            opt.passo(grads)
        perda_tr /= n_lotes
        perda_va = np.abs(prever(modelo, Xva) - Yva[None]).mean(axis=(1, 2))
        melhorou = perda_va < melhor_val - 1e-6
        for s in np.where(melhorou & ativo)[0]:
            for k in modelo.p:
                melhores[k][s] = modelo.p[k][s]
            melhor_val[s] = perda_va[s]
            melhor_epoca[s] = ep
        sem_melhora[melhorou & ativo] = 0
        sem_melhora[~melhorou & ativo] += 1
        ativo &= sem_melhora < paciencia
        historico.append({"epoca": ep, "treino": perda_tr.copy(), "validacao": perda_va.copy()})
        if log and (ep % 10 == 0 or not ativo.any()):
            log(f"   época {ep:3d}  treino {perda_tr.mean():.4f}  validação {perda_va.mean():.4f}  ativas {ativo.sum()}/{S}")
        if not ativo.any():
            break
    for k in modelo.p:
        modelo.p[k][...] = melhores[k]
    return historico, melhor_epoca, melhor_val
