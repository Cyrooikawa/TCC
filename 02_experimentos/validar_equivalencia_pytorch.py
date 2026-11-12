"""
Confere se as redes em NumPy (redes_numpy.py) calculam exatamente o mesmo que o PyTorch.

Copia os pesos de cada rede NumPy para torch.nn.LSTM / torch.nn.GRU (uni e
bidirecional) + torch.nn.Linear e compara, em precisão dupla, a previsão e os
gradientes de todos os parâmetros. Confere também a interpolação linear e o
MaxPool usados no N-HiTS contra as funções do PyTorch.

Precisa do PyTorch instalado (versão para CPU basta):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
Uso:
    python validar_equivalencia_pytorch.py
"""

import sys

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch não está instalado. Instale com:\n"
          "    pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(2)

from redes_numpy import NHITS, RecorrenteDireta, matriz_interpolacao

torch.set_default_dtype(torch.float64)
FALHAS = []


def comparar(nome, a, b, tol=1e-9):
    dif = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    ok = dif < tol
    print(f"[{'OK     ' if ok else 'FALHOU '}] {nome}: maior diferença {dif:.2e}")
    if not ok:
        FALHAS.append(nome)


def testar_recorrente(celula, D, H=8, h=5, B=4, L=21):
    rng = np.random.default_rng(42)
    rede = RecorrenteDireta(celula, H, h, sementes=[7], D=D, dtype=np.float64)
    p = rede.p
    Cls = torch.nn.LSTM if celula == "lstm" else torch.nn.GRU
    rnn = Cls(input_size=1, hidden_size=H, batch_first=True, bidirectional=(D == 2))
    cabeca = torch.nn.Linear(D * H, h)
    with torch.no_grad():
        for d in range(D):
            suf = "_reverse" if d == 1 else ""
            getattr(rnn, f"weight_ih_l0{suf}").copy_(torch.tensor(p["Wx"][0, d].T))
            getattr(rnn, f"weight_hh_l0{suf}").copy_(torch.tensor(p["Wh"][0, d].T))
            if celula == "lstm":
                getattr(rnn, f"bias_ih_l0{suf}").copy_(torch.tensor(p["b"][0, d, 0]))
                getattr(rnn, f"bias_hh_l0{suf}").zero_()
            else:
                getattr(rnn, f"bias_ih_l0{suf}").copy_(torch.tensor(p["bx"][0, d, 0]))
                getattr(rnn, f"bias_hh_l0{suf}").copy_(torch.tensor(p["bh"][0, d, 0]))
        cabeca.weight.copy_(torch.tensor(p["Wo"][0].T))
        cabeca.bias.copy_(torch.tensor(p["bo"][0, 0]))

    X = rng.standard_normal((B, L))
    R = rng.standard_normal((B, h))
    # NumPy
    y_np = rede.forward(X[None], guardar=True)[0]
    g = rede.backward(R[None])
    # PyTorch
    saida = rnn(torch.tensor(X[..., None]))
    hn = saida[1][0] if celula == "lstm" else saida[1]          # (D, B, H)
    final = torch.cat([hn[d] for d in range(D)], dim=1)          # [direta, reversa]
    y_t = cabeca(final)
    (y_t * torch.tensor(R)).sum().backward()

    nome = f"{'Bi-' if D == 2 else ''}{celula.upper()}"
    comparar(f"{nome}: previsão", y_np, y_t.detach().numpy())
    for d in range(D):
        suf = "_reverse" if d == 1 else ""
        comparar(f"{nome}: gradiente Wx{suf}", g["Wx"][0, d].T, getattr(rnn, f"weight_ih_l0{suf}").grad.numpy())
        comparar(f"{nome}: gradiente Wh{suf}", g["Wh"][0, d].T, getattr(rnn, f"weight_hh_l0{suf}").grad.numpy())
        if celula == "lstm":
            comparar(f"{nome}: gradiente do viés{suf}", g["b"][0, d, 0], getattr(rnn, f"bias_ih_l0{suf}").grad.numpy())
        else:
            comparar(f"{nome}: gradiente bx{suf}", g["bx"][0, d, 0], getattr(rnn, f"bias_ih_l0{suf}").grad.numpy())
            comparar(f"{nome}: gradiente bh{suf}", g["bh"][0, d, 0], getattr(rnn, f"bias_hh_l0{suf}").grad.numpy())
    comparar(f"{nome}: gradiente da camada de saída", g["Wo"][0].T, cabeca.weight.grad.numpy())


def testar_componentes_nhits():
    rng = np.random.default_rng(3)
    for n_in, n_out in [(1, 7), (3, 7), (22, 90), (45, 90), (45, 180), (90, 180)]:
        nos = rng.standard_normal((5, n_in))
        nosso = nos @ matriz_interpolacao(n_in, n_out, np.float64)
        torch_ = F.interpolate(torch.tensor(nos)[:, None, :], size=n_out, mode="linear").numpy()[:, 0, :]
        comparar(f"N-HiTS: interpolação linear {n_in} -> {n_out}", nosso, torch_)
    for L, k in [(28, 2), (91, 2), (182, 2)]:
        x = rng.standard_normal((1, 4, L))
        nosso, _ = NHITS._pool(x, k)
        torch_ = torch.nn.MaxPool1d(kernel_size=k, stride=k, ceil_mode=True)(torch.tensor(x[0])[:, None, :]).numpy()[:, 0, :]
        comparar(f"N-HiTS: MaxPool (janela {L}, núcleo {k})", nosso[0], torch_)


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}\n")
    for celula in ("lstm", "gru"):
        for D in (1, 2):
            testar_recorrente(celula, D)
    testar_componentes_nhits()
    print("\n" + ("AS REDES EM NUMPY SÃO EQUIVALENTES ÀS DO PYTORCH" if not FALHAS
                  else f"{len(FALHAS)} COMPARAÇÃO(ÕES) FALHARAM: {FALHAS}"))
    sys.exit(0 if not FALHAS else 1)
