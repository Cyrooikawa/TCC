"""
Figura 1 da monografia: pipeline completo do trabalho, dos dados brutos à comparação.

Como executar (segundos):
    python figura_pipeline.py
Saída: figuras/pipeline.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RAIZ = Path(__file__).resolve().parent
AZUL, LARANJA, VERDE, CINZA = "#1d5fae", "#eb6834", "#1b8f63", "#52514e"
ETAPAS = [  # (título, linhas, cor da borda)
    ("1. Dados brutos ARTESP", ["145 arquivos CSV, volume diário", "1º/01/2020 a 31/07/2026"], AZUL),
    ("2. Organização por lote", ["24 arquivos de 5 lotes da RMS", "núcleo (L13, L20) e unidirecional (L12, L33, L34)"], AZUL),
    ("3. Extração", ["concatenação por lote", "verificação de esquema e do ano do arquivo"], AZUL),
    ("4. Tratamento", ["nomes padronizados, filtro dos 27 municípios", "volume até 10 eixos, sentidos por praça"], AZUL),
    ("5. Séries diárias", ["calendário completo: OK, ausente, invalidado", "sem imputação; modelagem a partir de 2021"], AZUL),
    ("6. Recorte das praças", ["núcleo: 4 praças bidirecionais", "unidirecional: Itu e Sorocaba"], AZUL),
    ("7. Divisão cronológica", ["treino 1.426 dias (validação das redes: 285)", "teste 612 dias; origens 87, 75 e 62"], AZUL),
    ("8. Pré-processamento", ["estatísticos: série original (SARIMA em milhares)", "redes: log(1+x) e Min-Max do treino"], AZUL),
    ("9a. Modelos estatísticos", ["ingênuo sazonal, SARIMA, Prophet", "um modelo por praça"], LARANJA),
    ("9b. Redes neurais", ["LSTM, GRU, Bi-LSTM, Bi-GRU, N-HiTS", "5 inicializações; saída direta multissaída"], VERDE),
    ("10. Avaliação", ["7, 90 e 180 dias; SMAPE, MAPE, MAE, RMSE, DTW", "mesmas praças, origens e horizontes"], AZUL),
    ("11. Comparação", ["Friedman e Wilcoxon-Holm", "análises de sensibilidade"], AZUL),
]
fig, ax = plt.subplots(figsize=(8.6, 10.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 124)
ax.axis("off")
W, H = 44, 14
POS = {0: (3, 106), 1: (53, 106), 2: (53, 87), 3: (3, 87), 4: (3, 68), 5: (53, 68),
       6: (53, 49), 7: (3, 49), 8: (3, 27), 9: (53, 27), 10: (3, 5), 11: (53, 5)}


def caixa(i):
    x, y = POS[i]
    t, linhas, cor = ETAPAS[i]
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=1.6", lw=1.6, ec=cor,
                                fc="#eef3fa" if cor == AZUL else ("#fdf1ea" if cor == LARANJA else "#eaf6f0")))
    ax.text(x + W / 2, y + H - 3.6, t, ha="center", va="center", fontsize=10.2, fontweight="bold", color="#111")
    for k, l in enumerate(linhas):
        ax.text(x + W / 2, y + H - 7.6 - 3.4 * k, l, ha="center", va="center", fontsize=8.1, color="#222")


def seta(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=CINZA))


for i in range(len(ETAPAS)):
    caixa(i)
c = lambda i: (POS[i][0] + W / 2, POS[i][1] + H / 2)  # noqa: E731
seta((3 + W, 113), (53, 113))            # 1 -> 2
seta((75, 106), (75, 101))               # 2 -> 3
seta((53, 94), (3 + W, 94))              # 3 -> 4
seta((25, 87), (25, 82))                 # 4 -> 5
seta((3 + W, 75), (53, 75))              # 5 -> 6
seta((75, 68), (75, 63))                 # 6 -> 7
seta((53, 56), (3 + W, 56))              # 7 -> 8
seta((25, 49), (25, 41))                 # 8 -> 9a
seta((40, 49), (60, 41))                 # 8 -> 9b
seta((25, 27), (25, 19))                 # 9a -> 10
seta((63, 27), (40, 19))                 # 9b -> 10
seta((3 + W, 12), (53, 12))              # 10 -> 11
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(RAIZ / "figuras" / "pipeline.png", dpi=220)
print("figura gravada: figuras/pipeline.png")
