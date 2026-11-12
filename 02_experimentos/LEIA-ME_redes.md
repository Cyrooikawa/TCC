# Redes neurais e resultados completos (8 modelos)

Copie o conteúdo de `02_codigo_e_resultados/` para a pasta de mesmo nome do projeto.

## O que tem aqui

| Arquivo | Para que serve |
|---|---|
| `redes_numpy.py` | LSTM, GRU, Bi-LSTM, Bi-GRU e N-HiTS em NumPy (gradientes conferidos por diferenças finitas) |
| `redes_neurais.py` | Treina e avalia as 5 redes nos 3 horizontes, com o mesmo desenho dos modelos estatísticos |
| `resultados_completos.py` | Gera as tabelas e figuras das seções 4.3, 4.4 e 4.5 com os 8 modelos |
| `resultados/*_metricas_por_origem.csv` | Métricas das redes por origem (mesmo formato dos demais modelos) |
| `resultados/redes/` | Por semente, previsões diárias (.npz) e resumo do treino (épocas, tempo, histórico) |
| `resultados/completos_*.csv` | Tabelas consolidadas (4, 7, 8, 9, 10, 11, sementes, faixas) |
| `figuras/` | Figuras 5 e 9 atualizadas, 10, 11 e 12 (curvas de aprendizado) |
| `apoio/` | Teste de mesa das métricas (imagens) |
| `validar_procedimentos.py` | 71 verificações automáticas: série, divisão, origens, métricas, ingênuo, pareamento, vazamento e resultados esperados |
| `validar_equivalencia_pytorch.py` | Compara as redes em NumPy com torch.nn.LSTM/GRU (previsões e gradientes) |

## Como validar

```
python validar_procedimentos.py                       # deve terminar com TODAS AS VERIFICAÇÕES PASSARAM
python validar_procedimentos.py --reproduzir NHITS 7  # refaz uma rede (cerca de 5 min) e compara
python validar_equivalencia_pytorch.py                # precisa do PyTorch (versão CPU)
```

## Como reproduzir no seu computador

```
pip install torch --index-url https://download.pytorch.org/whl/cpu   # só para o teste de equivalência
pip install numpy pandas scipy matplotlib python-dateutil holidays
python redes_neurais.py --processos 2      # cerca de 2 horas; grava resultados/redes e *_metricas_por_origem.csv
python resultados_completos.py             # tabelas e figuras com os 8 modelos
```

Para rodar só uma combinação: `python redes_neurais.py GRU 7`.

## Configuração usada (itens que o texto deixou em aberto)

- Validação: 20% finais do treino (16/02/2024 a 26/11/2024), só para a parada antecipada.
- Janela de entrada: 28, 91 e 182 dias para os horizontes de 7, 90 e 180 dias.
- Saída direta dos h dias; parâmetros estimados uma vez e mantidos em todas as origens.
- Recorrentes: 1 camada de 64 unidades (por direção nas bidirecionais); N-HiTS: 3 blocos, MLP 2 × 512.
- Adam, taxa 0,001, perda MAE, lote 32 (recorrentes) e 256 (N-HiTS), até 200 épocas, paciência 20, recorte do gradiente em 1.
- Cinco sementes; a métrica de cada origem é a média das cinco.
- Um modelo por rede e horizonte com as janelas das seis praças (cada praça normalizada com log1p + Min-Max do próprio treino).
