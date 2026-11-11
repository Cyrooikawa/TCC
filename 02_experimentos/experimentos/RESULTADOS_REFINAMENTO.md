# Resultados com a configuração ajustada

Versão para a monografia: `monografia/Resultados_configuracao_ajustada.docx`, com texto,
tabelas e figuras prontos para colar. Este arquivo é o resumo.

## Regra usada

- Três ajustes de configuração, um por problema já diagnosticado na primeira execução.
- Cada um muda um único elemento e não acrescenta informação que o modelo não recebia.
- A escolha usa só o treino: AIC no SARIMA, SMAPE em origens dentro dos 20% finais do treino
  no Prophet, perda de validação na LSTM. O teste só mede o efeito.
- Entre as variações de configuração testadas para cada modelo, o ajuste escolhido é o de
  melhor critério no treino.

## Ajustes

| Modelo | Problema | Ajuste | Critério no treino | SMAPE original | SMAPE ajustado |
|---|---|---|---|---|---|
| SARIMA | previsão perde o ciclo semanal | ordem de menor AIC (grade de 144 ordens) | AIC 29570 → 27868 | 6,77 / 7,66 / 7,60 | 5,92 / 6,49 / 6,46 |
| Prophet | erro concentrado nos feriados longos | janela própria para Carnaval, Sexta-feira Santa, Natal e Ano Novo | validação 4,85 / 4,58 → 4,66 / 4,45 | 4,64 / 4,71 / 4,68 | 4,43 / 4,49 / 4,47 |
| LSTM 180 d | 3 sementes presas num platô | paciência 40, até 400 épocas | perda 0,0631 → 0,0444 | 10,15 | 7,67 |

## Resultado com a configuração ajustada (SMAPE, %)

| Modelo | 7 dias | 90 dias | 180 dias |
|---|---|---|---|
| Prophet | 4,43 | 4,49 | 4,47 |
| SARIMA | 5,92 | 6,49 | 6,46 |
| N-HiTS | 6,13 | 6,58 | 6,84 |
| LSTM | 6,37 | 7,38 | 7,67 |
| Bi-GRU | 6,41 | 6,55 | 7,14 |
| GRU | 6,44 | 6,54 | 7,38 |
| Bi-LSTM | 6,46 | 6,89 | 7,60 |
| Ingênuo sazonal | 7,40 | 8,72 | 8,91 |

- O Prophet tem o menor erro nas cinco métricas, nas seis praças e nos três horizontes.
- O SARIMA fica em segundo no SMAPE nos três horizontes e supera o ingênuo em todas as praças.
- Em 7 dias (média das praças por origem, Wilcoxon-Holm), o SARIMA difere do ingênuo e não
  difere de nenhuma rede.

O que não entrou e por quê está em `SUGESTOES_ORIENTADOR.md`.
