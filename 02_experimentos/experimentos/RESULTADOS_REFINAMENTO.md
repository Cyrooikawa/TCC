# Resultados dos refinamentos

Cada mudança foi testada isoladamente, com o mesmo treino, as mesmas origens semanais e as
mesmas praças do resultado oficial, e foi confirmada por um critério que não usa o teste
(AIC do treino, validação dentro do treino ou perda de validação). Os resultados oficiais
da monografia não foram alterados.

SMAPE médio das seis praças (%), antes e depois.

## Mudanças aplicadas

| Modelo | Mudança | Horizonte | Antes | Depois | Origens melhores | Critério sem teste |
|---|---|---|---|---|---|---|
| SARIMA | Ordem de menor AIC | 7 d | 6,76 | 5,92 | 71% | AIC do treino 29570 → 27868 |
| SARIMA | | 90 d | 7,65 | 6,49 | 79% | |
| SARIMA | | 180 d | 7,59 | 6,46 | 81% | |
| Prophet | Janela própria para feriados longos | 7 d | 4,64 | 4,43 | 59% | SMAPE de validação 4,85 → 4,66 (7 d) |
| Prophet | | 90 d | 4,71 | 4,49 | 71% | 4,58 → 4,45 (90 d) |
| Prophet | | 180 d | 4,68 | 4,47 | 80% | |
| LSTM | Paciência 40 e até 400 épocas | 180 d | 10,15 | 7,67 | 91% | perda de validação 0,0631 → 0,0444 |
| LSTM | Viés da porta de esquecimento = 1 | 180 d | 10,15 | 8,72 | 86% | perda de validação 0,0631 → 0,0531 |
| LSTM | Calendário como entrada | 7 d | 6,37 | 5,90 | 55% | perda de validação 0,0364 → 0,0331 |
| GRU | Calendário como entrada | 7 d | 6,44 | 5,86 | 58% | perda de validação 0,0356 → 0,0331 |
| Bi-LSTM | Calendário como entrada | 7 d | 6,46 | 6,02 | 56% | perda de validação 0,0363 → 0,0335 |
| Bi-GRU | Calendário como entrada | 7 d | 6,41 | 5,89 | 60% | perda de validação 0,0360 → 0,0334 |

O calendário é o dia da semana mais feriado, véspera e dia seguinte, o mesmo calendário do
Prophet. As redes com calendário também treinam mais rápido (por exemplo, GRU de 7 para 5 min
e Bi-LSTM de 17 para 6 min).

## Desempenho dos modelos com essas mudanças

| Modelo | 7 dias | 90 dias | 180 dias |
|---|---|---|---|
| Prophet | 4,43 | 4,49 | 4,47 |
| SARIMA | 5,92 | 6,49 | 6,46 |
| GRU | 5,86 | 6,54 | 7,38 |
| Bi-GRU | 5,89 | 6,55 | 7,14 |
| LSTM | 5,90 | 7,38 | 7,67 |
| Bi-LSTM | 6,02 | 6,89 | 7,60 |
| N-HiTS | 6,13 | 6,58 | 6,84 |
| Ingênuo sazonal | 7,40 | 8,72 | 8,91 |

Onde nenhuma mudança foi aplicada, o valor é o oficial.

Como no oficial, o Prophet é o melhor modelo nos três horizontes. O SARIMA refinado fica em
segundo em 90 e 180 dias, à frente de todas as redes, e em terceiro em 7 dias, atrás só do
GRU com calendário (5,86 contra 5,92).
