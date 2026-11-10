| Modelo | O que mudou | SMAPE 7 d | SMAPE 90 d | SMAPE 180 d | Origens melhores (7/90/180) | Semana comum / com feriado / seguinte (7 d) | p (7 d) | Resultado |
|---|---|---|---|---|---|---|---|---|
| INGENUO | Mesma semana do ano anterior, corrigida pelo crescimento | 7,40 → 7,09 (-0,32) | 8,72 → 6,97 (-1,75) | 8,91 → 6,81 (-2,10) | 53% / 51% / 62% | 3,6→5,0 / 11,4→11,4 / 9,6→4,3 | 0,481 | melhora; critério sem teste confirma |
| INGENUO | Média das últimas 4 semanas | 7,40 → 6,98 (-0,42) | 8,72 → 7,52 (-1,20) | 8,91 → 7,66 (-1,25) | 50% / 61% / 65% | 3,6→4,9 / 11,4→10,6 / 9,6→5,4 | 0,276 | melhora; critério sem teste confirma |
| SARIMA | COMBINADO: menor AIC + feriados | 6,76 → 5,52 (-1,24) | 7,65 → 5,96 (-1,69) | 7,59 → 5,97 (-1,62) | 77% / 91% / 92% | 4,3→3,3 / 10,8→9,1 / 5,5→4,4 | < 0,001 | melhora; critério sem teste confirma |
| SARIMA | COMBINADO: menor AIC + feriados + Fourier (2) | 6,76 → 5,51 (-1,26) | 7,65 → 5,74 (-1,91) | 7,59 → 5,74 (-1,86) | 74% / 90% / 94% | 4,3→3,5 / 10,8→8,9 / 5,5→4,4 | < 0,001 | melhora; critério sem teste confirma |
| SARIMA | Termo de deriva (crescimento) | 6,76 → 6,92 (+0,15) | 7,65 → 8,03 (+0,37) | 7,59 → 8,05 (+0,45) | 41% / 30% / 25% | 4,3→4,6 / 10,8→10,7 / 5,5→5,6 | < 0,001 | piora; critério sem teste confirma |
| SARIMA | Feriado, véspera e dia seguinte como variáveis externas | 6,76 → 6,28 (-0,48) | 7,65 → 7,44 (-0,21) | 7,59 → 7,70 (+0,11) | 65% / 59% / 65% | 4,3→4,2 / 10,8→9,6 / 5,5→5,5 | < 0,001 | misto; critério sem teste confirma |
| SARIMA | Fourier anual, 1 harmônico | 6,76 → 6,70 (-0,07) | 7,65 → 7,37 (-0,28) | 7,59 → 7,25 (-0,34) | 51% / 63% / 70% | 4,3→4,3 / 10,8→10,7 / 5,5→5,4 | 0,139 | melhora; critério sem teste confirma |
| SARIMA | Fourier anual, 2 harmônicos | 6,76 → 6,70 (-0,07) | 7,65 → 7,45 (-0,21) | 7,59 → 7,36 (-0,23) | 52% / 60% / 68% | 4,3→4,3 / 10,8→10,6 / 5,5→5,7 | 0,221 | melhora; critério sem teste confirma |
| SARIMA | Fourier anual, 3 harmônicos | 6,76 → 6,65 (-0,11) | 7,65 → 7,27 (-0,38) | 7,59 → 7,26 (-0,34) | 53% / 66% / 69% | 4,3→4,2 / 10,8→10,5 / 5,5→5,7 | 0,031 | melhora; critério sem teste confirma |
| SARIMA | Ordem de menor AIC do ajuste final | 6,76 → 5,92 (-0,84) | 7,65 → 6,49 (-1,16) | 7,59 → 6,46 (-1,13) | 71% / 79% / 81% | 4,3→3,5 / 10,8→9,9 / 5,5→4,6 | < 0,001 | melhora; critério sem teste confirma |
| PROPHET | COMBINADO: janelas longas + multiplicativo | 4,64 → 4,38 (-0,26) | 4,71 → 4,47 (-0,24) | 4,68 → 4,46 (-0,22) | 63% / 72% / 75% | 3,3→3,0 / 7,1→6,9 / 3,4→3,2 | < 0,001 | melhora; critério sem teste confirma |
| PROPHET | changepoint_prior_scale = 0,005 (padrão 0,05) | 4,64 → 4,76 (+0,13) | 4,71 → 4,82 (+0,11) | 4,68 → 4,80 (+0,12) | 43% / 44% / 35% | 3,3→3,5 / 7,1→7,2 / 3,4→3,4 | < 0,001 | piora; critério sem teste confirma |
| PROPHET | changepoint_prior_scale = 0,5 (padrão 0,05) | 4,64 → 4,68 (+0,04) | 4,71 → 4,74 (+0,03) | 4,68 → 4,73 (+0,05) | 49% / 46% / 46% | 3,3→3,3 / 7,1→7,2 / 3,4→3,5 | 0,265 | misto; critério sem teste confirma |
| PROPHET | Regressor de férias escolares | 4,64 → 4,62 (-0,02) | 4,71 → 4,69 (-0,02) | 4,68 → 4,66 (-0,02) | 56% / 65% / 70% | 3,3→3,2 / 7,1→7,2 / 3,4→3,3 | < 0,001 | sem efeito; critério sem teste confirma |
| PROPHET | holidays_prior_scale = 0,1 (padrão 10) | 4,64 → 4,62 (-0,02) | 4,71 → 4,69 (-0,01) | 4,68 → 4,66 (-0,02) | 45% / 53% / 67% | 3,3→3,3 / 7,1→7,1 / 3,4→3,4 | 0,533 | sem efeito; critério sem teste confirma |
| PROPHET | holidays_prior_scale = 1 (padrão 10) | 4,64 → 4,64 (-0,00) | 4,71 → 4,71 (+0,00) | 4,68 → 4,68 (+0,00) | 49% / 53% / 49% | 3,3→3,3 / 7,1→7,1 / 3,4→3,4 | 0,547 | sem efeito; critério sem teste NÃO confirma |
| PROPHET | Janela própria para os feriados longos | 4,64 → 4,43 (-0,21) | 4,71 → 4,49 (-0,21) | 4,68 → 4,47 (-0,21) | 59% / 71% / 80% | 3,3→3,1 / 7,1→6,9 / 3,4→3,2 | < 0,001 | melhora; critério sem teste confirma |
| PROPHET | Sazonalidade multiplicativa | 4,64 → 4,57 (-0,06) | 4,71 → 4,66 (-0,05) | 4,68 → 4,66 (-0,02) | 55% / 56% / 48% | 3,3→3,2 / 7,1→7,1 / 3,4→3,3 | 0,003 | misto; critério sem teste NÃO confirma |
| BIGRU | Calendário (dia da semana e feriados) como entrada | 6,41 → 5,89 (-0,52) | — | — | 60% / — / — | 4,0→3,7 / 10,2→9,4 / 5,4→4,8 | < 0,001 | melhora; critério sem teste confirma |
| BILSTM | Calendário (dia da semana e feriados) como entrada | 6,46 → 6,02 (-0,44) | — | — | 56% / — / — | 4,1→4,0 / 10,3→9,4 / 5,5→5,0 | < 0,001 | melhora; critério sem teste confirma |
| GRU | Calendário (dia da semana e feriados) como entrada | 6,44 → 5,86 (-0,58) | 6,54 → 6,88 (+0,34) | 7,38 → 8,04 (+0,66) | 58% / 40% / 36% | 3,9→3,6 / 10,4→9,3 / 5,4→5,1 | < 0,001 | misto; critério sem teste NÃO confirma |
| LSTM | Calendário (dia da semana e feriados) como entrada | 6,37 → 5,90 (-0,47) | — | — | 55% / — / — | 3,9→3,7 / 10,4→9,4 / 5,2→5,0 | < 0,001 | melhora; critério sem teste confirma |
| NHITS | Calendário (dia da semana e feriados) como entrada | 6,13 → 5,57 (-0,56) | 6,58 → 8,40 (+1,83) | 6,84 → 8,88 (+2,04) | 55% / 29% / 22% | 3,9→4,0 / 9,8→8,2 / 4,9→4,7 | < 0,001 | misto; critério sem teste confirma |
| BIGRU | Janela de entrada de 91 dias no horizonte de 7 | 6,41 → 6,06 (-0,35) | — | — | 57% / — / — | 4,0→3,6 / 10,2→10,3 / 5,4→4,3 | 0,001 | melhora; critério sem teste NÃO confirma |
| BILSTM | Janela de entrada de 91 dias no horizonte de 7 | 6,46 → 6,16 (-0,30) | — | — | 54% / — / — | 4,1→3,8 / 10,3→10,3 / 5,5→4,5 | 0,006 | melhora; critério sem teste NÃO confirma |
| GRU | Janela de entrada de 91 dias no horizonte de 7 | 6,44 → 5,99 (-0,44) | — | — | 60% / — / — | 3,9→3,5 / 10,4→10,3 / 5,4→4,3 | < 0,001 | melhora; critério sem teste confirma |
| LSTM | Janela de entrada de 91 dias no horizonte de 7 | 6,37 → 6,58 (+0,21) | — | — | 35% / — / — | 3,9→4,1 / 10,4→10,8 / 5,2→5,2 | < 0,001 | piora; critério sem teste confirma |
| NHITS | Janela de entrada de 91 dias no horizonte de 7 | 6,13 → 6,00 (-0,12) | — | — | 55% / — / — | 3,9→3,9 / 9,8→10,0 / 4,9→3,9 | 0,166 | melhora; critério sem teste NÃO confirma |
| NHITS | 256 unidades por camada (oficial: 512) | 6,13 → 6,11 (-0,02) | 6,58 → 6,56 (-0,02) | 6,84 → 7,04 (+0,20) | 52% / 55% / 42% | 3,9→3,9 / 9,8→9,9 / 4,9→4,7 | 0,419 | misto; critério sem teste confirma |
| LSTM | Paciência 40 e até 400 épocas | — | — | 10,15 → 7,67 (-2,47) | — / — / 91% | — | — | melhora; critério sem teste confirma |
| GRU | 128 unidades (oficial: 64) | 6,44 → 6,31 (-0,13) | 6,54 → 6,56 (+0,02) | 7,38 → 7,15 (-0,23) | 60% / 46% / 90% | 3,9→3,8 / 10,4→10,2 / 5,4→5,3 | < 0,001 | misto; critério sem teste confirma |
| GRU | 32 unidades (oficial: 64) | 6,44 → 6,41 (-0,03) | 6,54 → 6,53 (-0,01) | 7,38 → 7,47 (+0,09) | 53% / 50% / 36% | 3,9→3,9 / 10,4→10,4 / 5,4→5,4 | 0,154 | misto; critério sem teste confirma |
| LSTM | Viés da porta de esquecimento = 1 | — | — | 10,15 → 8,72 (-1,43) | — / — / 86% | — | — | melhora; critério sem teste confirma |
