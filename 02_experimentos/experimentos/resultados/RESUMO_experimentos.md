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
| GRU | Calendário (dia da semana e feriados) como entrada | 6,44 → 5,86 (-0,58) | — | — | 58% / — / — | 3,9→3,6 / 10,4→9,3 / 5,4→5,1 | < 0,001 | melhora; critério sem teste confirma |
| NHITS | Calendário (dia da semana e feriados) como entrada | 6,13 → 5,57 (-0,56) | — | — | 55% / — / — | 3,9→4,0 / 9,8→8,2 / 4,9→4,7 | < 0,001 | melhora; critério sem teste NÃO confirma |
