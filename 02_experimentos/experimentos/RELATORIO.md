# Experimentos de refinamento — o resultado melhora?

Testes das ideias de "Como refinar" do *Guia dos modelos*, feitos como experimento paralelo
(seção 6 do guia). **Nenhum resultado oficial foi alterado**: tudo está em `02_experimentos/`,
cópia de `02_codigo_e_resultados/`, e as saídas ficam em `experimentos/resultados/`.
Adotar qualquer mudança na monografia é decisão do orientador.

## Como foi feito

- Mesma divisão treino/teste, mesmas origens semanais (87 / 75 / 62 por praça), mesmas seis
  praças, mesma métrica (SMAPE, média das praças) e, nas redes, as mesmas cinco sementes.
- Uma mudança por vez. As três linhas marcadas "COMBINADO" juntam mudanças que já tinham
  sido testadas sozinhas.
- Cada linha traz um critério que **não usa o teste**: AIC do treino (SARIMA), SMAPE em
  origens dentro dos últimos 20% do treino (ingênuo e Prophet) ou perda de validação (redes).
  O número do teste aparece só como registro de antes e depois.
- A tabela completa, com % de origens melhores, separação por tipo de semana e teste de
  Wilcoxon, está em `resultados/RESUMO_experimentos.md` (e `.csv`).

SMAPE (%) em 7 / 90 / 180 dias. "Antes" é a configuração oficial reexecutada neste ambiente.

## Resposta curta

Sim, em vários modelos. O que melhora com apoio do critério sem teste:

| Modelo | Mudança | Antes | Depois | Critério sem teste |
|---|---|---|---|---|
| SARIMA | Ordem de menor AIC | 6,76 / 7,65 / 7,59 | 5,92 / 6,49 / 6,46 | AIC 29570 → 27868 |
| SARIMA | Menor AIC + feriados (combinado) | 6,76 / 7,65 / 7,59 | 5,52 / 5,96 / 5,97 | AIC 29570 → 26052 |
| Prophet | Janela própria para feriados longos | 4,64 / 4,71 / 4,68 | 4,43 / 4,49 / 4,47 | validação 4,85 / 4,58 → 4,66 / 4,45 |
| LSTM (180 d) | Paciência 40, até 400 épocas | 10,15 | 7,67 | perda 0,0631 → 0,0444 |
| LSTM (180 d) | Viés da porta de esquecimento = 1 | 10,15 | 8,72 | perda 0,0631 → 0,0531 |
| GRU, LSTM, Bi-GRU, Bi-LSTM (7 d) | Calendário como entrada | 6,37 a 6,46 | 5,86 a 6,02 | perda cai nas quatro |
| Ingênuo | Média das últimas 4 semanas | 7,40 / 8,72 / 8,91 | 6,98 / 7,52 / 7,66 | validação 6,44 / 6,77 → 5,48 / 5,81 |

A ordem dos modelos não muda no topo: o Prophet continua o melhor nos três horizontes.
O SARIMA refinado passa as redes oficiais e fica em segundo.

## Por modelo

### Ingênuo sazonal
- **Média de 4 semanas**: melhora nos três horizontes, confirmada na validação. Piora nas
  semanas comuns (3,6 → 4,9) e ganha nas semanas depois de feriado (9,6 → 5,4); em 7 dias
  a diferença não é significativa (p = 0,28).
- **Mesma semana do ano anterior**: 7,09 / 6,97 / 6,81. Ganho grande em 90 e 180 dias, mas
  na validação piora em 7 dias (6,44 → 6,75).
- Atenção: o ingênuo é a referência. Trocar a referência muda a leitura de todos os outros
  modelos, então vale mais como análise de sensibilidade.

### SARIMA
- **Menor AIC**: melhora clara (71% a 81% das origens, p < 0,001). Cinco das seis praças vão
  para (3,1,2)(1,0,2)7.
- **Feriados como variável externa**: melhora em 7 dias (6,28) e nas semanas com feriado
  (10,8 → 9,6); em 180 dias piora um pouco (7,70).
- **Fourier anual** (1, 2 ou 3 harmônicos): ganho pequeno (até −0,38 ponto); o AIC quase
  não muda.
- **Deriva**: piora (6,92 / 8,03 / 8,05) e o AIC também piora. Descartar.
- **Combinados**: menor AIC + feriados dá 5,52 / 5,96 / 5,97. Somar Fourier leva a
  5,51 / 5,74 / 5,74, mas o AIC não melhora (26052 nos dois), então o critério sem teste
  não justifica o Fourier.
- Ressalvas: o otimizador não convergiu em Itu (menor AIC + feriados) e em Sorocaba
  (Fourier 2). O SARIMA é sensível à versão das bibliotecas: com numpy/scipy mais novos o
  oficial dá 6,79 / 7,86 / 7,85; com as versões fixadas, 6,76 / 7,65 / 7,59 (oficial:
  6,77 / 7,66 / 7,60).

### Prophet
- **Janelas longas** (Natal, Ano Novo, Carnaval, Sexta-feira Santa): melhora de 0,21 ponto
  nos três horizontes, confirmada na validação, 59% a 80% das origens.
- **Multiplicativo**: 4,57 / 4,66 / 4,66 no teste, mas a validação piora. Não adotar por
  esse critério.
- **Combinado janelas longas + multiplicativo**: 4,38 / 4,47 / 4,46, o menor erro de todos
  os experimentos. A validação melhora (4,71 / 4,49), porém menos do que só com janelas
  longas (4,66 / 4,45): pelo critério sem teste, a escolha é janelas longas sozinha.
- **Férias escolares**, **holidays_prior_scale** (1 e 0,1): sem efeito (≤ 0,02 ponto).
- **changepoint_prior_scale** 0,005: piora; 0,5: levemente pior. O padrão está bem.

### Redes recorrentes
- **LSTM em 180 dias**: o erro alto oficial (10,15) vem de treino interrompido cedo.
  Paciência 40 leva a 7,67, melhor em 91% das origens, com 25 min de treino contra 22.
  O viés de esquecimento em 1 também ajuda (8,72), com treino mais curto (17 min).
  Os dois não foram testados juntos nem nos outros horizontes.
- **Calendário como entrada, 7 dias**: melhora as quatro recorrentes em cerca de 0,5 ponto,
  sobretudo nas semanas com feriado (≈ 10,3 → 9,4), com treino mais rápido.
- **Calendário em 90 e 180 dias (GRU)**: piora no teste (6,54 → 6,88; 7,38 → 8,04), embora
  a validação de 90 dias melhore. Não usar em horizontes longos.
- **Janela de 91 dias no horizonte de 7**: no teste, GRU 5,99, Bi-GRU 6,06, Bi-LSTM 6,16
  melhoram e a LSTM piora (6,58). A validação só confirma no GRU, e as janelas de validação
  não são as mesmas, então a evidência é fraca. O treino fica 2 a 3 vezes mais lento.
- **Unidades da GRU**: 128 dá ganhos pequenos (6,31 / 6,56 / 7,15) com treino 40% mais
  longo; 32 fica igual em 7 e 90 dias e piora em 180. As 64 oficiais são razoáveis.

### N-HiTS
- **Calendário**: 7 dias cai para 5,57 no teste, mas a validação piora (0,0351 → 0,0380);
  em 90 e 180 dias piora muito (8,40 e 8,88). Não há base para adotar.
- **Janela de 91 dias**: 6,13 → 6,00, não significativo (p = 0,17) e sem apoio da validação.
- **256 unidades**: praticamente igual em 7 e 90 dias, pior em 180; treino 3 vezes mais
  rápido.

## Ranking se as mudanças com apoio do critério fossem adotadas

| Modelo | 7 d | 90 d | 180 d | Configuração |
|---|---|---|---|---|
| Prophet | 4,43 | 4,49 | 4,47 | janelas longas |
| SARIMA | 5,52 | 5,96 | 5,97 | menor AIC + feriados |
| GRU | 5,86 | 6,54 | 7,38 | calendário só em 7 d |
| Bi-GRU | 5,89 | 6,55 | 7,14 | calendário só em 7 d |
| LSTM | 5,90 | 7,38 | 7,67 | calendário em 7 d; paciência 40 em 180 d |
| Bi-LSTM | 6,02 | 6,89 | 7,60 | calendário só em 7 d |
| N-HiTS | 6,13 | 6,58 | 6,84 | oficial |
| Ingênuo | 7,40 | 8,72 | 8,91 | oficial (referência) |

## O que não foi testado

- GRU com duas camadas e dropout no N-HiTS (exigem mudar a implementação das redes).
- Inclusão das praças Morro I e II (depende dos dados brutos e do orientador).
- Busca ampla de hiperparâmetros.
- Combinações nas redes (por exemplo, paciência 40 + viés de esquecimento).

## Limites

- Uma série de teste só, 612 dias: diferenças abaixo de 0,1 ponto são ruído.
- Foram testadas cerca de 35 variantes; algumas melhoras no teste aparecem por acaso.
  Por isso a coluna do critério sem teste é a que deve decidir.
- O p de Wilcoxon trata origens e praças como independentes, o que é otimista.

## Como reproduzir

Na pasta `02_experimentos/experimentos/`:

```
python exp_estatisticos.py ingenuo
python exp_estatisticos.py sarima
python exp_estatisticos.py prophet validacao
python exp_estatisticos.py prophet teste
python exp_redes.py --fila 3
python consolidar.py
```

Os estatísticos foram rodados com numpy 2.4.6, scipy 1.17.1, statsmodels 0.15.0,
pmdarima 2.1.1 e prophet 1.4.0. A fila das redes leva algumas horas.
