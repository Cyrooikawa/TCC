# Diagnóstico da ordem do SARIMA

## O problema

Na Figura 6 da monografia (Boituva), a previsão do SARIMA vira uma reta depois de
poucas semanas. A ordem usada, SARIMA(3,1,2)(0,0,2)7, tem só médias móveis na parte
sazonal (P = 0), que carregam a lembrança dos últimos erros por cerca de 14 dias. Sem
termo autorregressivo sazonal, nada regenera o ciclo semanal, e a previsão converge
para um nível constante.

## O que foi verificado

**Amplitude semanal da previsão de 180 dias em Boituva** (máximo menos mínimo de cada
semana, em veículos por dia; reexecutado nesta sessão com `diagnostico_ordem_sarima.py amplitude`):

| Semana | 1 | 2 | 3 | 4 | 5 | 6 | 8 em diante |
|---|---|---|---|---|---|---|---|
| Ordem fixada (0,0,2)7 | 4.770 | 1.205 | 365 | 57 | 10 | 2 | 0 |
| Alternativa (1,0,1)7 | 6.438 | 6.510 | 6.541 | 6.553 | 6.556 | 6.557 | cerca de 6.500 |

A amplitude real de uma semana típica no fim do treino é de cerca de 10 mil veículos por dia.

**Alternativas com termo autorregressivo sazonal**, mesmo (p,d,q). O AIC é calculado só
no treino, e o SMAPE é a média das origens de teste, em 7, 90 e 180 dias. Reexecutado
nesta sessão com `diagnostico_ordem_sarima.py alternativas`:

| Praça | Variante | Ordem | AIC | SMAPE 7 / 90 / 180 |
|---|---|---|---|---|
| Alambari | fixada | (0,1,1)(0,0,1)7 | 4.965,7 | 7,01 / 7,71 / 7,66 |
| Alambari | P=1, Q=1 | (0,1,1)(1,0,1)7 | **4.357,4** | **5,27 / 5,85 / 5,90** |
| Alambari | P=1, Q=0 | (0,1,1)(1,0,0)7 | 4.797,0 | 6,36 / 7,52 / 7,56 |
| Boituva | fixada | (3,1,2)(0,0,2)7 | 7.366,4 | 9,62 / 11,13 / 10,83 |
| Boituva | P=1, Q=1 | (3,1,2)(1,0,1)7 | **7.110,6** | **8,59 / 9,76 / 9,48** |
| Boituva | P=1, Q=0 | (3,1,2)(1,0,0)7 | 7.366,8 | 9,47 / 11,07 / 10,80 |
| Itu | fixada | (0,1,2)(0,0,2)7 | 7.391,2 | 6,54 / 7,36 / 7,37 |
| Itu | P=1, Q=1 | (0,1,2)(1,0,1)7 | **7.082,2** | **5,77 / 6,17 / 6,34** |
| Itu | P=1, Q=0 | (0,1,2)(1,0,0)7 | 7.356,0 | 6,30 / 7,23 / 7,27 |

**Reprodução da busca** (sessão anterior, mesma função `escolher_ordem` do projeto): no
ambiente de apoio, o `auto_arima` devolveu a ordem fixada em Alambari, Bloqueio de
Boituva, Boituva, Porto Feliz e Itu, e uma ordem diferente em Sorocaba. A busca levou
de 17 a 136 segundos por praça. A ordem de Itu foi escolhida no ambiente de apoio, e
não no computador do autor.

**Busca com a série em milhares de veículos** (sessão anterior): devolveu ordens
diferentes, (3,1,0)(2,0,0)7 em Alambari, (3,1,2)(0,0,2)7 em Itu e (1,1,2)(0,0,2)7 em
Boituva, sem chegar ao AIC das alternativas acima. Em Alambari, o SMAPE piorou
(7,59 / 13,56 / 15,48). A escala sozinha não explica o problema.

## Conclusão

A busca passo a passo do `auto_arima` não encontrou o modelo de menor AIC do próprio
espaço de busca em Alambari, Boituva e Itu, e é sensível ao ambiente e à escala. Com
P = 1 e Q = 1, o SARIMA mantém o ciclo semanal, tem AIC menor e supera o modelo
ingênuo nas três praças. O Prophet continua melhor nas três.

As conclusões da monografia que dependem do achatamento precisam ser revistas: o
SARIMA perdendo para o ingênuo em Alambari, e o menor SMAPE com pior DTW em Boituva e
Itu. As praças de Bloqueio de Boituva, Porto Feliz e Sorocaba não foram testadas com
alternativas.

## Próximo passo recomendado

Refazer a seleção com busca em grade dentro dos mesmos limites (p e q até 3, P e Q até
2, d = 1, D = 0), escolhendo pelo AIC do treino. Depois, rerodar `arima_baseline.py`,
`comparar_modelos.py` e `gerar_figuras.py`, e atualizar a Tabela 1, as Tabelas 4 a 7,
a Figura 6 e o texto de 4.1, 4.3.1 e 4.3.2.

## Atualização de 08/11/2026: busca completa (`sarima_busca_completa.py`)

As 144 combinações (p e q até 3, P e Q até 2, d = 1 e D = 0, conferidos com os testes
KPSS e OCSB do auto_arima) foram ajustadas em todas as praças, de dois jeitos:

- **Grade A, critério do auto_arima:** série do treino com as lacunas interpoladas, em
  veículos, com intercepto e 50 iterações. É o ajuste que o auto_arima usa para escolher.
- **Grade B, ajuste final:** série em milhares, com as lacunas preservadas no filtro de
  Kalman, sem intercepto e até 200 iterações. É o ajuste que gera as previsões.

O que a busca mostrou:

1. **O auto_arima tem um teste de raízes.** Depois de ajustar cada candidato, ele descarta
   sem aviso todo modelo com raiz inversa acima de 0,99 (`_root_test` do pmdarima). As
   ordens de menor AIC da Tabela 14, com P = 1 e Q = 1, têm raiz AR sazonal ≈ 1,000 e
   MA sazonal ≈ 0,9995 (Φ ≈ 1 e Θ ≈ −1). Por isso, nenhuma busca do auto_arima chegaria a
   elas. Em Alambari, Boituva e Itu, nenhum modelo com P e Q > 0 passa no teste.
2. **A busca passo a passo também perdeu modelos admissíveis de AIC menor**, nas seis praças
   (de 10 pontos em Itu a 1.438 em Bloqueio de Boituva, na grade A).
3. **A escolha e o ajuste final não concordam.** Em Bloqueio de Boituva e Sorocaba, a ordem
   do auto_arima completo tem AIC pior que a ordem atual no ajuste final (1.566 contra 1.560
   e 6.468 contra 5.972). A interpolação das lacunas longas (sem ciclo semanal) pesa na
   escolha.
4. **Neste ambiente (Linux), a ordem oficial de Boituva não passa no teste de raízes**
   (MA ≈ 0,996). Ela está na fronteira de 0,99, o que explica a sensibilidade da busca ao
   computador.

SMAPE médio das seis praças (%), mesmo procedimento de avaliação do `arima_baseline.py`:

| Variante | Critério | 7 dias | 90 dias | 180 dias |
|---|---|---|---|---|
| Atual | auto_arima passo a passo (grade A, com teste de raízes) | 6,77 | 7,66 | 7,60 |
| AUTOARIMA_COMPLETO | auto_arima com stepwise=False e max_order=None | 6,83 | 9,93 | 11,00 |
| MENOR_AIC_SEM_TESTE | menor AIC da grade A, sem teste de raízes | 6,02 | 6,75 | 6,79 |
| MENOR_AIC_AJUSTE_FINAL | menor AIC da grade B, entre os que convergiram | 5,92 | 6,49 | 6,46 |

A busca completa do auto_arima piora o SARIMA por causa de Sorocaba: a ordem
(3,1,0)(2,0,0)7 perde a fase do ciclo semanal (20,8% em 90 dias e 26,7% em 180).

A variante MENOR_AIC_AJUSTE_FINAL escolhe (3,1,2)(1,0,2)7 em cinco praças e
(0,1,2)(2,0,1)7 em Bloqueio de Boituva. Ela:
- passa nas checagens de segurança do `arima_baseline.py`;
- mantém o ciclo semanal em todo o horizonte;
- tem Ljung-Box (14 dias) com p de 0,30 a 0,61 em Alambari, Porto Feliz, Sorocaba e Itu.

Todas as ordens dessa variante têm raiz AR sazonal quase unitária. A escolha entre as
variantes é do orientador e deve seguir o critério de método, não o erro no teste.
