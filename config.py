"""
Configuração central do experimento.

Todas as decisões metodológicas ficam aqui, num lugar só. Os demais scripts
(preparar_series.py, arima_baseline.py e os próximos modelos) importam estes
valores, em vez de cada um ter sua própria cópia -- assim, mudar uma decisão
aqui muda o experimento inteiro de forma coerente.

Este arquivo não é executado sozinho.
"""

# ---------------------------------------------------------------------------
# Grupos de praças
# ---------------------------------------------------------------------------

# Núcleo da comparação principal: praças bidirecionais (medem fluxo total, nos
# dois sentidos), série contínua desde 2020, sem troca de instrumento de
# medição, e escala de volume compatível entre si (milhares de veículos/dia).
PRACAS_NUCLEO = ["ALAMBARI", "BOITUVA", "BLOQUEIO DE BOITUVA", "PORTO FELIZ"]

# Grupo unidirecional: praças físicas de Sorocaba e Itu. Entram no trabalho,
# mas NÃO na mesma tabela comparativa do núcleo, porque o pedágio só cobra um
# sentido (Sorocaba: Sul; Itu: Leste) -- medem metade do fluxo, enquanto o
# núcleo mede o fluxo total nos dois sentidos. Ambas têm série contínua de
# 2020 a 2026, e a troca de lote L12 -> L33 em 2025 não causou quebra no nível
# de volume (Sorocaba: 30-33 mil/dia; Itu: 34-39 mil/dia).
PRACAS_UNIDIRECIONAIS = ["SOROCABA", "ITU"]

# Excluídas da modelagem (registradas aqui só para documentar a decisão):
# - PORTICO ITU 1/2 e PORTICO SALTO 1/2/3: municípios da RMS, mas volume de
#   74 a 156 veículos/dia, ordem de grandeza incompatível com as demais.
# - QUADRA: não pertence à RMS (nem administrativa nem estatisticamente).
# - Alumínio, Araçoiaba Leste, Araçoiaba Oeste e São Roque (praças físicas):
#   unidirecionais, com volume estável, mas DESATIVADAS em 30/09/2025 e
#   substituídas por pórticos -- a série termina cerca de 10 meses antes das
#   demais. Observação: uma versão antiga da base indicava "quedas de 22% a
#   75%" nessas praças em 2025; isso era artefato (comparação de 2024 inteiro
#   com 2025 só até setembro, e nome com espaço extra no arquivo da ARTESP
#   dividindo a mesma praça em duas), não queda real de volume.
# - Pórticos free-flow do lote L33 (out/2025 em diante): série curta demais.

# ---------------------------------------------------------------------------
# Períodos invalidados (dado existe no arquivo da ARTESP, mas não é confiável)
# ---------------------------------------------------------------------------

# Outubro/2022 no lote L13-Colinas: volume ~1,85x o normal nas três praças do
# lote (Boituva, Bloqueio de Boituva, Porto Feliz), ao mesmo tempo em que
# novembro/2022 está ausente no mesmo lote. Confirmado no arquivo bruto
# (VOLUME_PEDAGIADO-L13-DIARIO-2022.csv), não é efeito do pipeline. Alambari
# (outro lote, L20) não apresenta o problema no mesmo mês. A hipótese mais
# provável é consolidação do registro de novembro dentro de outubro, mas como
# a mecânica exata não pode ser comprovada, o mês é tratado como lacuna, em vez
# de tentar dividir o valor entre os dois meses.
PERIODOS_INVALIDOS = {
    "BOITUVA":             [("2022-10-01", "2022-10-31")],
    "BLOQUEIO DE BOITUVA": [("2022-10-01", "2022-10-31")],
    "PORTO FELIZ":         [("2022-10-01", "2022-10-31")],
}

# ---------------------------------------------------------------------------
# Desenho experimental (combinado com o orientador)
# ---------------------------------------------------------------------------

# 2020 fica fora da modelagem (choque de demanda da pandemia, seção 3.1.4).
DATA_INICIO_MODELAGEM = "2021-01-01"

# 70% iniciais da série para treino, 30% finais para teste.
PROPORCAO_TREINO = 0.70

# Horizontes de previsão, em dias.
HORIZONTES = {"curto": 7, "medio": 90, "longo": 180}

# Dentro do período de teste, a previsão é repetida a partir de várias
# "origens" (pontos de partida), deslocadas de 7 em 7 dias. Os parâmetros do
# modelo continuam sendo os estimados só no treino -- o que muda é apenas o
# ponto a partir do qual se prevê. Isso gera várias medidas de erro por praça,
# em vez de uma só, e alimenta o teste de significância estatística depois.
PASSO_ORIGENS_DIAS = 7

# Sazonalidade semanal (sexta alta, domingo baixo).
PERIODO_SAZONAL = 7

# Faixa da restrição de Sakoe-Chiba usada no DTW (dias de deslocamento máximo).
JANELA_DTW = 7

# ---------------------------------------------------------------------------
# Ordens do ARIMA fixadas (reprodutibilidade)
# ---------------------------------------------------------------------------

# A escolha automática (auto_arima, busca "stepwise" pelo menor AIC, só com os
# dados de treino) pode chegar a modelos diferentes em computadores com versões
# diferentes de pandas/numpy -- aconteceu com Sorocaba. Por isso, as ordens
# escolhidas na execução oficial do autor (ambiente registrado em
# versoes_bibliotecas.txt) ficam fixadas aqui. A ordem de ITU foi escolhida
# pelo mesmo procedimento (auto_arima no treino) quando a praça foi incluída,
# e fixada da mesma forma. Formato:
#     praça: ((p, d, q), (P, D, Q, período sazonal))
# Para refazer a escolha automática, apague a praça deste dicionário (ou deixe
# o dicionário vazio) e rode arima_baseline.py de novo.
ORDENS_ARIMA = {
    "ALAMBARI":            ((0, 1, 1), (0, 0, 1, 7)),
    "BOITUVA":             ((3, 1, 2), (0, 0, 2, 7)),
    "BLOQUEIO DE BOITUVA": ((2, 1, 1), (1, 0, 1, 7)),
    "PORTO FELIZ":         ((2, 1, 0), (1, 0, 1, 7)),
    "SOROCABA":            ((2, 1, 2), (1, 0, 1, 7)),
    "ITU":                 ((0, 1, 2), (0, 0, 2, 7)),
}
