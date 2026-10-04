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

# Caso separado: praça física de Sorocaba. Entra no trabalho, mas NÃO na mesma
# tabela comparativa do núcleo, porque é UNIDIRECIONAL (o pedágio só cobra
# quem entra na cidade -- mede metade do fluxo, enquanto o núcleo mede o
# fluxo total). A troca de lote L12 -> L33 em 2025 não causou quebra no nível
# de volume desta praça (permanece na faixa de 30-33 mil veículos/dia), por
# isso a série é modelável por si só.
PRACAS_CASO_SOROCABA = ["SOROCABA"]

# Excluídas da modelagem (registradas aqui só para documentar a decisão):
# - PORTICO ITU 1/2 e PORTICO SALTO 1/2/3: municípios da RMS, mas volume de
#   74 a 156 veículos/dia, ordem de grandeza incompatível com o núcleo.
# - QUADRA: não pertence à RMS (nem administrativa nem estatisticamente).
# - Alumínio, Araçoiaba Leste/Oeste, São Roque, Itu (praça física): volume
#   despenca na transição de concessão em 2025 -- quebra estrutural.

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
