# Sugestões para discutir com o orientador

Os resultados oficiais da monografia só mudam por decisão do orientador. Os refinamentos
validados estão em `RESULTADOS_REFINAMENTO.md`. Abaixo, o que precisa ser decidido ou feito.

## 1. Decisões

1. **Adotar ou não** os três ajustes de `RESULTADOS_REFINAMENTO.md` como configuração principal.
   O texto de `monografia/Resultados_configuracao_ajustada.docx` já assume que sim. Isso
   resolve duas decisões que estavam abertas: a ordem do SARIMA (análise A2) e a LSTM de
   180 dias, que antes ficaria como resultado real, sem retreino.
2. **Raiz sazonal quase unitária no SARIMA:** as ordens de menor AIC têm termo AR sazonal
   próximo de 1 (por isso o auto_arima as descartava). O texto explica isso como uma
   aproximação de diferença sazonal; confirmar se o orientador aceita essa leitura.
3. **Referência (ingênuo sazonal):** manter o ingênuo oficial ou trocar por uma versão mais
   forte (média das últimas 4 semanas ou mesma semana do ano anterior). A troca muda a leitura
   de todos os outros modelos, por isso é decisão metodológica.
4. **Registrar no texto** que foram testadas várias variantes sobre o mesmo período de teste.
   O .docx já cita as variações de configuração rejeitadas pelo critério de treino.

## 2. Validação adicional (para blindar o resultado)

1. **Dados novos:** a série termina em 31/07/2026. Obter agosto a outubro de 2026 e usar como
   teste novo, sem nenhum ajuste, é a prova mais forte de que a melhora se mantém.
2. **Origem móvel no treino:** repetir a escolha das mudanças com validação por origens móveis
   dentro do treino, sem usar o teste.
3. **Mais sementes e repetições** nas redes, para separar ganho real de variação de treino.
4. **Registrar o ambiente** (versões de numpy, scipy e statsmodels): o SARIMA muda de resultado
   com versões diferentes das bibliotecas.

## 3. Experimentos que ficaram em aberto

1. **Calendário como entrada (trabalho futuro da conclusão):** os resultados preliminares
   ficaram fora do texto porque não são consistentes entre horizontes.
   - Em 7 dias, LSTM, GRU, Bi-LSTM e Bi-GRU melhoram cerca de 0,5 ponto (para 5,86 a 6,02),
     e a perda de validação confirma; no N-HiTS o teste melhora, mas a validação piora.
   - Na GRU de 90 dias a validação melhora e o teste piora (6,54 → 6,88); em 180 dias pioram
     os dois. No N-HiTS, 90 e 180 dias pioram muito.
   - No SARIMA, feriados como variável externa baixam o AIC (29570 → 27623) e melhoram 7 dias,
     mas pioram um pouco 180 dias.
   - Antes de usar, definir a regra (por exemplo, por horizonte) sem olhar o teste.
2. **LSTM em 180 dias:** combinar paciência 40 com viés de esquecimento, e repetir nos
   horizontes de 7 e 90 dias.
3. **SARIMA:** testar feriados e Fourier anual sobre a ordem de menor AIC, com critério de
   decisão definido antes de olhar o teste. Resolver a não convergência em Itu.
4. **Prophet:** definir as janelas de feriado por validação interna, e não por conhecimento
   prévio dos erros; avaliar sazonalidade multiplicativa só se a validação a apoiar.
5. **Janela de entrada de 91 dias no horizonte de 7 dias:** só o GRU foi apoiado pela
   validação; refazer com a mesma validação para as demais.
6. **N-HiTS:** estudar por que o calendário piora em 90 e 180 dias; testar dropout e outras
   larguras de camada.
7. **Recorrentes:** GRU de duas camadas e busca de hiperparâmetros (unidades, taxa de
   aprendizado) com validação.
8. **Praças Morro I e II:** incluir, se o orientador autorizar (exige os dados brutos).

## 4. Para a defesa

- Preparar a resposta para "por que essas mudanças e não outras": cada uma saiu do guia, foi
  testada isoladamente e confirmada por critério sem teste.
- Ter à mão a tabela completa (`resultados/RESUMO_experimentos.md`) caso a banca pergunte por
  qualquer variante.
