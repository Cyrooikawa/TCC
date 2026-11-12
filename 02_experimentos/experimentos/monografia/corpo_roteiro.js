// Roteiro de reunião com o orientador (a partir do Prophet) + anexo de melhorias
const CAP = (rotulo, t) => new Paragraph({
  children: [new TextRun({ text: rotulo + " – ", bold: true, size: 20 }), ...runs(t, { size: 20 })],
  keepNext: true, spacing: { before: 200, after: 80, line: 240 },
});
const PP = (t) => P(t, { semRecuo: true });
const B_ = (t) => c.push(ITEM(t));
const TEMPO = (t) => new Paragraph({ children: [new TextRun({ text: t, italics: true, size: 20, color: "595959" })], spacing: { before: 0, after: 120 } });
const DECIDIR = (itens) => {
  c.push(new Paragraph({
    children: [new TextRun({ text: "Para decidir com o orientador", bold: true, size: 22 })],
    shading: { type: ShadingType.CLEAR, color: "auto", fill: "EAF1FB" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2A78D6", space: 6 } },
    spacing: { before: 160, after: 40 }, keepNext: true,
  }));
  itens.forEach((t, i) => c.push(new Paragraph({
    children: [new TextRun({ text: "☐  ", size: 22 }), ...runs(t, { size: 22 })],
    shading: { type: ShadingType.CLEAR, color: "auto", fill: "EAF1FB" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2A78D6", space: 6 } },
    spacing: { before: 0, after: i === itens.length - 1 ? 200 : 40, line: 276 },
    keepNext: i < itens.length - 1,
  })));
};
const DIZER = (t) => PP("**O que mostrar:** " + t);

// ================= capa =================
c.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Roteiro de alinhamento com o orientador")] }));
c.push(new Paragraph({ children: runs("TCC – Previsão do volume diário de tráfego em praças de pedágio da RMS · continuação a partir do Prophet", { size: 22, color: "595959" }), spacing: { after: 240 } }));
PP; c.push(PP("Este roteiro continua a conversa do ponto em que ela parou, o Prophet, e segue a ordem do Guia dos modelos: Prophet, redes neurais (o que vale para todas, LSTM, GRU, bidirecionais e N-HiTS), o calendário como entrada e os temas que atravessam o trabalho. Cada tópico traz o que mostrar, os números e as decisões a tomar. O Anexo reúne todas as melhorias levantadas, testadas ou não, para o orientador avaliar antes de qualquer mudança no código oficial."));
c.push(CAP("Tabela 1", "Pauta sugerida (cerca de 70 minutos)"));
c.push(tabela(["Bloco", "Tópico", "Tempo"], [
  ["0", "Atualização rápida: o que mudou desde a última conversa", "5 min"],
  ["1", "Prophet", "10 min"],
  ["2", "Redes neurais: o que vale para as cinco", "10 min"],
  ["3", "LSTM, GRU e bidirecionais", "10 min"],
  ["4", "N-HiTS", "5 min"],
  ["5", "Calendário como entrada (a melhoria de maior potencial)", "10 min"],
  ["6", "Temas gerais: testes estatísticos, referência e conclusão", "10 min"],
  ["7", "Fechamento: decisões e próximos passos", "10 min"],
], [0.6, 5, 1], { tam: 20 }));
c.push(new Paragraph({ children: runs("Fonte: elaborado pelo autor.", { size: 18 }), spacing: { before: 60, after: 200 } }));

// ================= 0 =================
c.push(H2("0. Atualização rápida (5 min)"));
c.push(TEMPO("Objetivo: situar o orientador antes de retomar o Prophet."));
B_("Depois da última conversa, testei as ideias de refinamento do Guia dos modelos numa cópia separada do projeto, sem mexer nos resultados oficiais. Foram cerca de 35 variações, uma mudança por vez.");
B_("Regra usada: cada mudança foi escolhida por um critério calculado só no treino (AIC no SARIMA, validação dentro do treino no Prophet e nas redes). O teste só mediu o efeito.");
B_("Três ajustes passaram nesse critério e corrigem problemas já descritos na monografia: a ordem do SARIMA, a janela dos feriados longos no Prophet e a paciência da LSTM de 180 dias. O capítulo 4 foi reescrito com eles, mas como proposta: a versão oficial só muda com a aprovação do orientador.");
B_("O SARIMA, discutido na conversa anterior, foi o que mais mudou: com a ordem de menor AIC, o erro cai de 6,8 / 7,7 / 7,6% para 5,9 / 6,5 / 6,4%, e ele passa a segundo lugar, à frente das redes. Vale um minuto para registrar isso antes de seguir.");
DECIDIR(["Aprovar que o SARIMA use a ordem de menor AIC da busca exaustiva (decisão pendente da conversa anterior)."]);

// ================= 1 Prophet =================
c.push(H2("1. Prophet (10 min)"));
c.push(TEMPO("Guia dos modelos, seção 3.3. Capítulo 4: seções 4.1, 4.3.1, 4.4.5 e 4.5.1."));
DIZER("o Prophet soma uma tendência, um padrão semanal, um padrão anual e o efeito dos feriados. É o único modelo que sabe a data e se há feriado à frente.");
c.push(CAP("Tabela 2", "Prophet: resultado e análises de sensibilidade (SMAPE médio das seis praças, %)"));
c.push(tabela(["Variante", "7 dias", "90 dias", "180 dias", "O que mostra"], [
  ["Configuração original (janela de 2 dias antes e 1 depois para todos os feriados)", "4,6", "4,7", "4,7", "Já era o melhor modelo em tudo"],
  ["Configuração ajustada (janela própria para os feriados longos)", "*4,4", "*4,5", "*4,5", "Melhora nos três horizontes e nas seis praças"],
  ["Ajustado uma única vez, só com o treino", "4,6", "4,5", "4,5", "O reajuste em cada origem quase não ajuda"],
  ["Sem o calendário de feriados", "5,5", "5,5", "5,3", "O calendário vale cerca de 1 ponto"],
  ["Sem feriados e sem sazonalidade anual", "6,3", "6,2", "6,1", "No longo prazo, ainda melhor que as redes"],
], [3.2, 0.7, 0.7, 0.7, 2.4], { tam: 18, colsTexto: 1 }));
c.push(new Paragraph({ children: runs("Fonte: elaborado pelo autor.", { size: 18 }), spacing: { before: 60, after: 160 } }));
PP; c.push(PP("**Pontos para falar:**"));
B_("**Resultado:** melhor modelo nas 6 praças, nos 3 horizontes e nas 5 métricas, com diferença significativa contra cada um dos outros sete modelos em 7 dias. O erro não cresce com o horizonte: prever o dia 150 é tão difícil quanto prever o dia 3.");
B_("**O ajuste:** Carnaval, Sexta-feira Santa, Natal e Ano-Novo ganharam janela maior (por exemplo, de quarta a segunda-feira na Páscoa). As janelas foram fixadas pela duração típica da viagem, sem busca de valores, e a validação dentro do treino confirmou a melhora (4,85 → 4,66% em 7 dias).");
B_("**Ponto fraco que continua:** no feriado de abril de 2025 em Boituva, o Prophet antecipou o aumento de quinta e sexta, mas superestimou o domingo (49.453 previstos contra 33.244) e subestimou a segunda (37.293 contra 59.552). Em 2025, Tiradentes caiu na segunda-feira seguinte à Páscoa e deslocou o retorno para esse dia, combinação que não existia nos anos de treino. Essa passou a ser a semana de maior erro do Prophet (17,6%).");
B_("**Outras variações testadas e rejeitadas pelo critério de treino:** sazonalidade multiplicativa, outros valores de changepoint_prior_scale e holidays_prior_scale e o regressor de férias escolares. Nenhuma melhorou a validação; detalhes no Anexo.");
B_("**Correção no texto:** a frase \"o erro de 7 dias do Prophet é o mesmo na primeira e na segunda metade do teste (4,8%)\" foi retirada, porque não consegui reproduzi-la; no recálculo, o erro é maior na segunda metade.");
DECIDIR([
  "Aprovar a janela própria para os feriados longos como configuração principal do Prophet.",
  "Concordar com a explicação do erro na Páscoa de 2025 (coincidência com Tiradentes).",
  "Confirmar a retirada da frase sobre as duas metades do teste.",
  "Manter o reajuste do Prophet em cada origem como avaliação principal (o ajuste único fica como sensibilidade).",
]);

// ================= 2 Redes geral =================
c.push(H2("2. Redes neurais: o que vale para as cinco (10 min)"));
c.push(TEMPO("Guia dos modelos, seção 4.1. Capítulo 4: seções 4.4.3, 4.5.2 e 4.5.3."));
DIZER("cada rede recebe uma janela com os últimos dias de volume e devolve os próximos dias de uma vez. Não recebe a data nem o calendário. Uma rede por horizonte, treinada com as seis praças juntas, cinco sementes.");
B_("**Resultado:** as cinco redes superam o ingênuo, mas ficam atrás do Prophet e do SARIMA ajustado. SMAPE de 6,1 a 6,5% em 7 dias, de 6,5 a 7,4% em 90 e de 6,8 a 7,7% em 180.");
B_("**Onde ganham:** nas semanas seguintes a feriado (erro de 4,9 a 5,5%, contra 9,6% do ingênuo), porque olham mais que uma semana para trás.");
B_("**Onde perdem:** nas semanas com feriado (9,8 a 10,4%, contra 6,9% do Prophet), porque não sabem que vem um feriado.");
B_("**Horizonte:** o erro quase não cresce ao longo dos 180 dias, porque a previsão sai toda de uma vez. Mas a rede treinada para 180 dias erra mais que a de 90 nos mesmos dias (menos janelas de treino: 1.555 contra 3.648). E a rede de 90 dias acertou a primeira semana melhor que a de 7 dias em 4 das 5 arquiteturas, sinal de que a janela de 28 dias é curta.");
B_("**Implementação:** as redes foram escritas em NumPy e conferidas contra o PyTorch (38 de 38 comparações iguais). O texto da seção 3.1.8 ainda menciona NeuralForecast e PyTorch.");
DECIDIR([
  "Manter a implementação em NumPy, documentada na seção 3.1.8 com a conferência contra o PyTorch.",
  "Registrar a janela de 28 dias como limitação (a janela de 91 dias foi testada; ver Anexo, item B5).",
]);

// ================= 3 LSTM GRU Bi =================
c.push(H2("3. LSTM, GRU e bidirecionais (10 min)"));
c.push(TEMPO("Guia dos modelos, seções 4.2 a 4.4. Capítulo 4: seções 4.4.5, 4.5.3 e 4.5.4."));
c.push(CAP("Tabela 3", "Redes recorrentes: SMAPE médio das seis praças (%)"));
c.push(tabela(["Rede", "7 dias", "90 dias", "180 dias", "Comentário"], [
  ["LSTM (original)", "6,4", "7,4", "10,1", "Em 180 dias, pior que o ingênuo: 3 sementes travaram"],
  ["LSTM (ajustada: paciência 40)", "6,4", "7,4", "*7,7", "Ajuste só em 180 dias; erro menor em 91% das origens"],
  ["GRU", "6,4", "*6,5", "7,4", "A mais estável; desvio entre sementes ≤ 0,28 ponto"],
  ["Bi-LSTM", "6,5", "6,9", "7,6", "Dobro do custo, ganho pequeno"],
  ["Bi-GRU", "6,4", "6,5", "7,1", "Diferença para a GRU dentro da variação entre sementes"],
], [2.2, 0.7, 0.7, 0.7, 3.4], { tam: 18, colsTexto: 1 }));
c.push(new Paragraph({ children: runs("Fonte: elaborado pelo autor.", { size: 18 }), spacing: { before: 60, after: 160 } }));
B_("**LSTM de 180 dias:** com a paciência de 20 épocas, três das cinco sementes paravam nas épocas 16 a 21, presas num platô (perda de validação 0,073 contra 0,045 a 0,051 das outras). Com paciência 40 e teto de 400 épocas, as cinco saem do platô (melhor época entre 221 e 256) e o erro cai de 10,1% para 7,7%. Nenhuma outra rede teve esse problema, por isso o ajuste vale só para essa combinação.");
B_("**Atenção:** em conversa anterior, ficou combinado manter a LSTM de 180 dias como resultado real, sem retreino. A proposta muda essa decisão. Alternativa: manter 10,1% como principal e mostrar 7,7% como sensibilidade.");
B_("**Alternativa testada:** iniciar o viés da porta de esquecimento em 1 (Jozefowicz et al., 2015) também ajuda (8,7%), mas menos que a paciência, e a validação preferiu a paciência.");
B_("**GRU e bidirecionais:** a GRU foi a mais estável. A bidirecionalidade custa de 1,2 a 2,2 vezes o tempo e rende pouco; ajudou a LSTM em 90 dias. Variar o número de unidades (32 ou 128) não mudou o resultado de forma consistente.");
DECIDIR([
  "LSTM de 180 dias: adotar a paciência 40 como principal, ou manter o resultado original e mostrar o ajuste como sensibilidade.",
  "Concordar que as bidirecionais ficam no trabalho como comparação, sem recomendação de uso.",
]);

// ================= 4 N-HiTS =================
c.push(H2("4. N-HiTS (5 min)"));
c.push(TEMPO("Guia dos modelos, seção 4.5."));
B_("**Resultado:** melhor rede em 7 dias (6,1%) e em 180 dias (6,8%); treina em 3 a 5 minutos, a mais rápida, apesar de ter de 0,9 a 1,4 milhão de parâmetros. Em 180 dias, é a rede mais próxima do SARIMA ajustado (6,8% contra 6,4%) e a que mais vence o ingênuo origem a origem (62% das origens).");
B_("**Sobreajuste:** no fim do treino, a perda de treino passa a ficar abaixo da de validação; a parada antecipada segurou o efeito.");
B_("**Testado:** a rede menor (256 unidades) não mudou o resultado. O calendário como entrada melhorou o teste em 7 dias (5,6%), mas piorou a validação, e piorou muito em 90 e 180 dias (8,4 e 8,9%). Não há base para adotar.");
DECIDIR(["Manter o N-HiTS na configuração de referência, sem ajustes."]);

// ================= 5 calendário =================
c.push(H2("5. Calendário como entrada (10 min)"));
c.push(TEMPO("Guia dos modelos, seção 4.6. É a discussão mais importante para a conclusão do trabalho."));
DIZER("a maior parte da distância entre as redes e o Prophet está na informação de entrada, não na arquitetura. Sem o calendário, o próprio Prophet perde cerca de 1 ponto; em 7 dias, o calendário explica de 37% a 60% da distância entre o Prophet e a melhor rede.");
B_("**O que foi testado:** dia da semana e feriado, véspera e dia seguinte como entradas das redes; e feriados como variável externa no SARIMA (SARIMAX).");
B_("**Resultado em 7 dias:** LSTM, GRU, Bi-LSTM e Bi-GRU melhoram cerca de 0,5 ponto (para 5,9 a 6,0%), com a validação confirmando. O ganho vem das semanas com feriado (≈ 10,3 → 9,4%).");
B_("**Resultado em 90 e 180 dias:** piora. Na GRU de 90 dias, a validação melhora e o teste piora (6,5 → 6,9%), um caso em que validação e teste discordam. No N-HiTS, piora em todos os horizontes de validação.");
B_("**SARIMA com feriados:** o AIC melhora bastante e o erro de 7 dias cai, mas o de 180 dias piora um pouco.");
B_("**Por que ficou fora do capítulo:** o efeito não é consistente entre horizontes, e adotá-lo só onde funcionou seria escolher pelo teste. Além disso, mudaria a informação que cada modelo recebe, que é parte do desenho combinado.");
DECIDIR([
  "Manter o calendário como trabalho futuro, citando os resultados preliminares na conclusão; ou",
  "Incluir uma análise adicional com o calendário em todos os modelos, com a regra de adoção definida antes (por exemplo, só no horizonte de 7 dias).",
]);

// ================= 6 temas gerais =================
c.push(H2("6. Temas gerais (10 min)"));
B_("**Testes estatísticos:** o capítulo usa a média das seis praças em cada origem como bloco (87 blocos em 7 dias, 6 em 90 e 3 em 180), porque as praças são correlacionadas (0,42 em média). Consequência: em 7 dias, o SARIMA tem erro menor que as redes, mas a diferença não é significativa; em 90 e 180 dias, nenhum teste pode dar significativo, e a comparação é descritiva.");
B_("**Referência ingênua:** versões mais fortes do ingênuo foram testadas (média das últimas 4 semanas: 7,0 / 7,5 / 7,7%; mesma semana do ano anterior: 7,1 / 7,0 / 6,8%). Trocar a referência muda a leitura de todos os modelos; a recomendação é manter a atual.");
B_("**Transparência:** o capítulo informa que outras variações foram testadas e rejeitadas pelo critério de treino, e mostra a configuração original nas Tabelas 13 e 15. Vale confirmar se o orientador quer a lista completa num apêndice.");
B_("**Reprodutibilidade:** o SARIMA ajustado varia um pouco entre computadores (em Itu, 5,60% ou 5,73% em 7 dias). O pacote de validação (VALIDAR_RESULTADOS.bat) roda tudo e confere os números.");
B_("**Conclusão central proposta:** o Prophet é o melhor nos três horizontes e o SARIMA, o segundo; as redes ficam à frente apenas do ingênuo. Em 7 dias, a vantagem do Prophet vem do calendário; em 90 e 180 dias, persiste sem ele.");
DECIDIR([
  "Aprovar os blocos por origem como forma principal dos testes.",
  "Manter o ingênuo atual como referência (versões fortes só como sensibilidade, se quiser).",
  "Definir se a lista completa de variações testadas entra em apêndice.",
  "Aprovar a conclusão central proposta.",
]);

// ================= 7 fechamento =================
c.push(H2("7. Fechamento (10 min)"));
PP; c.push(PP("**Próximos passos, depois das decisões:**"));
B_("Aplicar as decisões no código oficial (só os itens aprovados) e rodar o pacote de validação no meu computador.");
B_("Atualizar o capítulo 4 conforme as decisões e ajustar a seção 3.1.8 (busca da ordem do SARIMA, paciência da LSTM, implementação em NumPy).");
B_("Escrever a Conclusão, o Resumo e o Abstract com os números finais.");
B_("Opcional, se houver tempo: usar os meses de agosto a outubro de 2026 como teste novo, sem nenhum ajuste, para confirmar a ordem dos modelos.");
c.push(H2("Registro das decisões"));
c.push(tabela(["Decisão", "Aprovado", "Rejeitado", "Como sensibilidade", "Observação"], [
  ["SARIMA: ordem de menor AIC", "☐", "☐", "☐", ""],
  ["Prophet: janela dos feriados longos", "☐", "☐", "☐", ""],
  ["LSTM 180 dias: paciência 40", "☐", "☐", "☐", ""],
  ["Calendário como entrada", "☐", "☐", "☐", ""],
  ["Testes com blocos por origem", "☐", "☐", "☐", ""],
  ["Referência ingênua atual", "☐", "☐", "☐", ""],
  ["Implementação das redes em NumPy", "☐", "☐", "☐", ""],
  ["Conclusão central proposta", "☐", "☐", "☐", ""],
], [2.8, 0.8, 0.8, 1.1, 2], { tam: 18, colsTexto: 1 }));

// ================= ANEXO =================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H1("Anexo – Melhorias sugeridas para avaliação"));
PP; c.push(PP("Nenhuma melhoria abaixo foi aplicada ao código oficial (pasta 02_codigo_e_resultados). As que foram testadas rodaram numa cópia separada; os três ajustes do grupo A estão disponíveis nessa cópia por uma chave em config.py, que permite rodar a configuração original ou a ajustada. Os valores são SMAPE médio das seis praças, em 7 / 90 / 180 dias, comparados com a configuração original. \"Critério no treino\" é a medida usada para decidir sem olhar o teste."));
c.push(H2("A. Testadas e aprovadas pelo critério no treino (propostas para adoção)"));
c.push(tabela(["#", "Melhoria", "Antes", "Depois", "Critério no treino", "Custo", "Recomendação"], [
  ["A1", "SARIMA: ordem de menor AIC na busca exaustiva (144 ordens)", "6,8 / 7,7 / 7,6", "5,9 / 6,5 / 6,4", "AIC menor nas 6 praças", "Baixo (já feito)", "Adotar"],
  ["A2", "Prophet: janela própria para Carnaval, Sexta-feira Santa, Natal e Ano-Novo", "4,6 / 4,7 / 4,7", "4,4 / 4,5 / 4,5", "Validação 4,85 → 4,66 (7 d)", "Baixo", "Adotar"],
  ["A3", "LSTM 180 dias: paciência 40 e teto de 400 épocas", "10,1 (180 d)", "7,7 (180 d)", "Perda 0,063 → 0,044", "+3 min de treino", "Adotar ou usar como sensibilidade"],
], [0.4, 2.6, 1, 1, 1.5, 1, 1.3], { tam: 16, colsTexto: 2 }));
c.push(H2("B. Testadas, com resultado misto ou sem efeito (para discutir)"));
c.push(tabela(["#", "Melhoria", "Resultado no teste", "Critério no treino", "Leitura", "Recomendação"], [
  ["B1", "Calendário (dia da semana e feriados) como entrada das redes", "7 d: 5,9 a 6,0 nas recorrentes (era 6,4 a 6,5); 90 e 180 d: piora; N-HiTS: 5,6 / 8,4 / 8,9", "Confirma em 7 d nas recorrentes; não no N-HiTS; GRU 90 d discorda do teste", "Promissor só no curto prazo", "Trabalho futuro ou análise adicional com regra definida antes"],
  ["B2", "SARIMA com feriados como variável externa (SARIMAX)", "6,3 / 7,4 / 7,7 (sobre a ordem original)", "AIC 29.570 → 27.623", "Ajuda em 7 d, piora levemente em 180 d", "Trabalho futuro, junto com B1"],
  ["B3", "SARIMA com termos de Fourier anuais", "6,7 / 7,3 / 7,3 (3 harmônicos)", "AIC quase igual", "Ganho pequeno", "Não adotar"],
  ["B4", "Prophet com sazonalidade multiplicativa", "4,6 / 4,7 / 4,7; com A2: 4,4 / 4,5 / 4,5", "Validação piora", "Melhora no teste não confirmada", "Não adotar"],
  ["B5", "Janela de entrada de 91 dias no horizonte de 7", "GRU 6,0; Bi-GRU 6,1; Bi-LSTM 6,2; N-HiTS 6,0; LSTM 6,6", "Só a GRU melhora (comparação aproximada)", "Inconsistente; treino 2 a 3 vezes mais lento", "Citar como limitação"],
  ["B6", "LSTM: viés da porta de esquecimento em 1", "8,7 (180 d)", "Perda 0,063 → 0,053", "Alternativa à A3, pior que ela", "Não adotar (ou combinar com A3 no futuro)"],
  ["B7", "Unidades: GRU com 32 ou 128; N-HiTS com 256", "Diferenças de −0,2 a +0,2", "Perda praticamente igual", "Sem efeito", "Não adotar"],
  ["B8", "Prophet: férias escolares e outros priors", "Diferenças ≤ 0,02; changepoint 0,005 piora", "Sem melhora", "Sem efeito", "Não adotar"],
  ["B9", "Ingênuo mais forte (média de 4 semanas; mesma semana do ano anterior)", "7,0 / 7,5 / 7,7 e 7,1 / 7,0 / 6,8", "Validação confirma a média de 4 semanas", "Muda a régua de todos os modelos", "Manter o atual; citar como sensibilidade, se quiser"],
], [0.4, 2, 2, 1.6, 1.4, 1.5], { tam: 16, colsTexto: 6 }));
c.push(H2("C. Não testadas (trabalho futuro)"));
c.push(tabela(["#", "Melhoria", "Por que vale a pena", "Esforço", "Depende do orientador"], [
  ["C1", "Teste novo com agosto a outubro de 2026", "É a prova mais forte de que a ordem dos modelos se mantém", "Baixo, se os dados estiverem disponíveis", "Sim"],
  ["C2", "Validação por origem móvel dentro do treino", "Escolha das configurações mais robusta que os 20% finais", "Médio", "Não"],
  ["C3", "Busca sistemática de hiperparâmetros das redes", "Hoje as redes usam configuração fixa", "Alto (horas de execução)", "Sim"],
  ["C4", "GRU com duas camadas; dropout no N-HiTS", "Captar padrões mais longos; reduzir sobreajuste", "Médio (muda redes_numpy.py)", "Não, se como sensibilidade"],
  ["C5", "Incluir as praças Morro I e II", "Mais séries para o modelo global das redes", "Médio (dados brutos)", "Sim"],
  ["C6", "Mais sementes nas redes", "Separar ganho real de variação de treino", "Médio", "Não"],
], [0.4, 2.2, 2.6, 1.4, 1.2], { tam: 16, colsTexto: 5 }));
PP; c.push(PP("**Como avaliar antes de incluir:** cada item aprovado entra no código oficial como mudança isolada, roda-se o pacote de validação (VALIDAR_RESULTADOS.bat) e confere-se o resultado com o esperado antes de atualizar o texto. Os itens do grupo A já têm valores esperados registrados (esperado_configuracao_ajustada.json)."));
