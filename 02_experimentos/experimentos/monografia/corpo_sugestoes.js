// Corpo do documento de sugestões ao orientador
const H2s = (t) => H2(t);
const B_ = (t) => c.push(ITEM(t));
const PP = (t) => P(t, { semRecuo: true });

c.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Sugestões para o orientador")] }));
c.push(new Paragraph({ children: runs("TCC – Previsão do volume diário de tráfego em praças de pedágio da RMS · Cyro Oikawa · novembro de 2026", { size: 22, color: "595959" }), spacing: { after: 240 } }));
c.push(PP("Este documento reúne as decisões e os próximos passos que dependem do orientador, depois dos testes de refinamento dos modelos descritos no Guia dos modelos. Os testes foram feitos numa cópia separada do projeto, sem alterar os resultados oficiais. O texto da seção 4 já preparado para a monografia assume a configuração ajustada descrita abaixo."));

c.push(H2s("1. O que foi feito"));
c.push(PP("Foram testadas cerca de 35 variações de configuração nos oito modelos, sempre com uma mudança por vez, as mesmas origens e as mesmas métricas. Para cada variação foi calculado um critério que não usa o conjunto de teste: o AIC no treino (SARIMA), o SMAPE em origens semanais dentro dos 20% finais do treino (ingênuo e Prophet) e a perda de validação (redes). Três ajustes atendem a esse critério, corrigem problemas já descritos na monografia e não acrescentam informação aos modelos:"));
c.push(CAP("Tabela 1", "Ajustes propostos como configuração principal (SMAPE médio das seis praças, %)"));
c.push(tabela(["Modelo", "Ajuste", "Critério no treino", "Original (7 / 90 / 180 d)", "Ajustado (7 / 90 / 180 d)"], [
  ["SARIMA", "Ordem de menor AIC entre as 144 da grade", "AIC 29.570 → 27.868", "6,77 / 7,66 / 7,60", "5,92 / 6,49 / 6,46"],
  ["Prophet", "Janela própria para Carnaval, Sexta-feira Santa, Natal e Ano-Novo", "SMAPE de validação 4,85 / 4,58 → 4,66 / 4,45", "4,64 / 4,71 / 4,68", "4,43 / 4,49 / 4,47"],
  ["LSTM (180 d)", "Paciência de 40 épocas e teto de 400", "Perda de validação 0,063 → 0,044", "10,15", "7,67"],
], [1, 2.2, 1.8, 1.4, 1.4], { tam: 17 }));
c.push(FONTE_A());
c.push(PP("Com esses ajustes, o Prophet continua o melhor modelo em todas as métricas, praças e horizontes; o SARIMA passa a segundo nos três horizontes e supera o ingênuo em todas as praças; as redes ficam entre o SARIMA e o ingênuo."));

c.push(H2s("2. Decisões pedidas ao orientador"));
B_("**Adotar os três ajustes como configuração principal.** A seção 4 preparada já assume que sim e mostra os valores da configuração original numa tabela de antes e depois. A adoção encerra duas decisões que estavam abertas: a ordem do SARIMA (análise A2 de 08/11) e a LSTM de 180 dias, que antes seria mantida sem retreino.");
B_("**Raiz sazonal quase unitária no SARIMA.** As ordens de menor AIC têm o termo autorregressivo sazonal muito próximo de 1, motivo pelo qual o auto_arima as descartava. O texto apresenta essa estrutura como uma aproximação de diferença sazonal de 7 dias. Confirmar se essa leitura é aceita.");
B_("**Forma dos testes de significância (análise A3).** A seção 4 usa os 522 blocos de praça e origem, como prevê a seção 3.2.8, e apresenta a variante com a média das praças por origem como limitação. Na variante conservadora, o SARIMA deixa de diferir das redes em 7 dias. Decidir qual das duas é a principal.");
B_("**Referência do estudo (ingênuo sazonal).** Versões mais fortes do ingênuo também foram testadas: a média das últimas 4 semanas (SMAPE de 6,98 / 7,52 / 7,66) e a mesma semana do ano anterior, corrigida pelo crescimento (7,09 / 6,97 / 6,81). Trocar a referência muda a leitura de todos os modelos; a recomendação é manter o ingênuo atual e citar as versões fortes, se for o caso, como sensibilidade.");
B_("**Transparência sobre os testes.** O texto informa que outras variações de configuração foram avaliadas e rejeitadas pelo critério de treino. Confirmar se esse registro é suficiente ou se deve entrar uma tabela com todas as variações.");

c.push(H2s("3. Validação adicional recomendada"));
B_("**Período de teste novo.** A série termina em 31/07/2026. Usar os meses seguintes (agosto a outubro de 2026), sem nenhum ajuste, como teste adicional é a forma mais forte de confirmar que a ordem dos modelos se mantém.");
B_("**Validação por origem móvel dentro do treino.** Repetir a escolha dos ajustes com várias origens móveis no treino, e não só com os 20% finais.");
B_("**Ambiente de execução.** O SARIMA muda de resultado com versões diferentes de numpy e scipy. Registrar as versões usadas e rodar tudo no computador do autor antes da versão final.");

c.push(H2s("4. Experimentos em aberto (trabalho futuro)"));
B_("**Calendário como entrada das redes e do SARIMA.** Ficou fora dos resultados porque o efeito não é consistente entre horizontes. Em 7 dias, LSTM, GRU, Bi-LSTM e Bi-GRU melhoram cerca de 0,5 ponto (para 5,86 a 6,02%), e a perda de validação confirma; no N-HiTS, a validação piora. Em 90 dias, a GRU melhora na validação e piora no teste (6,54 → 6,88%); em 180 dias e no N-HiTS em 90 e 180 dias, o erro aumenta. No SARIMA, os feriados como variáveis externas reduzem o AIC (29.570 → 27.623) e melhoram 7 dias, mas pioram levemente 180 dias. Antes de usar, é preciso definir a regra de adoção por horizonte sem olhar o teste. Isso apoia o trabalho futuro já previsto na conclusão: dar o calendário a todos os modelos.");
B_("**LSTM de 180 dias.** Testar a paciência maior junto com o viés da porta de esquecimento em 1, e estender a paciência maior às demais redes e horizontes.");
B_("**SARIMA.** Testar feriados e termos de Fourier sobre a ordem de menor AIC, com a regra de decisão definida antes; em uma das combinações, o otimizador não convergiu em Itu.");
B_("**Prophet.** Testar outras janelas de feriado e a sazonalidade multiplicativa com validação interna; a combinação das janelas longas com sazonalidade multiplicativa reduziu o erro no teste, mas a validação preferiu as janelas longas sozinhas.");
B_("**Redes.** Janela de entrada de 91 dias no horizonte de 7 dias (só a GRU teve apoio da validação), GRU com duas camadas, dropout no N-HiTS e busca sistemática de hiperparâmetros com validação.");
B_("**Praças Morro I e II.** Incluir, se autorizado, a partir dos dados brutos.");

c.push(H2s("5. O que falta no texto da monografia"));
B_("Aplicar as alterações das seções 4.1 a 4.3 listadas no arquivo da seção 4 (descrição do SARIMA e do Prophet, Quadro 1, Tabela 6, frases sobre o SARIMA e o DTW).");
B_("Refazer as figuras de previsão com a configuração ajustada: SARIMA e Prophet em Boituva (Figura 6 e afins), componentes do Prophet e testes de mesa das semanas de Boituva.");
B_("Atualizar a Conclusão, o Resumo e o Abstract com os números da configuração ajustada.");

c.push(H2s("6. Para a defesa"));
B_("**\"Por que esses ajustes e não outros?\"** Cada um corrige um problema já diagnosticado na primeira execução, muda um único elemento, não acrescenta informação e foi escolhido por um critério calculado só no treino.");
B_("**\"Os ajustes não favorecem os modelos estatísticos?\"** O ajuste da LSTM de 180 dias favorece as redes, e as variações testadas nas demais redes mudaram pouco a perda de validação. O calendário, que favoreceria as redes em 7 dias, fica como trabalho futuro porque acrescenta informação que os modelos não recebiam e porque seu efeito não é consistente entre horizontes; dado também ao SARIMA, ele o melhoraria em 7 dias.");
B_("Manter à mão a tabela completa de todas as variações testadas (02_experimentos/experimentos/resultados/RESUMO_experimentos.md), caso a banca pergunte por alguma delas.");
