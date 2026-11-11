const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  HeadingLevel, BorderStyle, ShadingType, ImageRun, LevelFormat, PageBreak,
} = require("docx");

const S = "/tmp/claude-0/-home-user-TCC/f17788a8-ecde-51ec-9026-5b920241eecd/scratchpad";
const D = JSON.parse(fs.readFileSync(path.join(S, "export/dados.json"), "utf8"));
const FIG = path.join(S, "refinado/figuras");
const SAIDA = process.argv[2];

const FONTE = "Times New Roman";
const LARG = 9071; // A4 com margens de 3 cm (esq.) e 2 cm (dir.)
const CINZA = "F2F2F2";

// ---------- texto ----------
function runs(texto, base = {}) {
  // **negrito** e _itálico_ simples
  const partes = texto.split(/(\*\*[^*]+\*\*|~[^~]+~)/).filter(Boolean);
  return partes.map((p) => {
    if (p.startsWith("**")) return new TextRun({ text: p.slice(2, -2), bold: true, ...base });
    if (p.startsWith("~") && p.endsWith("~") && p.length > 2) return new TextRun({ text: p.slice(1, -1), italics: true, ...base });
    return new TextRun({ text: p, ...base });
  });
}
const P = (t, o = {}) => new Paragraph({
  children: runs(t), alignment: AlignmentType.JUSTIFIED,
  indent: o.semRecuo ? undefined : { firstLine: 709 },
  spacing: { line: 360, before: 0, after: 120 }, ...o.par,
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
// instrução para o autor: caixa cinza, não vai para a monografia
const NOTA = (t) => new Paragraph({
  children: runs(t, { italics: true, color: "595959", size: 20 }),
  shading: { type: ShadingType.CLEAR, color: "auto", fill: CINZA },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: "A6A6A6", space: 6 } },
  spacing: { before: 120, after: 160, line: 276 },
});
const LEGENDA = (t) => new Paragraph({
  children: runs(t, { size: 20 }), alignment: AlignmentType.CENTER, keepNext: true,
  spacing: { before: 240, after: 80, line: 240 },
});
const FONTE_TAB = (t = "Fonte: elaborado pelo autor.") => new Paragraph({
  children: runs(t, { size: 18 }), spacing: { before: 60, after: 240, line: 240 },
});
const ITEM = (t) => new Paragraph({
  children: runs(t), numbering: { reference: "marcas", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, after: 60 },
});
const ATUAL = (t) => new Paragraph({
  children: [new TextRun({ text: "Texto atual: ", bold: true, size: 20, color: "595959" }), ...runs(t, { size: 20, color: "595959", italics: true })],
  spacing: { before: 60, after: 60, line: 276 },
});
const PROPOSTO = (t) => new Paragraph({
  children: [new TextRun({ text: "Texto proposto: ", bold: true }), ...runs(t)],
  alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 200, line: 360 },
});

// ---------- tabelas ----------
const borda = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const nada = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function tabela(cab, linhas, pesos, o = {}) {
  const tam = o.tam || 18;
  const soma = pesos.reduce((a, b) => a + b, 0);
  const w = pesos.map((p) => Math.floor((p / soma) * LARG));
  w[w.length - 1] += LARG - w.reduce((a, b) => a + b, 0);
  const cel = (txt, i, ehCab, ultima, grupoFim) => {
    let t = String(txt), negrito = ehCab;
    if (t.startsWith("*")) { t = t.slice(1); negrito = true; }
    return new TableCell({
      width: { size: w[i], type: WidthType.DXA },
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      borders: {
        top: ehCab ? borda : nada,
        bottom: ehCab || ultima ? borda : (grupoFim ? { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" } : nada),
        left: nada, right: nada,
      },
      children: [new Paragraph({
        alignment: i < (o.colsTexto || 1) ? AlignmentType.LEFT : AlignmentType.CENTER,
        spacing: { line: 240, before: 0, after: 0 },
        children: [new TextRun({ text: t, bold: negrito, size: tam, font: FONTE })],
      })],
    });
  };
  const rows = [new TableRow({ tableHeader: true, children: cab.map((c, i) => cel(c, i, true, false)) })];
  linhas.forEach((l, k) => {
    const grupoFim = o.grupo && k < linhas.length - 1 && (k + 1) % o.grupo === 0;
    rows.push(new TableRow({ cantSplit: true, children: l.map((c, i) => cel(c, i, false, k === linhas.length - 1, grupoFim)) }));
  });
  return new Table({ width: { size: LARG, type: WidthType.DXA }, columnWidths: w, rows });
}
function figura(arq, larguraPx = 600) {
  const buf = fs.readFileSync(path.join(FIG, arq));
  const W = buf.readUInt32BE(16), Hh = buf.readUInt32BE(20);
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: larguraPx, height: Math.round(larguraPx * Hh / W) } })],
  });
}
// coluna repetida (horizonte/praça) só na primeira linha do grupo
function agrupar(linhas, col = 0) {
  let ant = null;
  return linhas.map((l) => { const n = [...l]; if (n[col] === ant) n[col] = ""; else ant = n[col]; return n; });
}

const MOD = ["Ingênuo sazonal", "SARIMA", "Prophet", "LSTM", "GRU", "Bi-LSTM", "Bi-GRU", "N-HiTS"];
const MODC = ["Ingênuo", "SARIMA", "Prophet", "LSTM", "GRU", "Bi-LSTM", "Bi-GRU", "N-HiTS"];

const c = [];

// ================= capa / instruções =================
c.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Resultados com a configuração ajustada")] }));
c.push(new Paragraph({ children: runs("Texto e tabelas para colar na monografia", { size: 24, color: "595959" }), spacing: { after: 240 } }));
c.push(NOTA("Como usar este arquivo. Tudo o que está em caixa cinza é instrução para você e não vai para a monografia. O restante pode ser colado como está. A numeração aparece como [X]: ajuste à sequência do seu texto. As tabelas estão em Times New Roman 9, com título acima e fonte abaixo, no padrão do IBGE adotado pela ABNT; valores em negrito são o menor erro da linha."));
c.push(NOTA("O arquivo tem três partes. Parte 1: a seção nova que descreve os ajustes de configuração. Parte 2: as tabelas e figuras de resultados recalculadas com a configuração ajustada, para substituir as atuais. Parte 3: os trechos do texto que mudam com a nova narrativa, com a redação atual (resumida) e a proposta."));
c.push(NOTA("O termo usado é \"ingênuo sazonal\". Se a troca geral para \"persistência sazonal\" for feita, troque aqui também. Os números da configuração original continuam registrados na Tabela [X] da Parte 1, o que mantém a comparação transparente."));

// ================= PARTE 1 =================
c.push(H1("Parte 1. Seção nova: ajustes de configuração"));
c.push(NOTA("Sugestão de posição: seção própria no capítulo 4, logo depois da descrição do experimento (4.1) e antes dos resultados por modelo, porque todas as tabelas comparativas passam a usar a configuração ajustada."));
c.push(H2("4.X Ajustes de configuração dos modelos"));
c.push(P("Na primeira execução do experimento, três resultados mostraram comportamentos ligados à configuração dos modelos, e não à classe de modelo: a previsão do SARIMA perdia o ciclo semanal depois de poucas semanas; o erro do Prophet se concentrava nas semanas com feriados prolongados; e, na LSTM de 180 dias, três das cinco sementes ficaram presas num platô no início do treino. Para cada um desses pontos foi avaliado um ajuste de configuração, descrito nesta seção. Os resultados comparativos deste capítulo usam a configuração ajustada, e a Tabela [X] registra os valores da configuração original."));
c.push(P("Cada ajuste mudou um único elemento e manteve o restante do protocolo: a mesma divisão entre treino e teste, as mesmas origens semanais (87, 75 e 62 nos horizontes de 7, 90 e 180 dias), as seis praças, as mesmas métricas e, nas redes, as mesmas cinco sementes. Nenhum ajuste acrescentou informação que o modelo não recebia antes. A escolha foi feita por um critério calculado só com os dados de treino, próprio de cada modelo: o AIC no SARIMA; o SMAPE em origens semanais dentro dos 20% finais do treino (de 16/02 a 26/11/2024) no Prophet; e a perda de validação na LSTM, a mesma usada na parada antecipada. O conjunto de teste serviu apenas para medir o efeito de cada ajuste."));
c.push(P("Outras variações de configuração foram avaliadas da mesma forma e tiveram critério de treino pior que o do ajuste escolhido: no SARIMA, o termo de deriva e os termos de Fourier para a sazonalidade anual; no Prophet, a sazonalidade multiplicativa e outros valores de ~changepoint_prior_scale~ e ~holidays_prior_scale~; na LSTM de 180 dias, o viés da porta de esquecimento iniciado em 1. Nas demais redes, testes exploratórios com outros números de unidades e com janela de entrada de 91 dias no horizonte de 7 dias mudaram pouco a perda de validação e não foram adotados. Ajustes que acrescentam informação de calendário a modelos que não a recebiam, como os feriados como variável externa no SARIMA ou como entrada das redes, ficam como trabalho futuro (seção 5)."));

c.push(H3("4.X.1 SARIMA: ordem de menor AIC"));
c.push(P("Na configuração original, a ordem de cada praça foi escolhida pelo ~auto_arima~ em busca passo a passo. Essa busca tem duas limitações que apareceram nos dados: percorre só parte do espaço de ordens e descarta, sem aviso, os modelos com raiz inversa acima de 0,99. Por isso, em Alambari, Boituva e Itu, a ordem escolhida não tinha termo autorregressivo sazonal, e a previsão perdia o ciclo semanal em poucas semanas, convergindo para um nível constante (Figura [X])."));
c.push(P("No ajuste, as 144 combinações do mesmo espaço de busca (p e q até 3, P e Q até 2, d = 1 e D = 0) foram ajustadas na série de treino de cada praça, com as lacunas tratadas pelo filtro de Kalman, como no ajuste que gera as previsões. Foi escolhida a ordem de menor AIC entre as que convergiram. A Tabela [X] compara as ordens."));
c.push(LEGENDA("Tabela [X] – Ordens do SARIMA na configuração original e na ajustada"));
c.push(tabela(["Praça", "Ordem original", "AIC", "Ordem ajustada", "AIC", "Ljung-Box (p)"], D.ordens, [2.4, 1.9, 1.1, 1.9, 1.1, 1.3]));
c.push(FONTE_TAB("Nota: AIC calculado no treino, com a série em milhares de veículos. Ljung-Box: valor-p do teste nos resíduos da ordem ajustada, com 14 defasagens. Fonte: elaborado pelo autor."));
c.push(P("A ordem (3,1,2)(1,0,2)7 foi escolhida em cinco praças, e a (0,1,2)(2,0,1)7 em Bloqueio de Boituva. O AIC cai em todas as praças. O teste de Ljung-Box não rejeita a hipótese de resíduos não correlacionados em Alambari, Porto Feliz, Sorocaba e Itu; em Bloqueio de Boituva fica no limite (p = 0,050), e em Boituva rejeita (p = 0,001). Nas ordens escolhidas, o termo autorregressivo sazonal fica muito próximo de 1, motivo pelo qual o ~auto_arima~ as descartava. Essa estrutura se aproxima de uma diferença sazonal de 7 dias combinada com médias móveis sazonais, o que mantém o ciclo semanal ao longo de todo o horizonte."));
c.push(P("Com a nova ordem, o SMAPE médio do SARIMA cai de 6,77, 7,66 e 7,60% para 5,92, 6,49 e 6,46% nos horizontes de 7, 90 e 180 dias, com erro menor em 71% a 81% das origens (Tabela [X])."));

c.push(H3("4.X.2 Prophet: janelas dos feriados prolongados"));
c.push(P("Na configuração original, todos os feriados usam a mesma janela de efeito: dois dias antes e um depois. Nos feriados que costumam gerar viagens de vários dias, o efeito no tráfego começa antes e termina depois dessa janela. O ajuste deu a quatro deles uma janela própria, definida pela duração típica do deslocamento: Carnaval, de quatro dias antes a dois depois; Sexta-feira Santa, de dois dias antes a três depois (de quarta a segunda-feira); Natal, de quatro dias antes a dois depois; e Confraternização Universal, de quatro dias antes a três depois. Os demais feriados mantiveram a janela original. As janelas foram fixadas antes da avaliação, sem busca de valores."));
c.push(P("No SMAPE das origens de validação, dentro do treino, o erro caiu de 4,85 para 4,66% em 7 dias e de 4,58 para 4,45% em 90 dias. No teste, o SMAPE cai de 0,21 a 0,22 ponto percentual nos três horizontes, de 4,64, 4,71 e 4,68% para 4,43, 4,49 e 4,47%, com erro menor nas seis praças e em 58% a 81% das origens. O ganho aparece nos três tipos de semana: 7,1 para 6,9% nas semanas com feriado, 3,3 para 3,1% nas semanas comuns e 3,4 para 3,2% nas semanas seguintes a um feriado."));

c.push(H3("4.X.3 LSTM de 180 dias: paciência da parada antecipada"));
c.push(P("Na configuração original, as redes interrompem o treino quando a perda de validação passa 20 épocas sem melhorar, com teto de 200 épocas. Na LSTM de 180 dias, três das cinco sementes chegaram ao melhor resultado nas épocas 16, 19 e 21, com perda de validação de 0,073, contra 0,045 e 0,051 das outras duas, e foram interrompidas antes de sair desse platô. Nas demais redes e horizontes, as cinco sementes terminaram com perdas próximas entre si, com diferença máxima de 0,005."));
c.push(P("O ajuste ampliou a paciência para 40 épocas e o teto para 400, só nessa rede. As cinco sementes saíram do platô e chegaram ao melhor resultado entre as épocas 221 e 256, com perda de validação de 0,043 a 0,046, no mesmo nível das demais redes de 180 dias (0,044 a 0,048). O SMAPE de 180 dias cai de 10,15 para 7,67%, com erro menor em 91% das origens e nas seis praças. O SMAPE de cada semente fica entre 7,1 e 8,1%, e o tempo de treino passa de 22 para 25 minutos."));
c.push(LEGENDA("Tabela [X] – Efeito dos ajustes de configuração no SMAPE médio das seis praças (%)"));
c.push(tabela(["Modelo", "Horizonte", "Original", "Ajustado", "Diferença (p.p.)", "Origens com erro menor", "Praças com erro menor"],
  agrupar(D.antes_depois), [1.2, 1.1, 1, 1, 1.2, 1.3, 1.3], { colsTexto: 2 }));
c.push(FONTE_TAB("Nota: a diferença é o valor ajustado menos o original; negativo indica erro menor. Fonte: elaborado pelo autor."));

// ================= PARTE 2 =================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H1("Parte 2. Tabelas e figuras com a configuração ajustada"));
c.push(NOTA("Estas tabelas e figuras substituem as de mesmo conteúdo na monografia: a tabela consolidada (atual Tabela 8), o SMAPE e o DTW por praça (atuais Tabelas 4 e 7), os tipos de semana (atual Tabela 9), as faixas de antecedência (Figura 11) e o mapa de SMAPE por praça. Foram geradas pelo próprio resultados_completos.py e pelo faixas_antecedencia_direto.py, sem alteração de código, trocando só os resultados do SARIMA, do Prophet e da LSTM de 180 dias. Os modelos sem ajuste têm os mesmos valores de antes."));

c.push(LEGENDA("Tabela [X] – Desempenho consolidado dos oito modelos, média das seis praças"));
c.push(tabela(["Horizonte", "Modelo", "SMAPE (%)", "MAPE (%)", "MAE rel.", "RMSE rel.", "DTW rel.", "Origens em que vence o ingênuo"],
  agrupar(D.t8), [1.1, 1.5, 1, 1, 1, 1, 1, 1.5], { colsTexto: 2, grupo: 8 }));
c.push(FONTE_TAB("Nota: MAE, RMSE e DTW divididos pelos do ingênuo sazonal em cada praça e resumidos pela média geométrica das praças. Fonte: elaborado pelo autor."));
c.push(P("O Prophet tem o menor erro nas cinco métricas, nas seis praças e nos três horizontes. O SARIMA fica em segundo no SMAPE nos três horizontes, à frente de todas as redes neurais. Entre as redes, o N-HiTS tem o menor erro em 7 e 180 dias, e a GRU em 90 dias. Todos os modelos superam o ingênuo sazonal nos três horizontes."));

c.push(LEGENDA("Tabela [X] – SMAPE (%) por praça, horizonte e modelo"));
c.push(tabela(["Praça", "Dias", ...MODC], agrupar(D.t4), [2.1, 0.7, 1, 1, 1, 1, 1, 1, 1, 1], { colsTexto: 1, grupo: 3, tam: 16 }));
c.push(FONTE_TAB());
c.push(figura("smape_praca_modelo.png", 600));
c.push(LEGENDA("Figura [X] – SMAPE por praça e modelo"));
c.push(FONTE_TAB());

c.push(LEGENDA("Tabela [X] – DTW (veículos por dia) por praça, horizonte e modelo"));
c.push(tabela(["Praça", "Dias", ...MODC], agrupar(D.t7), [2.1, 0.7, 1, 1, 1, 1, 1, 1, 1, 1], { colsTexto: 1, grupo: 3, tam: 16 }));
c.push(FONTE_TAB());
c.push(P("O DTW do SARIMA fica abaixo do DTW do ingênuo sazonal em todas as praças e horizontes. Com a ordem ajustada, a previsão mantém o ciclo semanal, e deixa de existir a divergência entre SMAPE e DTW observada em Boituva e Itu na configuração original."));
c.push(figura("dtw_relativo_praca_modelo.png", 600));
c.push(LEGENDA("Figura [X] – DTW relativo ao ingênuo sazonal por praça e modelo"));
c.push(FONTE_TAB());

c.push(LEGENDA("Tabela [X] – SMAPE de 7 dias (%) por tipo de semana e participação no erro total"));
c.push(tabela(["Tipo de semana", "Janelas", ...MODC], D.t9, [2.3, 0.9, 1, 1, 1, 1, 1, 1, 1, 1], { colsTexto: 1, tam: 16 }));
c.push(FONTE_TAB("Nota: entre parênteses, a participação de cada tipo de semana no erro total do modelo. Fonte: elaborado pelo autor."));
c.push(P("As semanas com feriado no período previsto são 34% das janelas de 7 dias e concentram de 53% a 58% do erro de todos os modelos. Na semana seguinte a um feriado, as redes erram de 4,9% a 5,5%, o SARIMA 4,6% e o Prophet 3,2%, contra 9,6% do ingênuo sazonal, que repete a semana do feriado."));

c.push(LEGENDA("Tabela [X] – SMAPE (%) por faixa de antecedência, previsões de 180 dias"));
c.push(tabela(["Modelo", "Dias 1 a 7", "Dias 8 a 90", "Dias 91 a 180"], D.faixas, [2, 1, 1, 1]));
c.push(FONTE_TAB("Nota: média das seis praças nas 62 origens com previsão de 180 dias; nas redes, calculado direto da previsão diária. Fonte: elaborado pelo autor."));
c.push(figura("faixas_antecedencia.png", 600));
c.push(LEGENDA("Figura [X] – SMAPE por faixa de antecedência"));
c.push(FONTE_TAB());
c.push(P("Nos dias 91 a 180, o SARIMA (6,53%) erra menos que todas as redes, de 6,96% no N-HiTS a 7,78% na LSTM, e só o Prophet (4,53%) erra menos que ele. O erro das redes praticamente não cresce depois do dia 90: muda no máximo 0,2 ponto percentual entre as faixas de 8 a 90 e de 91 a 180 dias."));

c.push(LEGENDA("Tabela [X] – Teste de Wilcoxon com correção de Holm, 7 dias, média das praças por origem"));
c.push(tabela(["", ...D.sig_cols.map((x) => x.replace(" sazonal", ""))], D.sig, [1.4, 1, 1, 1, 1, 1, 1, 1], { tam: 16 }));
c.push(FONTE_TAB("Nota: valor-p ajustado por Holm nos 28 pares, com 87 blocos (uma origem semanal, média das seis praças). Em negrito, p < 0,05. Teste de Friedman: p < 0,001 em 7 dias (87 blocos), 0,006 em 90 dias (6 blocos) e 0,017 em 180 dias (3 blocos). Fonte: elaborado pelo autor."));
c.push(NOTA("Esta tabela usa a média das praças por origem como bloco, a forma recomendada na análise A3 por causa da correlação entre praças. Se o orientador mantiver a Tabela 10 atual, com 522 blocos, os valores recalculados estão em tabelas_csv/completos_tabela10_significancia.csv, junto deste arquivo."));
c.push(P("Em 7 dias, o teste de Friedman rejeita a igualdade dos modelos (p < 0,001). Após a correção de Holm, o Prophet difere de todos os modelos, e o SARIMA, o N-HiTS e a LSTM diferem do ingênuo sazonal. O SARIMA não difere de nenhuma rede. Entre as redes, o N-HiTS difere da GRU, da Bi-LSTM e da Bi-GRU, e a LSTM difere da Bi-LSTM. Em 90 e 180 dias há só 6 e 3 blocos sem sobreposição, e o menor valor-p possível do Wilcoxon (0,031 e 0,25) não permite significância após a correção; a comparação é descritiva. Nesses horizontes, o Prophet erra menos que cada um dos outros modelos em 88% a 100% das origens de 90 dias e em todas as de 180 dias."));

// ================= PARTE 3 =================
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H1("Parte 3. Trechos do texto que mudam"));
c.push(NOTA("Para cada ponto: onde fica, o que o texto diz hoje (resumido, a partir das notas do projeto, porque o .docx da monografia não estava disponível) e a redação proposta. Confira a frase exata no seu texto antes de substituir."));

const pontos = [
  ["Descrição do SARIMA (capítulo 3 e seção 4.1)",
    "a ordem de cada praça é escolhida pelo auto_arima, em busca passo a passo pelo AIC.",
    "A ordem de cada praça é a de menor AIC entre as 144 combinações do espaço de busca (p e q até 3, P e Q até 2, d = 1 e D = 0), ajustadas na série de treino com as lacunas tratadas pelo filtro de Kalman (seção 4.X.1 e Tabela [X])."],
  ["Descrição do Prophet (capítulo 3 e seção 4.1)",
    "calendário de feriados nacionais e estaduais, com janela de dois dias antes e um depois.",
    "O calendário inclui os feriados nacionais e estaduais, o Carnaval e o Corpus Christi, com janela de efeito de dois dias antes e um depois. Os feriados que costumam gerar viagens longas têm janela própria: Carnaval e Natal, de quatro dias antes a dois depois; Sexta-feira Santa, de dois dias antes a três depois; e Confraternização Universal, de quatro dias antes a três depois (seção 4.X.2)."],
  ["Quadro de hiperparâmetros das redes (seção 4.1)",
    "paciência de 20 épocas e até 200 épocas.",
    "Parada antecipada com paciência de 20 épocas e teto de 200, exceto na LSTM de 180 dias, que usa paciência de 40 e teto de 400 (seção 4.X.3)."],
  ["Resultados do SARIMA (seção 4.3)",
    "o SARIMA perdeu para o ingênuo sazonal em algumas praças.",
    "O SARIMA supera o ingênuo sazonal nas seis praças e nos três horizontes, em SMAPE e em DTW, com SMAPE médio de 5,92, 6,49 e 6,46% contra 7,40, 8,72 e 8,91% do ingênuo."],
  ["Divergência entre SMAPE e DTW (seção 4.3.2)",
    "a divergência entre SMAPE e DTW do SARIMA em Boituva e Itu é o resultado contraintuitivo mais relevante do estudo.",
    "Na configuração original, o SARIMA tinha DTW maior que o do ingênuo sazonal em Boituva e Itu, porque a previsão perdia o ciclo semanal depois de poucas semanas. Com a ordem de menor AIC, essa divergência desaparece: o DTW do SARIMA fica abaixo do ingênuo em todas as praças e horizontes, com DTW relativo de 0,82, 0,78 e 0,76 nos três horizontes."],
  ["Classificação geral (seção 4.4)",
    "o Prophet é o melhor; a posição do SARIMA em relação às redes depende da ordem escolhida.",
    "O Prophet tem o menor erro nas cinco métricas, nas seis praças e nos três horizontes. O SARIMA fica em segundo no SMAPE nos três horizontes (5,92, 6,49 e 6,46%), à frente de todas as redes. Entre as redes, o N-HiTS é a melhor em 7 e 180 dias, e a GRU em 90 dias. Todos os modelos superam o ingênuo sazonal."],
  ["Testes de significância (seção 4.4.4)",
    "o Prophet difere de todos; LSTM, GRU, Bi-GRU e N-HiTS diferem do SARIMA em 7 dias.",
    "Usar o parágrafo que acompanha a tabela de Wilcoxon na Parte 2. O ponto que muda: o SARIMA passa a diferir do ingênuo sazonal e deixa de diferir das redes."],
  ["Faixas de antecedência (seções 4.4.3 e 4.5)",
    "nos dias 91 a 180, o N-HiTS (7,0), a Bi-GRU (7,2) e a GRU (7,4) erram menos que o SARIMA (7,7), e a Bi-LSTM empata.",
    "Nos dias 91 a 180, o SARIMA (6,53%) erra menos que todas as redes, de 6,96% no N-HiTS a 7,78% na LSTM; só o Prophet (4,53%) erra menos que ele. O erro das redes praticamente não cresce depois do dia 90, e a rede de 90 dias prevê a primeira semana melhor que a de 7 dias em quatro das cinco arquiteturas."],
  ["Semanas com feriado (seção 4.4)",
    "as semanas com feriado são 34% das janelas e concentram de 53 a 56% do erro; na semana seguinte, as redes erram de 4,9 a 5,5%, contra 9,6% do ingênuo.",
    "As semanas com feriado no período previsto são 34% das janelas de 7 dias e concentram de 53% a 58% do erro de todos os modelos. Na semana seguinte a um feriado, as redes erram de 4,9% a 5,5%, o SARIMA 4,6% e o Prophet 3,2%, contra 9,6% do ingênuo sazonal."],
  ["LSTM de 180 dias (seções 4.4 e 4.5)",
    "a LSTM de 180 dias errou 10,15%, porque três das cinco sementes travaram num platô; o resultado foi mantido.",
    "Na configuração original, três das cinco sementes da LSTM de 180 dias ficaram presas num platô no início do treino, e o SMAPE foi de 10,15%, acima do ingênuo sazonal. Com paciência de 40 épocas, as cinco sementes saem do platô e o SMAPE cai para 7,67%. A GRU continua mais estável que a LSTM: em 180 dias, erra 7,38%."],
  ["Resultados contraintuitivos (seção 4.5)",
    "o SARIMA perdeu para o ingênuo em algumas praças; o erro do Prophet não cresce com o horizonte; a LSTM de 180 dias falhou.",
    "O resultado contraintuitivo que permanece é que o erro do Prophet não cresce com o horizonte (4,43, 4,49 e 4,47%). Os outros dois resultados inesperados da primeira execução, o SARIMA perdendo para o ingênuo sazonal e a LSTM de 180 dias errando mais que ele, vinham da configuração e não se repetem com os ajustes da seção 4.X."],
  ["Prophet sem feriados (seção 4.4.5)",
    "sem feriados, o Prophet erra 5,45, 5,47 e 5,34% e continua o melhor; o calendário explica de 30 a 55% da distância para a melhor rede.",
    "Manter o parágrafo e acrescentar: \"As variantes sem feriados foram calculadas sobre a configuração original do Prophet. Com as janelas ajustadas dos feriados prolongados, o Prophet chega a 4,43% em 7 dias, contra 5,45% sem feriados, o que reforça o peso do calendário no horizonte curto.\""],
  ["Limitações das redes (seção 4.5.3)",
    "sem ajuste de hiperparâmetros, teto de 200 épocas, janela de 28 dias, sem retreino com a validação.",
    "As redes não passaram por ajuste sistemático de hiperparâmetros. A paciência de 20 épocas e o teto de 200 foram mantidos, exceto na LSTM de 180 dias, em que o platô de três sementes levou ao ajuste da seção 4.X.3. As janelas de entrada são fixas (28, 91 e 182 dias), e as redes não foram retreinadas com o período de validação."],
  ["Conclusão (seção 5)",
    "o calendário explica a vantagem do Prophet em 7 dias; em 90 e 180 dias ela persiste; a posição do SARIMA depende da ordem.",
    "Com a informação que cada modelo recebeu, o Prophet foi o melhor nos três horizontes, e o SARIMA ficou em segundo. Em 7 dias, a vantagem do Prophet vem principalmente do calendário de feriados; em 90 e 180 dias, ela persiste mesmo sem ele. As redes neurais superaram o ingênuo sazonal, mas não o SARIMA com a ordem de menor AIC. O trabalho futuro principal é oferecer o calendário de feriados a todos os modelos."],
];
pontos.forEach(([onde, atual, novo], i) => {
  c.push(new Paragraph({ children: [new TextRun({ text: `${i + 1}. ${onde}`, bold: true })], spacing: { before: 200, after: 40 }, keepNext: true }));
  c.push(ATUAL(atual));
  c.push(PROPOSTO(novo));
});

c.push(H2("O que ainda precisa ser refeito"));
c.push(NOTA("Itens que este arquivo não cobre, porque dependem de gráficos de previsão ou de tabelas que não são geradas pelo resultados_completos.py."));
[
  "Figura do SARIMA em Boituva (atual Figura 6) e figura com todos os modelos em Boituva: refazer com a nova ordem (figura_sarima_boituva.py e figura_boituva_todos_modelos.py), depois de copiar as ordens para config.ORDENS_ARIMA.",
  "Figura do Prophet em Boituva e dos componentes do Prophet: refazer com as janelas ajustadas.",
  "Figura do SMAPE por origem em 7 dias e figura de SMAPE por horizonte: as versões recalculadas estão na pasta figuras deste pacote (smape7_por_origem.png); a de horizontes vem do gerar_figuras.py.",
  "Tabela de sensibilidade (atual Tabela 11): as variantes continuam valendo, mas comparam com a configuração original; acrescentar uma nota.",
  "Tabela das sementes das redes: na LSTM de 180 dias, o SMAPE por semente passa a ser 8,07, 7,11, 7,98, 7,70 e 7,50%.",
  "Testes de mesa das semanas de Boituva (teste_mesa_semanas_boituva.py): refazer para o SARIMA e o Prophet.",
  "Resumo e Abstract: usar os números da configuração ajustada.",
].forEach((t) => c.push(ITEM(t)));

const doc = new Document({
  creator: "Cyro Oikawa",
  title: "Resultados com a configuração ajustada",
  styles: {
    default: { document: { run: { font: FONTE, size: 24 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONTE, size: 36, bold: true }, paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONTE, size: 28, bold: true }, paragraph: { spacing: { before: 240, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONTE, size: 24, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONTE, size: 24, bold: true, italics: true }, paragraph: { spacing: { before: 200, after: 120 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "marcas", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1701, left: 1701, bottom: 1134, right: 1134 } } },
    children: c,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(SAIDA, b); console.log("ok", SAIDA); });
