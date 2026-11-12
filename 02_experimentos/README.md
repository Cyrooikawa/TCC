# Previsão de Fluxo Veicular em Rodovias da Região de Sorocaba

Projeto de Final de Curso (UNESP Sorocaba). Análise comparativa entre modelos
estatísticos (ARIMA, Prophet) e arquiteturas de aprendizado profundo (LSTM, GRU,
Bi-LSTM, Bi-GRU, N-HiTS) para previsão de volume diário em praças de pedágio da
Região Metropolitana de Sorocaba (RMS), com dados abertos da ARTESP.

**Autor:** Cyro Hideyoshi Shimizu Oikawa
**Orientador:** Prof. Dr. Leopoldo André Dutra Lusquino Filho

## Recorte das praças

| Grupo | Praças | Por quê |
|---|---|---|
| Núcleo (comparação principal) | Alambari, Boituva, Bloqueio de Boituva, Porto Feliz | Bidirecionais, série contínua desde 2020, escala compatível |
| Unidirecional (separado) | Sorocaba, Itu (praças físicas) | O pedágio só cobra um sentido: medem metade do fluxo; séries contínuas 2020-2026 |
| Fora da modelagem | Pórticos antigos de Itu/Salto | Escala incompatível (74 a 156 veículos/dia) |
| Fora da modelagem | Alumínio, Araçoiaba Leste/Oeste, São Roque (praças físicas) | Desativadas em 30/09/2025 e substituídas por pórticos |
| Fora da modelagem | Pórticos free-flow do lote L33 | Existem só a partir de out/2025: série curta |
| Fora do estudo | Quadra | Não pertence à RMS |

Todas as decisões ficam centralizadas em `config.py`.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `organizar_por_ano.py` | Copia os CSVs baixados da ARTESP (pastas por ano) para `dados/brutos/`, separando núcleo e estudo de caso |
| `extracao.py` | Junta os arquivos de cada lote; descarta linhas de outro ano dentro do arquivo anual |
| `tratamento.py` | Padroniza praça/município, filtra a RMS (27 municípios), soma volume por categoria (leve/moto/pesado), marca direcionalidade |
| `preparar_series.py` | Gera a série diária de cada praça modelada, marcando dias ausentes e invalidados |
| `config.py` | Decisões do experimento: praças, períodos invalidados, 70% treino, horizontes 7/90/180 dias |
| `metricas.py` | MAE, RMSE, MAPE, SMAPE e DTW, usadas por todos os modelos |
| `modelo_ingenuo.py` | Referência mínima: repete a última semana observada. Todo modelo precisa errar menos que isto |
| `arima_baseline.py` | ARIMA (baseline) com avaliação nos três horizontes e teste de Ljung-Box nos resíduos |
| `prophet_modelo.py` | Prophet: tendência + sazonalidade semanal e anual + feriados de SP, Carnaval e Corpus Christi |
| `analises_sensibilidade.py` | Duas análises citadas no capítulo 4: ARIMA sem componente sazonal e Prophet com ajuste único no treino |
| `comparar_modelos.py` | Junta os resultados de todos os modelos numa tabela e conta em quantas datas cada um vence o ingênuo (pelo SMAPE) |
| `ver_metricas.py` | Mostra na tela qualquer métrica por modelo, praça e horizonte, e grava tabela para o Excel |
| `gerar_figuras.py` | Gera as figuras da monografia (pasta `figuras/`) e os números da análise exploratória (`resultados/eda_*.csv`) |
| `diagnostico_ordem_sarima.py` | Diagnóstico da escolha da ordem do SARIMA (ver `diagnostico/LEIA-ME_diagnostico.md`) |
| `versoes_bibliotecas.txt` | Versões exatas das bibliotecas da execução oficial (reprodutibilidade) |
| `visualizacao_tratamento.ipynb` | Gráficos das séries e dos resultados |
| `main.ipynb` | Legado da primeira versão do pipeline; não usar como fonte |

## Como executar (Windows, a partir da pasta do projeto)

Instalar dependências (uma vez só):
```
pip install -r requirements.txt
```

Organizar os arquivos brutos (só na primeira vez ou ao baixar arquivo novo):
```
python organizar_por_ano.py "C:\caminho\da\pasta\com\os\anos"
```

Rodar o pipeline, sempre nesta ordem:
```
python extracao.py
python tratamento.py
python preparar_series.py
python modelo_ingenuo.py
python arima_baseline.py
python prophet_modelo.py
python analises_sensibilidade.py
python comparar_modelos.py
python gerar_figuras.py
python ver_metricas.py
```

Tempo aproximado: ingênuo, segundos; ARIMA, cerca de 1 minuto; Prophet, 6 a 10
minutos (é reajustado em cada data de partida do teste).

Os resultados ficam em `resultados/`; a tabela principal é `comparacao_modelos.csv`.
As ordens do ARIMA estão fixadas em `config.py` (ORDENS_ARIMA) para que qualquer
computador reproduza o mesmo modelo. Para ver os gráficos, abra
`visualizacao_tratamento.ipynb` no Jupyter ou VS Code e rode todas as células.
