# Previsão de Tráfego Rodoviário (Região de Sorocaba) com Machine Learning

## Sobre o Projeto
Este repositório contém o código fonte e os dados processados do Projeto de Final de Curso (TCC) focado na análise e predição de fluxo veicular em rodovias da Região Metropolitana de Sorocaba (RMS). 

O objetivo principal desta pesquisa é realizar uma análise comparativa de desempenho entre modelos estatísticos clássicos e arquiteturas avançadas de *Deep Learning* para previsão de séries temporais em diferentes horizontes.

**Autor:** Cyro Hideyoshi Shimizu Oikawa  
**Orientador:** Prof. Dr. Leopoldo André Dutra Lusquino Filho  

## Metodologia e Tecnologias
O pipeline de dados foi desenvolvido utilizando **Python** e envolve as seguintes etapas:
1. **ETL e Pré-processamento:** Extração de dados abertos da ARTESP, unificação de arquivos, tratamento de valores nulos e filtragem de praças de pedágio com histórico inferior a 12 meses para evitar quebras estruturais.
2. **Análise Exploratória de Dados (EDA):** Identificação de sazonalidade anual, tendências de crescimento e anomalias históricas (ex: impacto de bloqueios pandêmicos em 2020).
3. **Modelagem (Em andamento):** * **Estatística Clássica:** ARIMA / Prophet.
   * **Redes Neurais Recorrentes:** LSTM, GRU, Bi-LSTM.

## Estrutura do Repositório
* `main.ipynb`: Notebook principal contendo todo o fluxo de Análise Exploratória (EDA), visualização de sazonalidade e p    reparação dos modelos preditivos.
* `base_RMS_final.csv`: Base de dados unificada, limpa e padronizada. Este é o arquivo fonte utilizado para alimentar as redes neurais. *(Nota: Os dados brutos originais da ARTESP foram omitidos do versionamento devido ao seu volume).*
* `requirements.txt`: Lista de dependências e bibliotecas Python utilizadas no projeto.

## 🚀 Como Executar
[]
1. Clone este repositório para a sua máquina local:
```bash
   git clone [https://github.com/Cyrooikawa/TCC.git](https://github.com/Cyrooikawa/TCC.git)