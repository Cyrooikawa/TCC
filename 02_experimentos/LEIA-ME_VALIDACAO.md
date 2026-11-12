# Como validar os resultados do capítulo 4 no seu computador

Objetivo: rodar os modelos no seu computador e confirmar que os números batem com os do
capítulo 4. Não é preciso saber programar: o caminho rápido é dar dois cliques num arquivo.

## O que você precisa

1. **Windows** com uns 3 GB livres.
2. **Python 3.12**, de python.org. Na primeira tela do instalador, marque
   **"Add python.exe to PATH"**.
3. **Esta pasta** (`02_experimentos`), que vem no pacote `Pacote_validacao_TCC.zip` ou no GitHub,
   branch `claude/model-performance-testing-voz6nr`.
4. **O arquivo de dados** `dados\processados\series_diarias.csv`. Ele já vem no pacote .zip.
   Se você baixou do GitHub, copie-o da sua pasta `02_codigo_e_resultados\dados\processados\`
   (o GitHub não guarda arquivos .csv deste projeto).

Use uma pasta **separada** da sua `02_codigo_e_resultados`. Assim os resultados oficiais
originais não são sobrescritos.

## Caminho rápido (dois cliques)

1. Descompacte o pacote, por exemplo em `C:\TCC_validacao\`. Evite pastas com acentos no nome.
2. Dê dois cliques em **`VALIDAR_RESULTADOS.bat`**.
3. Na primeira vez, ele instala as bibliotecas com as mesmas versões da execução oficial
   (alguns minutos).
4. Ele roda o ingênuo, o SARIMA e o Prophet (cerca de 10 minutos) e pergunta sobre as redes:
   - **1** pula as redes;
   - **2** roda só a LSTM de 180 dias, a única rede que mudou (cerca de 25 minutos);
   - **3** roda as 15 redes (de 2 a 4 horas, conforme o computador).
5. No fim, aparece uma tabela como esta:

```
Modelo           Dias   Obtido  Esperado   Dif.        Pior praça (dif.)   Resultado
Ingênuo sazonal     7     7.40      7.40   0.00       Porto Feliz (0.00)   OK
SARIMA              7     5.90      5.90   0.00               Itu (0.00)   OK
Prophet             7     4.43      4.43   0.00          Sorocaba (0.00)   OK
...
Resumo: 24 OK, 0 ATENÇÃO, 0 DIFERENTE, 0 FALTA
Tudo confere com os resultados do capítulo 4.
```

Tudo o que aparece na tela também fica em `log_validacao.txt`.

## Como ler o resultado

| Resultado | O que quer dizer | O que fazer |
|---|---|---|
| **OK** | O número bate com o do capítulo, dentro da tolerância | Nada |
| **ATENÇÃO** | Diferença pequena, em geral por outra versão de biblioteca ou outro processador | As conclusões não mudam; use nos textos os números da sua execução |
| **DIFERENTE** | Diferença maior que o esperado | Confira as versões das bibliotecas e a configuração em uso; me mande o `log_validacao.txt` |
| **FALTA** | O modelo ainda não foi rodado | Normal se você pulou as redes |

Tolerâncias (pontos percentuais de SMAPE, média das praças | pior praça): ingênuo 0,01 | 0,01;
Prophet 0,05 | 0,10; SARIMA 0,10 | 0,20; redes 0,30 | 0,60.

**Variação conhecida:** o SARIMA ajustado tem a raiz sazonal quase igual a 1, e o otimizador para
em pontos um pouco diferentes conforme o computador. Em Itu, o erro de 7 dias pode sair 5,60% ou
5,73%. Isso pode aparecer como ATENÇÃO numa praça, sem mudar a ordem dos modelos.

## Caminho manual (pelo terminal)

Abra o **Prompt de Comando** na pasta (digite `cmd` na barra de endereço do Explorador) e rode:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install numpy==2.4.6 pandas==2.3.3 scipy==1.17.1 matplotlib==3.11.2 statsmodels==0.15.0 pmdarima==2.1.1 prophet==1.4.0 holidays==0.104

python modelo_ingenuo.py              (segundos)
python arima_baseline.py              (cerca de 1 minuto; também gera a Tabela 6 em resultados\arima_modelos.csv)
python prophet_modelo.py              (4 a 10 minutos)
python redes_neurais.py LSTM 180      (cerca de 25 minutos; ou --processos 2 para as 15 redes)
python conferir_resultados.py         (compara com os esperados)
```

Depois de rodar **todas** as redes, gere as tabelas e figuras do capítulo:

```
python analises_sensibilidade.py      Tabela 13 (ARIMA sem sazonalidade, Prophet ajuste único)
python prophet_sem_feriados.py        Tabela 13 (Prophet sem feriados)
python prophet_sem_feriados.py --sem-anual   Tabela 13 (sem feriados e sem sazonalidade anual)
python faixas_antecedencia_direto.py  Tabela 10 e Figura 10
python resultados_completos.py        Tabelas 4, 7, 8, 9 e 14 e Figuras 5, 8, 9 e 11
python testes_blocos_por_origem.py    Tabela 11 e testes da Tabela 12
python gerar_figuras.py               Figuras 2, 3, 4 e 7
python figura_boituva_todos_modelos.py   Figura 6
python validar_procedimentos.py       verificações automáticas do procedimento
```

A ordem importa: as análises de sensibilidade vêm antes de `resultados_completos.py`, que as lê.

As tabelas ficam em `resultados\completos_*.csv` e `resultados\testes_blocos_*.csv`, e as figuras,
em `figuras\`.

## Configuração ajustada x original

A pasta roda, por padrão, a **configuração ajustada** (a do capítulo 4). As diferenças estão
todas em `config.py`:

| | Original (primeira execução) | Ajustada (capítulo 4) |
|---|---|---|
| SARIMA | Ordem da busca passo a passo do auto_arima | Ordem de menor AIC entre 144 (`ORDENS_ARIMA_AJUSTADA`) |
| Prophet | Todos os feriados: 2 dias antes, 1 depois | Janela própria para Carnaval, Sexta-feira Santa, Natal e Ano-Novo (`JANELAS_FERIADOS_LONGOS`) |
| LSTM 180 dias | Paciência 20, até 200 épocas | Paciência 40, até 400 épocas (`PARADA_ESPECIFICA`) |

Para reproduzir a **original**, numa cópia separada da pasta, rode antes:

```
set TCC_CONFIGURACAO=original
```

e depois os mesmos comandos. O `conferir_resultados.py` passa a comparar com
`esperado_configuracao_original.json`.

## Problemas comuns

- **"python não é reconhecido"**: reinstale o Python marcando "Add python.exe to PATH", ou use
  `py -3.12` no lugar de `python`.
- **Erro ao instalar o prophet ou o pmdarima**: confira se o Python é o 3.12 (`py -3.12 --version`).
  Versões mais novas do Python podem não ter essas bibliotecas prontas para Windows.
- **Falta series_diarias.csv**: copie de `02_codigo_e_resultados\dados\processados\`.
- **Números diferentes em todos os modelos**: provavelmente a configuração em uso não é a que você
  queria. A primeira linha do `conferir_resultados.py` mostra qual está ativa.
