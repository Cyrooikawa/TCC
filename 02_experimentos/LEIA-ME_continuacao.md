# Continuação de 08/11/2026: análises A1 a A4

Este pacote traz só os arquivos novos ou alterados. Copie a pasta `02_codigo_e_resultados/` por
cima da pasta de mesmo nome do projeto. Nenhum resultado oficial dos oito modelos mudou: os
arquivos `*_metricas_por_origem.csv`, `config.py` e as Tabelas 4, 7, 8, 9 e 10 continuam iguais.

## Scripts novos

| Script | Pendência | O que faz | Tempo |
|---|---|---|---|
| `faixas_antecedencia_direto.py` | A1 | SMAPE por faixa (dias 1 a 7, 8 a 90, 91 a 180). Nas redes, calcula direto da previsão diária da rede de 180 dias; nos estatísticos, usa as Equações 7 e 8, que valem para eles. Refaz a Figura 11 e confere os cálculos | segundos |
| `testes_blocos_por_origem.py` | A3 | Friedman e Wilcoxon-Holm com a média das seis praças por origem como bloco (87, 6 e 3 blocos), mais medidas descritivas e a correlação entre praças | segundos |
| `prophet_sem_feriados.py` | A4 | Prophet igual ao principal, mas com `holidays=None`. Com `--sem-anual`, tira também a sazonalidade anual | 2 a 20 min |
| `sarima_busca_completa.py` | A2 | Grade completa das 144 ordens, nos critérios do auto_arima e do ajuste final, e avaliação de três variantes. **Diagnóstico**: não muda o SARIMA oficial | 15 a 60 min |

## Script alterado

- `resultados_completos.py`:
  - a função `faixas` usa o cálculo direto para as redes, então a Figura 11 sai corrigida;
  - a Tabela 11 inclui o Prophet sem feriados e o Prophet sem feriados e sem sazonalidade anual.

## Resultados novos ou alterados

| Arquivo | Conteúdo |
|---|---|
| `resultados/completos_faixas_antecedencia.csv` | Faixas corrigidas (substitui o anterior) |
| `figuras/faixas_antecedencia.png` | Figura 11 corrigida (substitui a anterior) |
| `resultados/faixas_redes_por_horizonte_treinado.csv` | Mesmos dias previstos pelas redes de 7, 90 e 180 dias |
| `resultados/completos_tabela11_sensibilidade.csv` | Com as duas variantes do Prophet |
| `resultados/testes_blocos_por_origem_pares.csv` | 28 pares × 3 horizontes, com a comparação com a Tabela 10 |
| `resultados/testes_blocos_por_origem_friedman.csv` | Friedman com 87, 6 e 3 blocos |
| `resultados/testes_blocos_por_origem_dependencia.csv` | Correlação entre praças por par (7 dias) |
| `resultados/sensibilidade_prophet_sem_feriados*.csv` e `.npz` | Métricas por origem e previsões diárias das duas variantes |
| `diagnostico/sarima_*` | Grades A e B, ordens, métricas por origem e figuras das três variantes do SARIMA |
| `diagnostico/LEIA-ME_diagnostico.md` | Com a atualização de 08/11 |

## Como rodar no seu computador (Windows, na pasta 02_codigo_e_resultados)

```
python validar_equivalencia_pytorch.py
python validar_procedimentos.py
python faixas_antecedencia_direto.py
python testes_blocos_por_origem.py
python prophet_sem_feriados.py
python prophet_sem_feriados.py --sem-anual
python sarima_busca_completa.py --processos 4
python resultados_completos.py
```

O que esperar de cada um:
- `validar_equivalencia_pytorch.py` deve terminar com "AS REDES EM NUMPY SÃO EQUIVALENTES ÀS DO PYTORCH". Precisa de `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- `validar_procedimentos.py` deve terminar com "TODAS AS VERIFICAÇÕES PASSARAM" (71 verificações).
- `faixas_antecedencia_direto.py` deve terminar com "TODAS AS CONFERÊNCIAS PASSARAM".
- `sarima_busca_completa.py` reaproveita as grades salvas em `diagnostico/`. Use `--refazer` para refazê-las no seu computador.

## Execução de referência (ambiente de nuvem, 08/11/2026)

- **Ambiente:** Linux, 4 núcleos, Python 3.12, com as versões do `versoes_bibliotecas.txt` (NumPy 2.4.6, pandas 2.3.3, SciPy 1.17.1, statsmodels 0.15.0, pmdarima 2.1.1, Prophet 1.4.0, holidays 0.104, matplotlib 3.11.2, PyTorch 2.14.0 CPU).
- **Validadores:**
  - `validar_equivalencia_pytorch.py`: 38 de 38 comparações, maior diferença 6,4 × 10⁻¹⁵.
  - `validar_procedimentos.py`: 71 de 71.
- **Reprodução dos estatísticos nesta máquina:**
  - ingênuo e análises de sensibilidade idênticos;
  - SARIMA e Prophet com diferença de no máximo 0,01 ponto percentual na média das praças.
