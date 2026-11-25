# Texto para os slides finais

## Método e treino da CNN

- CNN 2D de três blocos sobre espectrogramas de 92 × 86. Três seeds de treino.
- Adam com taxa de aprendizado **0,001** e dropout **0,2**, ambos fixos. **Sem aumento de dados e sem busca de hiperparâmetros.**
- O checkpoint foi escolhido pela ROC-AUC e o limiar pelo F1 na validação A+B. As três execuções atingiram o teto de **14 épocas**.

**Nota de apresentação.** O Checkpoint 1 previa deslocamento temporal, mistura de ruído e ajuste de hiperparâmetros. A versão avaliada usou a configuração fixa acima. As melhores épocas foram 14, 14 e 13. A parada antecipada de quatro épocas sem melhora ficou inativa, e a convergência permanece em aberto. C e o contínuo B* foram avaliados com os modelos congelados. [Histórico de treino](resultados/treino_cnn.csv) · [Configuração congelada](resultados/cnn_congelada.json)

## Desempenho em clipes

| Partição | Métrica | Baseline MLP | CNN |
|---|---|---:|---:|
| A+B validação | ROC-AUC | 0,925 | 0,955 |
| A+B validação | F1 | 0,848 | 0,886 |
| A validação | Recall | **74,4%** | 66,7% |
| A validação | Precisão | 86,8% | **94,9%** |
| C teste | ROC-AUC | 0,834 | 0,925 |
| C teste | F1 | 0,756 | 0,855 |

**Nota de apresentação.** Em A, a CNN detectou uma fração menor dos 56 upcalls, com maior proporção de alarmes corretos. Isso foi medido no ponto de operação escolhido para A+B. [Resultados por conjunto](resultados/comparacao_clipes.csv)

## Risco de generalização em C

| Coleção | Com upcall | Sem upcall | ROC-AUC da CNN |
|---|---:|---:|---:|
| NOPP6_EST | 1.500 | 775 | 0,906 |
| SBNMS | 0 | 557 | — |
| GSC | 0 | 168 | — |

**Limitação.** Todos os positivos de C vêm de NOPP6_EST. O resultado global pode refletir diferenças entre coleções. Dentro da coleção que contém as duas classes, a CNN obteve ROC-AUC de **0,906**, e a baseline, **0,852**. Nas outras duas coleções, a taxa de falsos positivos é a métrica aplicável.

**Nota de apresentação.** As 148 repetições exatas de PCM fazem parte da avaliação oficial dos 3.000 clipes. A vantagem da CNN permaneceu na análise de sensibilidade com 2.852 sinais distintos. [Métricas por coleção](resultados/metricas_por_colecao_c.csv) · [Análise de sensibilidade](resultados/sensibilidade_duplicatas_c.csv)

## Áudio contínuo e custo dos alarmes

- **Teste original B*.** 25 h, 1.058 de 1.157 upcalls detectados, **91,4% de recall**, **239,3 falsos alarmes/h** e precisão de **15,0%**.
- **Limiar original.** O valor de 0,402 foi escolhido pelo F1 nos clipes A+B. Em B*, o detector percorreu todas as **179.750 janelas**, com base de ruído e frequência de chamados diferentes das dos clipes.
- **Teste externo NOAA.** Calibração em 8 h de Nantucket e teste em 16 h de Hatteras e Brunswick. A CNN recalibrada atingiu **88,4% de recall e 5,19 alarmes/h**. Com média móvel de cinco scores, atingiu **86,0% e 3,19/h**.

**Nota de apresentação.** Os 239,3 alarmes/h são o ponto de operação original em B*. O resultado NOAA mede outro acervo com limiar calibrado no contínuo. Na análise exploratória de B* com o mesmo limiar dos clipes, a média móvel reduziu os alarmes para 63,6/h e o recall para 84,2%. [Teste original B*](resultados/continuo_bstar.json) · [Curva exploratória B*](figuras/recall_por_falsos_alarmes_bstar.png) · [Teste externo NOAA](resultados/noaa_teste_independente.csv)
