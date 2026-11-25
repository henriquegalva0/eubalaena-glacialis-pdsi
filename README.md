# Eubalaenas — Detecção de upcalls da baleia-franca-do-Atlântico-Norte

Seminário da disciplina de **Processamento Digital de Sinais e Imagens** (UFG).

O projeto trata gravações de hidrofone como sinais digitais, converte cada trecho em espectrograma e treina uma rede neural convolucional para separar trechos com **upcall** da baleia-franca-do-Atlântico-Norte (*Eubalaena glacialis*) de trechos com apenas ruído de fundo.

## Equipe

| Matrícula | Nome | GitHub |
|-----------|------|--------|
| 202503995 | Cibelle Pires Botelho | [cibellepires](https://github.com/cibellepires) |
| 20254005 | Gustavo Pires Fontana | [espectrx](https://github.com/espectrx) |
| 20254006 | Henrique Galvão da Silva | [henriquegalva0](https://github.com/henriquegalva0) |
| 202504010 | João Guilherme Pires de Andrade | [jg_pandrade](https://github.com/jg_pandrade) |
| 202504014 | Júlia Jaime Biancolini | [jbiancolini](https://github.com/jbiancolini) |

## Problema

A baleia-franca-do-Atlântico-Norte é uma espécie criticamente ameaçada. A monitorização acústica passiva permite acompanhar sua presença 24 h por dia, mas gera mais áudio do que se consegue ouvir manualmente. O upcall é um som curto (~1 s) e de banda estreita, com frequência crescente em torno de 50–250 Hz, imerso em ruído de baixa frequência. O trabalho busca detectá-lo automaticamente.

**Tarefa:** classificação binária de clipes de 3 s.

- `1`: há upcall;
- `0`: não há upcall (ruído de fundo, navios e outros sons).

Isto é detecção de um som específico, não classificação geral de animais marinhos.

## Dados

Fonte: Kirsebom et al. (2020). A, B e B* vêm do Golfo de São Lourenço, Canadá. C vem do Golfo do Maine, EUA. DOI: <https://doi.org/10.20383/101.0241>

Baixe o `narw-data.zip` (~4,63 GB) pelo DOI e extraia o conteúdo para uma pasta chamada `narw-data`. Passe o caminho dessa pasta nos comandos abaixo. Os dados ficam fora do Git.

| Conjunto | Partição | Clipes | Com upcall |
|----------|----------|-------:|-----------:|
| A | treino | 1.767 | 744 |
| A | validação | 306 | 56 |
| B | treino | 3.309 | 2.033 |
| B | validação | 579 | 341 |
| C | teste neste projeto | 3.000 | 1.500 |

Contagens obtidas das planilhas de anotação. Os conjuntos A, B e C diferem em local, período e instalação do hidrofone.

O conjunto contínuo B* contém **50 gravações de aproximadamente 30 min**, com 1.157 upcalls anotados. Os WAVs contínuos originais estão a 32.768 Hz, e os clipes estão a 1.000 Hz. A pasta física de C se chama `train` no pacote original.

### Estrutura

```
narw-data/
├── data/
│   ├── clips/
│   │   ├── dataset_A/{train,val}/   annotations_A_*.csv + audio/
│   │   ├── dataset_B/{train,val}/   annotations_B_*.csv + audio/
│   │   └── dataset_C/train/         annotations_C_train.csv + audio/
│   └── continuous/dataset_B/        annotations_B_cont.csv + audio/
└── figures/specs/                   espectrogramas PNG (spec_X_Y.png)
```

- Áudio: `.wav`, 1.000 amostras/s.
- Anotações (`.csv`, separador `;`):
  - clipes: `clip_name`, `label`, `snr` (dB). O conjunto C inclui também `source_filename`, `source_start` e `source_end`;
  - contínuos: `filename`, `timestamp` (s, centro do upcall), `snr`.

Descrição completa em [dados.md](dados.md).

## Execução

```
Áudio (.wav) → Pré-processamento → Espectrograma (STFT) → Modelo → upcall / ruído
              (normalização,                              (baseline e CNN)
               filtro passa-faixa)
```

Em PowerShell, a partir da raiz do clone, com Python 3.12 e ambiente virtual:

```powershell
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt
$dataset = 'C:\caminho\para\narw-data'
& $python -m src.prepare --data $dataset
& $python -m src.diagnostics eda
& $python -m src.baseline train
& $python -m src.cnn train
& $python -m src.evaluate
```

O preparo usa janelas Hamming de 256 amostras, passo de 32, filtro Butterworth de 40–400 Hz, normalização pelo pico e subtração da mediana temporal por frequência no espectrograma em dB. Alguns WAVs oficiais incluem uma amostra no instante 3,000 s. O código mantém as primeiras 3.000 amostras.

A baseline é uma MLP com camadas internas de 64 e 32 unidades. As 129 features são a média, o desvio-padrão e o percentil 90 de 32 bandas log-mel e 10 MFCCs, mais inclinação, R² e diferença entre as medianas inicial e final do contorno de frequência dominante. A CNN usa três blocos convolucionais, média global e uma saída binária. O treino usa as seeds 7, 23 e 42. A CNN seleciona a época por ROC-AUC em A+B de validação. Cada modelo e seed obtém seu limiar pelo maior F1 na mesma validação.

### Configuração efetivamente avaliada

A CNN foi treinada com Adam, taxa de aprendizado fixa de **0,001** e dropout de **0,2**. Deslocamento temporal, mistura de ruído e busca de hiperparâmetros ficaram no planejamento. A validação A+B foi usada para selecionar checkpoints e limiares.

As três seeds atingiram o limite de **14 épocas**. As melhores épocas foram **14, 14 e 13** para as seeds 7, 23 e 42. A parada antecipada de quatro épocas sem melhora ficou inativa, então o histórico de validação ainda deixa em aberto se mais épocas ajudariam. [Histórico de treino](docs/resultados/treino_cnn.csv) · [Checkpoints e limiares selecionados](docs/resultados/cnn_congelada.json)

Depois de concluir os ajustes e congelar os checkpoints e limiares, execute os testes finais:

```powershell
& $python -m src.prepare --data $dataset --split test
& $python -m src.baseline test
& $python -m src.cnn test
& $python -m src.evaluate --split val test
& $python -m src.audit --data $dataset
& $python -m src.diagnostics eda --split train val test
& $python -m src.diagnostics errors --split test
& $python -m src.continuous --data $dataset
```

C fica reservado para testar a generalização. No contínuo, o código seleciona a seed da CNN de maior ROC-AUC em A+B de validação e aplica o limiar salvo. Reamostra o áudio para 1 kHz, pontua janelas de 3 s a cada 0,5 s e agrupa janelas consecutivas acima do limiar. Cada evento pode corresponder a uma anotação, com tolerância de 1 s nas bordas do intervalo agrupado. O horário estimado é o centro da janela de maior score no evento. Os falsos alarmes por hora usam a duração efetivamente processada.

Os resultados principais em **C e B*** correspondem à CNN congelada. As ablações posteriores ficaram na validação A+B. A curva de B* reutiliza os scores congelados como análise exploratória. A calibração de pós-processamento usou gravações da NOAA separadas por local entre calibração e teste.

### Resultados principais

- **C.** A CNN obteve ROC-AUC média de 0,925 e F1 de 0,855. Os **1.500 positivos** pertencem à coleção **NOPP6_EST**, que inclui também 775 negativos. SBNMS e GSC acrescentam 725 negativos. Dentro de NOPP6_EST, a ROC-AUC média foi de 0,906 para a CNN e 0,852 para a baseline. A leitura por coleção explicita a composição do teste. [Métricas por coleção em C](docs/resultados/metricas_por_colecao_c.csv) · [Sensibilidade a repetições exatas](docs/resultados/sensibilidade_duplicatas_c.csv)
- **Validação A.** No limiar escolhido em A+B, a CNN teve recall de **66,7%** e precisão de **94,9%**. A baseline teve recall de **74,4%** e precisão de **86,8%**. São médias de três seeds. [Comparação por conjunto](docs/resultados/comparacao_clipes.csv)
- **Contínuo B*.** O limiar escolhido por F1 nos clipes encontrou **1.058 de 1.157** chamadas e produziu **5.982 falsos alarmes** em 25 h, ou **239,3/h**. O limiar otimiza uma métrica por clipe em um conjunto de candidatos. Em gravações contínuas, ele é aplicado a todas as janelas, com muito mais oportunidades para alarmes. A calibração em gravações anotadas da NOAA reduziu a taxa da CNN com média móvel para **3,19/h**, com recall de **86,0%** nas 16 h de teste externo. [Resultado original em B*](docs/resultados/continuo_bstar.json) · [Teste externo por local](docs/resultados/noaa_teste_independente.csv)

### Saídas locais

- `artifacts/prepared/` guarda arrays NumPy e os metadados por partição. Use `--output` no preparo e `--prepared` nos demais comandos para armazenar os arrays em outra pasta.
- `artifacts/baseline/` e `artifacts/cnn/` guardam os checkpoints, os limiares e os scores por clipe. A CNN também salva o histórico de treino.
- `artifacts/avaliacao/` guarda as curvas ROC e precisão–recall, métricas por seed e faixa de SNR. Para interpretar C, o arquivo `metricas_por_colecao_c.csv` consolida as origens de gravação. `metricas_por_origem_c.csv` mantém o detalhe por WAV para auditoria.
- `artifacts/diagnostico/` guarda contagens, distribuições de SNR, espectrogramas médios e exemplos de erros.
- `artifacts/continuo/` guarda os scores das janelas, os eventos com horário e a avaliação por arquivo e consolidada.

## Testes complementares

Os experimentos foram salvos em `artifacts/novos_testes/`. Os checkpoints e as avaliações originais continuam sendo a referência. Foram acrescentadas verificações automatizadas de antialiasing, STFT, silêncio, correlação, suavização, ausência de chamadas, chamadas próximas, pareamento um a um e tratamento de anotações incertas.

### Detector clássico por correlação 2D

O detector percorre o espectrograma com um molde de rampa de aproximadamente 1 s e duração e inclinação fixas. O score do clipe é a maior correlação normalizada entre os deslocamentos avaliados em tempo e frequência. O limiar foi escolhido pelo F1 na validação A+B. A curva no contínuo compara também o molde à CNN.

| Conjunto | ROC-AUC do molde | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| Validação A+B | 0,966 | 85,5% | 93,7% | 0,894 |
| C | 0,868 | 96,5% | 61,5% | 0,752 |

O ganho em A+B deve ser interpretado considerando que os candidatos de A e B foram pré-selecionados por um detector tempo–frequência semelhante. Em C, a CNN teve ROC-AUC de 0,925 e F1 de 0,855, médias de três seeds. [Kirsebom et al., seção II.B](https://oliskir.github.io/assets/pdf/NARW-JASA2020.pdf)

### Pós-processamento e localização no contínuo

Comparamos o agrupamento original, a média móvel de cinco scores aplicada antes do limiar e a exigência de duas janelas consecutivas. Medimos recall, falsos alarmes por hora e erro absoluto do horário de pico entre os eventos pareados. Para B*, a curva é exploratória e reutiliza os scores da CNN congelada. Cada ponto por orçamento reporta o maior recall observado na grade de limiares de 0,01 a 0,99, incluindo o limiar original.

| Variante em B* | Recall no limiar dos clipes | Falsos alarmes/h no limiar dos clipes | Recall com teto de 25 alarmes/h | Alarmes/h nesse ponto |
|---|---:|---:|---:|---:|
| CNN original | 91,4% | 239,3 | 83,6% | 22,1 |
| CNN com média de cinco scores | 84,2% | 63,6 | 81,6% | 22,3 |
| CNN com duas janelas consecutivas | 88,9% | 103,9 | 84,4% | 24,0 |
| Molde 2D | 82,4% | 12,8 | 85,3% | 18,0 |

No limiar vindo dos clipes, a mediana do erro absoluto de horário da CNN foi de **0,61 s** com os scores originais e **0,17 s** com a média móvel, entre os eventos pareados de cada variante. A média móvel também reduziu o recall de 91,4% para 84,2%, de modo que os conjuntos de eventos pareados são diferentes. [Curva exploratória de B*](docs/figuras/recall_por_falsos_alarmes_bstar.png)

A avaliação externa usa 24 h do acervo NOAA NEFSC. São oito horas de cada local, em quatro blocos locais de duas horas que começam às 00h, 06h, 12h e 18h. As 8 h de **Nantucket** calibram um limiar por variante e selecionam a de maior recall com até **25 falsos alarmes por hora**. As 16 h de **Hatteras e Brunswick** avaliam os limiares congelados. Cada centro de upcall é o ponto médio da anotação temporal. Chamadas confirmadas entram no recall. Eventos sem correspondência confirmada que coincidem com uma chamada possível são contabilizados à parte. [Acervo anotado da NOAA NEFSC](https://doi.org/10.25921/2c09-ng58)

| Variante no teste NOAA (129 upcalls confirmados) | Limiar | Detectados | Recall | Precisão | Falsos alarmes/h |
|---|---:|---:|---:|---:|---:|
| CNN com limiar dos clipes A+B | 0,4023 | 121 | 93,8% | 4,8% | 151,25 |
| CNN original calibrada em Nantucket | 0,59 | 114 | 88,4% | 57,9% | 5,19 |
| CNN com média de cinco scores | 0,51 | 111 | 86,0% | 68,5% | 3,19 |
| CNN com duas janelas consecutivas | 0,54 | 113 | 87,6% | 60,8% | 4,56 |
| Molde 2D selecionado em Nantucket | 0,25 | 119 | 92,2% | 23,2% | 24,63 |

Nantucket selecionou o molde com **88,0% de recall e 23,0 alarmes/h**. No teste, o molde detectou mais chamadas e gerou mais alarmes que as CNNs calibradas. Sua taxa foi de **33,75/h em Brunswick** e **15,50/h em Hatteras**. A CNN com média móvel gerou a menor taxa no teste e teve erro de horário de até **1,0 s no percentil 95** dos eventos pareados. Os números descrevem a amostra selecionada de cada local.

### Ablações do pré-processamento

Cada CNN foi treinada novamente com as mesmas partições A+B, três seeds, até 14 épocas e o mesmo critério de parada. Uma variante omite o Butterworth no áudio e mantém o recorte espectral de 40–400 Hz. A outra mantém o filtro e omite a subtração da mediana. A comparação usa os mesmos clipes de validação, inclusive os cortes por SNR. Os resultados de C e B* correspondem ao modelo original.

| Pipeline da CNN | ROC-AUC | F1 | Recall |
|---|---:|---:|---:|
| Completo | 0,955 ± 0,003 | 0,886 ± 0,007 | 90,2% |
| Sem Butterworth no áudio | 0,942 ± 0,001 | 0,862 ± 0,004 | 86,0% |
| Sem subtração da mediana | 0,902 ± 0,017 | 0,812 ± 0,016 | 89,7% |

Valores da validação A+B. O desvio-padrão corresponde à variação entre as três seeds. Retirar a subtração da mediana produziu a maior queda de ROC-AUC e F1. As tabelas por conjunto, seed e SNR estão em `artifacts/novos_testes/ablacoes/`.

### Reprodução

Depois dos comandos da avaliação original, rode a sequência abaixo a partir da raiz do clone. A pasta NOAA pode ser definida fora do repositório para manter as gravações no armazenamento local.

```powershell
$exp = 'artifacts/novos_testes'
$noaa = Join-Path $env:LOCALAPPDATA 'Eubalaenas-PDSI/noaa_nefsc'
$anotacoes = Join-Path $dataset 'data/continuous/dataset_B/annotations_B_cont.csv'

& $python -m unittest discover -s tests -v
& $python -m src.template_detector clips
& $python -m src.template_detector continuous --data $dataset
& $python -m src.operating_curve --annotations $anotacoes --template "$exp/molde"
& $python -m src.noaa_validation download --data $noaa
& $python -m src.noaa_validation score --data $noaa
& $python -m src.noaa_validation compare --data $noaa

& $python -m src.prepare --data $dataset --output "$exp/prepared_sem_butterworth" --without-bandpass
& $python -m src.cnn train --prepared "$exp/prepared_sem_butterworth" --output "$exp/ablacoes/sem_butterworth"
& $python -m src.prepare --data $dataset --output "$exp/prepared_sem_mediana" --without-median
& $python -m src.cnn train --prepared "$exp/prepared_sem_mediana" --output "$exp/ablacoes/sem_mediana"
& $python -m src.ablation_report
& $python -m src.course_demos
```

## conhecimentos da disciplina aplicados

Os exemplos abaixo adaptam os exercícios da pasta `oi` a sinais controlados e gravações do projeto. O script `src/course_demos.py` gera as figuras e registra as medições em `artifacts/novos_testes/disciplina/medicoes.json`.

| Laboratório | Aplicação e resultado | Figuras |
|---|---|---|
| LE02 · Teorema da amostragem | Reamostragem de um sinal com tons de 150 e 700 Hz, de 32.768 para 1.000 Hz. A interpolação sem passa-baixa produziu um alias em 300 Hz. A reamostragem polifásica preservou 150 Hz e deixou o alias 71,8 dB abaixo dele. | [Antialiasing](docs/figuras/le02_antialiasing.png) |
| LE03 · Convolução 1D e 2D | Em `B_cont_13.wav`, sem upcalls anotados, a média de cinco scores reduziu o pico em 1.283,5 s de 0,441 para 0,338, abaixo do limiar de 0,402. A correlação 2D do molde com o espectrograma de `A_2032.wav` alcançou um pico de 0,397. | [Média móvel](docs/figuras/le03_media_movel.png) · [Molde e mapa de correlação](docs/figuras/le03_correlacao_2d.png) |
| LE06 · DFT em quatro versões | Em uma janela de 256 amostras, a DFT direta e a rFFT concordaram com erro relativo máximo de 6,73 × 10⁻⁸. O espaçamento entre bins é 3,906 Hz e o passo temporal é 32 ms. A crista de um chirp de 120 Hz/s foi estimada em 121,0 Hz/s. | [Da DFT à STFT](docs/figuras/le06_dft_stft.png) |
| LE11 · Filtros de imagem e tensor de estrutura | Derivadas gaussianas e o tensor de estrutura estimaram a orientação ao longo da crista em dois espectrogramas sintéticos com ruído. As inclinações medianas foram 117,6 Hz/s no chirp e −0,6 Hz/s no tom estacionário. | [Orientação no espectrograma](docs/figuras/le11_tensor_estrutura.png) |

## Documentos

- [Texto para os slides finais](docs/slides-finais.md) · Método, resultados por conjunto, limitações e notas de apresentação.
- [Resultados versionados](docs/resultados) · Métricas, histórico de treino, limiares congelados e tabelas usadas nos slides.
- [dados.md](dados.md): descrição do dataset.

## Referências

1. Kirsebom, O. S. et al. (2020). *Performance of a deep neural network at detecting North Atlantic right whale upcalls.* J. Acoust. Soc. Am. 147(4). <https://doi.org/10.1121/10.0001132>
2. Pré-print: <https://arxiv.org/abs/2001.09127v2>
3. Código dos autores (GPLv3): <https://doi.org/10.5281/zenodo.3736625>
4. Dados: <https://doi.org/10.20383/101.0241>
