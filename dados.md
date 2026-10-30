# Dados: gravações acústicas de "upcalls" da baleia-franca-do-Atlântico-Norte no Golfo de São Lourenço

Conjunto de dados: <https://doi.org/10.20383/101.0241>

## 1. Visão geral dos arquivos

### Estrutura de pastas

```
narw-data.zip
narw-data/
├── data/
│   ├── clips/
│   │   ├── dataset_A/
│   │   │   ├── train/  (annotations_A_train.csv + audio/)
│   │   │   └── val/    (annotations_A_val.csv + audio/)
│   │   ├── dataset_B/
│   │   │   ├── train/  (annotations_B_train.csv + audio/)
│   │   │   └── val/    (annotations_B_val.csv + audio/)
│   │   └── dataset_C/
│   │       └── train/  (annotations_C_train.csv + audio/)
│   └── continuous/
│       └── dataset_B/  (annotations_B_cont.csv + audio/)
└── figures/
    └── specs/
```

### O que tem em cada pasta

**data/clips**
- Contém trechos de áudio de **3 segundos**, usados para **treinar** a rede neural e **validar** o desempenho dela na classificação.
- Os dados são divididos em três conjuntos (A, B e C). Eles diferem no período, no local e na forma como o hidrofone (microfone subaquático) foi instalado. O artigo científico explica melhor.
- Cada conjunto tem as pastas `train` (treino) e `val` (validação).
- Os áudios são `.wav`, com taxa de amostragem de 1.000 amostras por segundo.
- Há também as **anotações feitas por especialistas**, em `.csv`, dizendo se há ou não um "upcall" da baleia em cada trecho. Elas trazem a **razão sinal-ruído (SNR)**, em decibéis (dB). As colunas são:
  - `clip_name`: nome do arquivo de áudio
  - `label`: 1 = há um upcall; 0 = não há
  - `snr`: razão sinal-ruído, em dB

**data/continuous**
- Contém arquivos de áudio de **30 segundos**, usados para **testar** a capacidade da rede de detectar upcalls em gravações contínuas.
- Também tem anotações de especialistas em `.csv`, com os momentos em que ocorrem upcalls. As colunas são:
  - `filename`: nome do arquivo de áudio
  - `timestamp`: instante central do upcall, em segundos a partir do início do arquivo
  - `snr`: razão sinal-ruído, em dB

**figures/specs**
- Imagens PNG de **espectrogramas** (representação visual do som) de todos os trechos de 3 s dos conjuntos A, B e C. Inclui também os upcalls identificados pelo especialista nos dados contínuos.
- Nome das imagens dos trechos de 3 s: `spec_X_Y.png`
  - X = conjunto (A, B ou C)
  - Y = número do arquivo de áudio
- Nome das imagens dos dados contínuos: `spec_B_cont_Y_Zs.png`
  - Y = número do arquivo de áudio
  - Z = deslocamento em segundos em relação ao início do arquivo

## 2. Pesquisas relacionadas

- Estes dados foram usados para treinar uma rede neural profunda que detecta upcalls da baleia-franca-do-Atlântico-Norte. O trabalho está descrito no artigo *"Performance of a Deep Neural Network at Detecting North Atlantic Right Whale Upcalls"* (Kirsebom, Frazao, Simard, Roy, Matwin e Giard, 2020, *Journal of the Acoustical Society of America*). O artigo também descreve os dados em detalhe: <https://doi.org/10.1121/10.0001132>
- Há uma versão preliminar (pré-print) no arXiv: <https://arxiv.org/abs/2001.09127v2>
- O software desenvolvido no trabalho está no Zenodo, com licença GPLv3: <https://doi.org/10.5281/zenodo.3736625>. Ele inclui scripts em Python para:
  - montar os conjuntos de treino e teste
  - calcular a razão sinal-ruído
  - visualizar espectrogramas
  - treinar a rede neural
  - usar a rede para analisar gravações contínuas

## Informações metodológicas

**1. Como os dados foram coletados/gerados:** a descrição está no artigo publicado (<https://doi.org/10.1121/10.0001132>) e no pré-print (<https://arxiv.org/abs/2001.09127v2>).

**2. Como os dados foram processados:** a descrição está nas mesmas fontes.
