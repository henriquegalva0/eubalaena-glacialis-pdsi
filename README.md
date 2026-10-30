# Eubalaenas — Detecção de upcalls da baleia-franca-do-Atlântico-Norte

Seminário da disciplina de **Processamento de Sinais e Dados** (UFG).

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

Fonte: Kirsebom et al. (2020), gravações no Golfo de São Lourenço, Canadá. DOI: <https://doi.org/10.20383/101.0241>

Os dados **não estão versionados** neste repositório (~6 GB, pasta `narw-data/` no `.gitignore`). Baixe o `narw-data.zip` pelo DOI e extraia na raiz do projeto.

| Conjunto | Partição | Clipes | Com upcall |
|----------|----------|-------:|-----------:|
| A | treino | 1.767 | 744 |
| A | validação | 306 | 56 |
| B | treino | 3.309 | 2.033 |
| B | validação | 579 | 341 |
| C | treino | 3.000 | 1.500 |

Contagens obtidas das planilhas de anotação. Os conjuntos A, B e C diferem em local, período e instalação do hidrofone.

Além dos clipes, há gravações **contínuas de 30 s** (conjunto B), com 1.157 upcalls anotados, para testar a detecção em janela deslizante.

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

## Metodologia planejada

```
Áudio (.wav) → Pré-processamento → Espectrograma (STFT) → Modelo → upcall / ruído
              (normalização,                              (baseline e CNN)
               filtro passa-faixa)
```

1. **Análise exploratória:** balanço de classes, distribuição de SNR, espectrogramas médios por classe e comparação entre os conjuntos A, B e C.
2. **Baseline:** features (por exemplo MFCC) com um classificador clássico.
3. **Modelo principal:** CNN 2D sobre o espectrograma.
4. **Avaliação:** precisão, recall, F1, AUC, desempenho por faixa de SNR e generalização entre locais (treino em A+B, teste em C).
5. **Teste contínuo:** janela deslizante nas gravações de 30 s.

## Documentos

- [tema_e_slides.md](tema_e_slides.md): tema, justificativa e roteiro dos 6 slides da parte 1.
- [dados.md](dados.md): descrição do dataset.

## Referências

1. Kirsebom, O. S. et al. (2020). *Performance of a deep neural network at detecting North Atlantic right whale upcalls.* J. Acoust. Soc. Am. 147(4). <https://doi.org/10.1121/10.0001132>
2. Pré-print: <https://arxiv.org/abs/2001.09127v2>
3. Código dos autores (GPLv3): <https://doi.org/10.5281/zenodo.3736625>
4. Dados: <https://doi.org/10.20383/101.0241>
