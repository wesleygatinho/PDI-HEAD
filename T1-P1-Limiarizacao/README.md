# T1 / P1 — Limiarização de imagens e vídeo

**Disciplina:** Processamento de Imagens — UFMA
**Tema:** segmentação por limiar — Otsu · Multi-Otsu · Adaptativa
**Trabalho em trio** — cada aluno apresenta uma técnica

---

## O enunciado

> Acessar as três páginas de referência (básico, Multi-Otsu, adaptativo),
> **aplicar um threshold em uma imagem e em um vídeo curto de 20 s**,
> **analisar a diferença de comportamento entre as técnicas** e
> **apresentar os conceitos e fundamentos matemáticos de cada uma**.
> Cada aluno apresenta uma técnica.

| # | Técnica | Responsável | Fonte |
|---|---|---|---|
| 1 | Limiarização global / **Otsu** | **Aluno A** | [scikit-image · plot_thresholding](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html) |
| 2 | **Multi-Otsu** | **Aluno B** | [scikit-image · plot_multiotsu](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html) |
| 3 | Limiarização **adaptativa** | **Aluno C** | [OpenCV · py_thresholding](https://docs.opencv.org/4.4.0/d7/d4d/tutorial_py_thresholding.html) |

---

## Como executar

```bash
pip install -r requirements.txt
```

```bash
python executar_tudo.py
```

Isso gera **todo** o material (binarizações, vídeos de saída, gráficos e
tabelas) em `resultados/`. Leva cerca de **1 minuto**.

Os slides já vêm compilados em
[`apresentacao/apresentacao.pdf`](apresentacao/apresentacao.pdf) — só é preciso
preencher os nomes do grupo no topo do `.tex` e recompilar. Ver
[`apresentacao/README.md`](apresentacao/README.md).

Para rodar as etapas separadamente:

```bash
python src/gerar_midia.py      # imagens de teste + vídeo de 20 s com gabarito
```

```bash
python src/exp01_imagem.py     # limiarização em imagens estáticas
```

```bash
python src/exp02_video.py      # limiarização quadro a quadro no vídeo
```

```bash
python src/exp03_analise.py    # análise comparativa e mapas de erro
```

Para usar **um vídeo próprio** em vez do sintético:

```bash
python src/exp02_video.py --video "caminho/para/seu_video.mp4"
```

> Sem gabarito, as métricas de IoU/F1 ficam indisponíveis, mas os gráficos de
> limiar ao longo do tempo, área segmentada e estabilidade temporal continuam
> todos válidos.

---

## Estrutura de pastas

```
T1-P1-Limiarizacao/
│
├── README.md                  ← você está aqui
├── requirements.txt
├── executar_tudo.py           ← executa o pipeline completo
│
├── docs/                      ← A PARTE ESCRITA (teoria + análise)
│   ├── 00_fundamentacao_teorica.md          base comum aos três
│   ├── 01_alunoA_limiarizacao_global_otsu.md
│   ├── 02_alunoB_multi_otsu.md
│   ├── 03_alunoC_limiarizacao_adaptativa.md
│   ├── 04_analise_comparativa.md            ← responde ao enunciado
│   └── 05_roteiro_apresentacao.md           divisão e roteiro do trio
│
├── apresentacao/              ← OS SLIDES (Beamer/LaTeX)
│   ├── apresentacao.tex       36 slides — preencha os nomes no topo
│   ├── apresentacao.pdf       já compilado
│   ├── README.md              como compilar e o mapa dos slides
│   └── figuras/               versões dos gráficos feitas para projeção
│
├── src/                       ← CÓDIGO
│   ├── limiarizacao.py        os 3 métodos implementados do zero + wrappers
│   ├── config.py              todos os parâmetros em um lugar só
│   ├── utils.py               E/S, métricas de avaliação, figuras
│   ├── gerar_midia.py         cria as imagens e o vídeo de teste
│   ├── exp01_imagem.py        experimento 1 — imagens
│   ├── exp02_video.py         experimento 2 — vídeo
│   └── exp03_analise.py       experimento 3 — análise das causas
│
├── data/
│   ├── imagens/               3 imagens reais + 1 sintética com gabarito
│   └── videos/                vídeo de 20 s + vídeo do gabarito
│
└── resultados/
    ├── imagem/                binarizações e painéis comparativos
    ├── video/                 mosaico_comparativo.mp4 ← principal
    ├── frames/                um quadro de destaque por fase
    ├── graficos/              todas as figuras
    └── metricas/              CSVs + RESUMO_imagem.md e RESUMO_video.md
```

---

## O material de teste

### Imagens

| Arquivo | Origem | Por que foi escolhida |
|---|---|---|
| `01_camera.png` | scikit-image | histograma nitidamente bimodal — o caso ideal de Otsu ($\eta = 0{,}857$) |
| `02_moedas.png` | scikit-image | fundo com iluminação desigual |
| `03_pagina.png` | scikit-image | texto digitalizado sob luz desigual — o caso clássico da adaptativa |
| `04_sintetica.png` | gerada | **possui gabarito**, permitindo medir IoU/F1/MCC |

### Vídeo (20 s, 640×480, 30 fps, 600 quadros)

Cena gerada com **gabarito analítico quadro a quadro**, dividida em 4 fases de
5 s, cada uma isolando um fator de degradação:

| Fase | Tempo | O que muda | Quem é penalizado |
|---|---|---|---|
| 1 — Uniforme | 0–5 s | nada (referência) | ninguém |
| 2 — Holofote móvel | 5–10 s | iluminação varia **no espaço** | **os métodos globais** |
| 3 — Escurecimento | 10–15 s | iluminação varia **no tempo** | **o limiar fixo** |
| 4 — Ruído | 15–20 s | $\sigma$ de 4→30, contraste 1,0→0,45 | **os métodos locais** |

Essa separação entre variação **espacial** (fase 2) e **temporal** (fase 3) é o
que torna possível atribuir causas, e não apenas observar efeitos.

---

## Resultados principais

### IoU médio por fase do vídeo (melhor de cada coluna em negrito)

| Método | Fase 1 | Fase 2 | Fase 3 | Fase 4 | Global |
|---|---|---|---|---|---|
| Limiar fixo T=127 | 0,657 | 0,140 | 0,636 | 0,634 | 0,516 |
| Otsu global | 0,944 | 0,139 | 0,943 | **0,834** | 0,715 |
| Multi-Otsu $t_1$ | 0,592 | 0,731 | 0,600 | 0,639 | 0,641 |
| Multi-Otsu $t_2$ | 0,966 | 0,138 | **0,965** | 0,626 | 0,674 |
| Adaptativa (b=35) | 0,621 | 0,528 | 0,561 | 0,341 | 0,513 |
| **Adaptativa ajustada (b=151)** | **0,971** | **0,956** | 0,950 | 0,575 | **0,863** |
| Sauvola | 0,787 | 0,757 | 0,756 | 0,627 | 0,732 |

### As três conclusões

1. **O limiar global falha com variação ESPACIAL de iluminação, não com variação
   temporal.** O mesmo Otsu marca 0,943 na fase 3 (a cena escurece por completo, e
   ele baixa o limiar sozinho de 147 para 47 sem perder qualidade) e 0,139 na
   fase 2 (a luz varia dentro do quadro). Dizer que "Otsu falha com iluminação
   variável" é impreciso.

2. **A limiarização adaptativa troca robustez à iluminação por fragilidade ao
   ruído.** Fase 2: 0,956 — quase 7× o Otsu. Fase 4: 0,575, com **22 %** dos
   pixels piscando entre quadros consecutivos. As duas coisas têm a mesma causa:
   ela decide olhando poucos pixels.

3. **Nenhum método venceu as quatro fases.** A escolha não é sobre qual técnica é
   melhor, é sobre **qual hipótese a sua cena satisfaz**.

Detalhamento completo em [`docs/04_analise_comparativa.md`](docs/04_analise_comparativa.md).

---

## Implementações próprias e validação

Os três métodos foram implementados **do zero em NumPy** (`src/limiarizacao.py`),
com a matemática explicitada nas docstrings, e depois **validados contra as
bibliotecas de referência**:

| Método | Implementação | Resultado da validação |
|---|---|---|
| Otsu | somas acumuladas, $O(N{+}L)$ | **limiar idêntico** ao skimage e ao OpenCV nas 4 imagens |
| Multi-Otsu | LUT de momentos + **programação dinâmica**, $O(ML^2)$ | **limiares idênticos** ao skimage em 12 casos ($M = 2..5$ × 3 imagens) |
| Adaptativa | **imagem integral**, $O(1)$ por pixel | 99,3 – 99,9 % de concordância com o OpenCV |
| Sauvola | dois momentos por imagem integral | — |

Sobre os 0,1 % de discordância na adaptativa: analisando os 293 pixels
divergentes, **100 % deles têm $|I - T| < 0{,}5$** (desvio máximo 0,499), contra
uma folga média de 16,7 níveis nos pixels concordantes. A causa é que o OpenCV
calcula a média local em aritmética inteira e a implementação própria usa
`float64` — é arredondamento, não erro.

Uma nota honesta sobre o processo: a primeira versão do Multi-Otsu usava busca
combinatória e **travou** em $M = 5$ (são 172 milhões de combinações). A troca
por programação dinâmica levou o tempo para **1,5 ms** — e o episódio virou um
dos pontos da apresentação do Aluno B.

---

## Figuras principais

| Arquivo | O que mostra |
|---|---|
| `resultados/video/mosaico_comparativo.mp4` | **o vídeo principal**: 6 painéis lado a lado com IoU ao vivo |
| `graficos/B_otsu_04_sintetica.png` | a identidade $\sigma_T^2 = \sigma_W^2 + \sigma_B^2$ em ação |
| `graficos/C2_multiotsu_M_04_sintetica.png` | efeito do número de classes $M$ |
| `graficos/D_adaptativa_04_sintetica.png` | o mapa de limiares $T(x,y)$ |
| `graficos/E_sensibilidade_04_sintetica.png` | varredura bloco × C (36 combinações) |
| `graficos/V1_limiar_no_tempo.png` | como cada método move o limiar nos 20 s |
| `graficos/V2_iou_no_tempo.png` | qualidade quadro a quadro |
| `graficos/V3_instabilidade_no_tempo.png` | cintilação (só existe em vídeo) |
| `graficos/X1_mapas_de_erro.png` | falsos positivos (vermelho) × falsos negativos (azul) |
| `graficos/X3_efeito_bloco_e_C.png` | por que o bloco precisa ser maior que o objeto |
| `graficos/X4_sintese.png` | painel-síntese para o slide de conclusão |

---

## Ambiente testado

Python 3.11.9 · numpy 2.4.4 · scikit-image 0.26.0 · opencv-python 4.13.0 ·
matplotlib 3.10.8 · Windows 11

### Sobre reprodutibilidade — uma nota honesta

Rodar os experimentos sobre o vídeo entregue em `data/videos/` reproduz **todas
as tabelas deste relatório, número a número** (verificado: duas execuções
consecutivas geram CSVs de métricas byte a byte idênticos — só as colunas de
tempo variam, como esperado).

A geração da cena é bit-exata (semente fixa, `config.SEMENTE = 42`). O
**codificador H.264 não é**: o libx264 escolhe o número de threads conforme a
carga da máquina, o que muda os bytes do arquivo. Como o codec tem perdas (erro
médio de ~3 níveis de cinza), regerar o vídeo desloca as métricas em até
**±0,03 de IoU** — o suficiente para os decimais não baterem, embora nenhuma
conclusão mude.

Por isso `gerar_midia.py` **reaproveita o vídeo se ele já existir**. O arquivo em
`data/videos/` faz parte da entrega. Para regerar do zero:

```bash
python src/gerar_midia.py --forcar
```

…e então reexecutar os experimentos, aceitando que os decimais mudarão um pouco.

### Tamanho em disco

O projeto completo ocupa cerca de 140 MB, quase tudo em vídeo (a fase de ruído é
incompressível por natureza). Para entregar só o essencial, basta enviar `docs/`,
`src/`, `data/`, `resultados/graficos/`, `resultados/metricas/` e o
`mosaico_comparativo.mp4` — o restante se reconstrói com `executar_tudo.py`.
