# P2 PDI — Segmentação de fluidos 2D: `gray_scott_reaction_diffusion` (The Well)

**Pergunta central:** onde estão as estruturas de interesse no fluido e como elas evoluem ao longo do tempo?

Segue o **protocolo padrão da turma** (3ª rodada do professor): **10 pré-processamentos (P1–P10) × 3 arquiteturas
(U-Net, SegFormer, Mask2Former) × 5 pós-processamentos (Q1–Q5)**, entrada de no máximo 128 px no maior eixo sem
upscaling, subconjunto ≤ 25 GB, 20 épocas, métricas por classe e colunas padronizadas (Seção 12). O mapeamento
item a item do protocolo para o código, com os desvios documentados, está em `PROTOCOLO_CHECKLIST.md`.
O tópico 7 (SAM) foi desconsiderado, conforme orientação.

O experimento tem as **4 condições**: arquitetura sozinha (P0+Q0), com pré (Pi+Q0), com pós (P0+Qj) e
pré + arquitetura + pós (Pi+Qj). O efeito de cada técnica é isolado por diferença pareada nas mesmas trajetórias
de teste (`paper/tables/tab_delta.tex`), base da discussão de por que melhora ou piora.

## Desenho experimental (resumo)

| Item | Escolha |
|---|---|
| Dados | **todas as 1.200 trajetórias** das divisões oficiais do The Well (200 por regime: 160 treino / 20 validação / 20 teste); **21 instantes** por trajetória (n = 0, 50, …, 1000; Δn = 50); 3,3 GB; identificadores e revisão do dataset em `data/subset_manifest.json` |
| Regimes | gliders, bubbles, maze, worms, spirals, spots |
| Ground truth | Otsu sobre a concentração **B** no instante de referência **n = 500**; o limiar τ de cada trajetória é aplicado a todos os instantes → máscaras que evoluem no tempo. Trajetórias cujo padrão se extinguiu antes de n = 500 (70 de gliders e 4 de spirals no treino; 12 na validação, 9 no teste) usam a mediana dos τ válidos do regime no treino |
| Entrada da rede | concentração **A** (substrato), na resolução **nativa 128 × 128** (proporção 1:1, já no teto do protocolo; sem reamostragem). O GT vem de B, então a rede não recebe o campo usado para criar a máscara |
| Pré-processamento | P0 = linha de base (campo adimensional bruto, sem filtro) + P1 Z-score, P2 RMS, P3 Robust, P4 Min-Max, P5 Percentile clipping (p1 = 0,252, p99 = 1,000), P6 Gaussiano, P7 Mediana, P8 Bilateral, P9 Difusão anisotrópica, P10 Wavelet (`src/gsseg/preprocess.py`) |
| Arquiteturas | U-Net/ResNet-34, SegFormer-B2, Mask2Former/Swin-T (`src/gsseg/models.py`); Mask2Former treinado **sem perda auxiliar** (custo do casamento húngaro na CPU) |
| Pós-processamento | Q0 (p > 0,5) + Q1 otimização de limiar, Q2 abertura, Q3 fechamento, Q4 filtro de componentes conexas, Q5 consistência temporal (`src/gsseg/postprocess.py`); parâmetros escolhidos na validação |
| Treino | **20 épocas** (630 iterações/época), lote 32, AdamW + cosseno com 1 época de aquecimento, bf16, sem early stopping (checkpoint de melhor Dice na validação); aumentos físicos válidos (deslocamento circular, rotações de 90°, espelhamento) |
| Seleção | a melhor combinação de cada condição é escolhida na **validação** e reportada no **teste** |
| Referência clássica | Otsu por quadro no campo A (sem aprendizado) |
| Numeração | E001–E150 = P1–P10 × arquitetura × Q1–Q5; E151–E198 = linhas de base (sem pré e/ou sem pós); REF = Otsu em A |

## Resultados principais (teste, 120 trajetórias)

| Arquitetura | Sozinha | + Pré | + Pós | + Pré + Pós |
|---|---|---|---|---|
| U-Net | 0,953 | 0,955 (P4) | 0,961 (Q1) | 0,962 (P2+Q1) |
| SegFormer | 0,892 | 0,892 (P8) | 0,893 (Q1) | 0,893 (P8+Q1) |
| Mask2Former | 0,768 | 0,782 (P5) | 0,769 (Q2) | 0,784 (P5+Q2) |
| Otsu em A (REF) | 0,888 | | | |

Dice médio. Os maiores "ganhos" vêm do instante inicial aleatório n = 0, onde A ainda não carrega informação
sobre B: excluindo n = 0 (`scripts/07_sensibilidade_n0.py`), Q1 passa a **piorar** U-Net (−0,28 p.p.) e
SegFormer (−1,21 p.p.), e o ganho de P5 no Mask2Former cai de 1,37 para 0,18 p.p. Q5 piora todas as
arquiteturas em 8,6–11,3 p.p. (≈ 30 p.p. nos regimes com estruturas que se movem).

## Tabelas pedidas pelo professor

* **Tabelas visuais** `paper/figures/visual_table_<regime>.*`: frame 0, dois intermediários (n = 50, organização
  do padrão; n = 500, frame de referência do GT) e frame final (n = 1000); para cada arquitetura: entrada A,
  entrada pré-processada, GT e as máscaras das 4 condições (TP/FP/FN). Visão geral em `fluid_by_time.*`/`gt_by_time.*`.
* **Tabela de métricas sequencial e padronizada** (Experimento | Pré | Arquitetura | Pós | IoU | Dice | Precision |
  Recall | Latência): E001–E198 → `outputs/results/results_sequential.csv` e `paper/tables/tab_sequential.tex`.
* **CSV no formato da Seção 12** (todas as colunas padronizadas, métricas por classe): `outputs/results/results_protocolo.csv`.
* **Tabela das 4 condições** × 3 arquiteturas: `outputs/results/results_standard.csv` (coluna `Fluid`, para juntar
  com os outros fluidos), `paper/tables/tab_main.tex`, `tab_classes.tex`, `tab_temporal.tex`, `tab_sensitivity.tex`.

## Como rodar

### Local (Windows)
```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/01_download_subset.py      # 3,3 GB, só os quadros necessários (HTTP Range)
.venv/Scripts/python scripts/02_build_dataset.py        # GT + manifesto + estatísticas de P1..P5
```
O treino completo não cabe numa GPU de 8 GB em tempo razoável; use o RunPod.

### RunPod — execução completa
1. Crie um pod (A5000 24 GB ou superior) com o template *RunPod PyTorch 2.x*, montando um volume em `/workspace`.
2. Envie este projeto (sem `data/`, `outputs/` e `.venv/`) para `/workspace/ATV_NOTA_02` — ex.: o zip
   `p2_gray_scott_code.zip` gerado por `scripts/make_bundle.py`.
3. No terminal do pod:
   ```bash
   cd /workspace/ATV_NOTA_02
   bash runpod/setup.sh
   tmux new -d -s run "bash runpod/run_all.sh > outputs/run_all.log 2>&1"
   tail -f outputs/run_all.log
   ```
4. Ao final, baixe `outputs/results/`, `outputs/models/*/meta.json`, `outputs/models/*/history.csv`,
   `paper/figures/` e `paper/tables/` para as mesmas pastas deste projeto e compile o artigo:
   ```bash
   cd paper && latexmk -pdf main.tex
   ```

O pipeline é retomável: se o pod cair, rode `run_all.sh` de novo e os modelos já treinados são pulados.
O `run_all.sh` lê a cota de CPU do contêiner (`cpu.max`), roda 5 filas de treino em paralelo (U-Net, SegFormer e
Mask2Former dividido em 3 filas), mede a latência com a GPU dedicada depois dos treinos e avalia com 6 processos.
Na A5000 a execução completa levou **≈ 11,5 h**.

**Memória (rodando localmente na RTX 4060 / 32 GB):** use um processo de GPU por vez e
`scripts/04_postprocess_eval.py --workers 1`. Vários processos numa GPU de 8 GB estouram a VRAM e o Windows passa
a usar a RAM do sistema como memória de vídeo, travando o PC.

## Estrutura
```
configs/experiment.yaml   configuração única (sementes, divisão, GT, pré/pós, treino, avaliação)
src/gsseg/                gt, preprocess, models, postprocess, metrics, experiments, utils
scripts/01..05            download → GT/pré → treino+predição → pós+métricas → tabelas/figuras (inglês)
scripts/06_artigo_sbc.py  tabelas e figuras em português para o artigo SBC
scripts/07_sensibilidade_n0.py  métricas por trajetória com e sem n = 0
runpod/                   setup.sh e run_all.sh
paper/                    artigo em inglês (elsarticle / Results in Engineering) + figures/ + tables/
artigo_sbc/               artigo em português (template SBC) + figuras/ + tabelas/
outputs/arquivo_protocolo_v1/  resultados da rodada anterior (240 trajetórias, Δn = 10), só para registro
```

## Artigo SBC (português) — `artigo_sbc/`

Versão do artigo no template SBC local (o mesmo dos artigos de PLN e de LLM do autor), em português com
Abstract e Resumo, redigida segundo `MESTRADO - ESTUDO LLM/04_escrita/artigo/seminarios_2026/PADRAO_ESCRITA.md`.
`sbc-template.sty` e `sbc.bst` foram copiados sem alteração (hashes em `artigo_sbc/PROVENIENCIA_TEMPLATE.md`).

```bash
.venv/Scripts/python scripts/06_artigo_sbc.py      # tabelas (vírgula decimal) e figuras em português
cd artigo_sbc && latexmk -pdf artigo.tex           # pdfLaTeX + BibTeX (sbc.bst)
```

Estrutura: Introdução (objetivo, PP1–PP4, contribuição e limites) → Fundamentação teórica e trabalhos
relacionados (Gray–Scott, aplicações, The Well, arquiteturas, pré/pós-processamento, métricas) → Materiais e
métodos (pipeline gráfico) → Resultados (tabelas visuais, uma subseção por PP, sensibilidade a n = 0) →
Discussão → Limitações → Conclusão → Apêndices (tabela sequencial E001–E198 e demais tabelas visuais).

## Artigo no template da disciplina — `artigo_pdi/`

Versão principal do artigo no template `template_pdi_com_tabelas_segmentacao.tex` (Elsevier, Results in
Engineering, duas colunas), com cerca de 14 páginas, em inglês (`main.tex`) e em português (`main_pt.tex`).
As tabelas das duas versões saem dos mesmos CSVs (`tables/` e `tables_pt/`), e o texto não usa travessão (`--`).

```bash
.venv/Scripts/python scripts/08_artigo_pdi.py       # tabelas EN e PT + cópia das figuras
cd artigo_pdi && latexmk -pdf main.tex && latexmk -pdf main_pt.tex
```
