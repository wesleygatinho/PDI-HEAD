# Checklist do protocolo da turma — `gray_scott_reaction_diffusion`

Correspondência entre o protocolo padrão (Seções 3–12) e esta implementação. Itens marcados "N/A" não se
aplicam a este dataset; a justificativa está ao lado, como pede a Seção 15 do protocolo.

## Decisões padrão (Seção 14)

| Item | Padrão da turma | Este dataset | Onde está |
|---|---|---|---|
| Resolução espacial | maior eixo ≤ 128 px | 128 × 128 nativo, **sem reamostragem** | `configs/experiment.yaml` (`input`) |
| Upscaling | proibido | nenhum (inclusive removida a ampliação interna 2× das Transformers) | `src/gsseg/models.py` (`input_scale = 1`) |
| Proporção | preservar | 1:1 nativa | — |
| Padding | só se exigido; excluir das métricas | N/A: 128 já é múltiplo de 32 | — |
| Armazenamento | teto de 25 GB | ~3,3 GB: todas as 1.200 trajetórias × 21 instantes × 2 campos | `data/subset_manifest.json` |
| Subamostragem | trajetórias e frames distribuídos no tempo | todas as trajetórias; n = 0, 50, …, 1000 | `scripts/01_download_subset.py` |
| Split | oficial, por trajetória | arquivos oficiais train/valid/test (160/20/20 por regime) | manifesto |
| Ground truth | critério físico documentado e congelado | `1[B > τ]`, τ = Otsu de B em n = 500 por trajetória; extinção → mediana do regime no treino | `src/gsseg/gt.py`, manifesto |
| Pré-processamento | P1–P10 + baseline, separados | P0 baseline + P1–P10 do protocolo | `src/gsseg/preprocess.py` |
| Arquiteturas | U-Net, SegFormer, Mask2Former | idem (Mask2Former sem supervisão profunda; ver abaixo) | `src/gsseg/models.py` |
| Épocas | 20 fixas, melhor checkpoint na validação | 20 épocas, sem early stopping; `best_epoch` registrado | `scripts/03_train_predict.py`, `meta.json` |
| Pós-processamento | Q1–Q5 + baseline, parâmetros na validação | Q0 + Q1–Q5; limiar, raio, área e janela escolhidos na validação | `src/gsseg/postprocess.py` (`TUNABLE`) |
| Métricas principais | Dice e IoU | Dice e IoU (classe estrutura) | `src/gsseg/metrics.py` |
| Métricas complementares | precisão, revocação, latência, memória, estabilidade, IC, seeds | + IoU/Dice da classe fundo, mIoU, AVE, CDE, TC/TCE, IC 95% por bootstrap | idem |
| Combinações | 10 × 3 × 5 = 150 por dataset | 150 (E001–E150) + 48 linhas de baseline (E151–E198) | `src/gsseg/experiments.py` |
| Treinos | 10 × 3 = 30 por dataset (+ baseline) | 33 (11 entradas × 3 arquiteturas) | `runpod/run_all.sh` |

## Baseline de pré-processamento (Seção 6)

O protocolo pede um baseline com a "normalização padrão escolhida, sem filtro espacial adicional". Para o
Gray–Scott, a normalização padrão é a própria adimensionalização do modelo: as concentrações já são
expressas em relação à concentração de alimentação de A (= 1). O baseline P0 é, portanto, o campo
adimensional sem transformação nem filtro. Os filtros P6–P10 são aplicados sobre essa mesma entrada, o que
separa o efeito da normalização (P1–P5 contra P0) do efeito do filtro (P6–P10 contra P0).

## Checklist mínimo (Seção 11)

- [x] Versão/variante do dataset registrada → `dataset_variant` (repo@commit) no manifesto.
- [x] Split oficial registrado → `trajectory_split` e IDs por split/regime no manifesto.
- [x] IDs das trajetórias e frames selecionados salvos → `splits.*.trajectory_ids`, `frames_selected`.
- [x] Resolução original e após redução registradas → 128×128 / 128×128.
- [x] Método de reamostragem registrado → "none".
- [x] Ground truth e critério congelados → regra, τ por trajetória e fallback no manifesto.
- [x] Estatísticas de normalização só do treino → `normalization_stats_train` e `data/processed/pre_stats.json`.
- [x] Parâmetros de pré e pós registrados → `PRE_PARAMS`, `POST_PARAMS`, `outputs/results/post_params.json`.
- [x] Arquitetura, hiperparâmetros e seed registrados → `configs/experiment.yaml`, `outputs/models/*/meta.json`.
- [x] Checkpoints e predições salvos → `outputs/models/*/best.pt`, `outputs/preds/`.
- [x] Métricas ignorando padding → N/A (não há padding).
- [x] Teste isolado até a avaliação final → seleção de checkpoint e de parâmetros somente na validação.

## Desvios documentados

- **Mask2Former sem supervisão profunda**: a perda é calculada só na camada final do decodificador, por custo
  (2,67 s contra 0,79 s por iteração medidos no RunPod). Arquitetura e inferência inalteradas.
- **Piloto de resolução (Seção 3)**: N/A. O dataset já está no teto de 128 px, e um nível maior exigiria
  upscaling, proibido pelo protocolo.
- **Piloto de volume 10/25/50 GB (Seção 4)**: recomendado para Euler. Aqui o subconjunto usa todas as
  trajetórias e fica bem abaixo do teto.
- **Seeds**: uma seed (42) por treino, por orçamento; a variação entre seeds não é estimada.
