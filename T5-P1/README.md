# T5 / P1 — Simulador de Espelho Corporal (Tema 6)

**Disciplina:** Processamento de Imagens — UFMA
**Tema:** estimativa de pose humana em tempo real — MediaPipe Pose + OpenCV
**Grupo:** André Luis Aguiar do Nascimento, João Guilherme Miranda Lago e Wesley dos Santos Gatinho
**Professor:** Haroldo Gomes Barroso Filho

O programa captura a webcam, extrai os **33 landmarks** corporais e redesenha
o esqueleto do participante em tempo real, como um espelho.

```
WEBCAM → OpenCV (captura) → MediaPipe Pose Landmarker → 33 landmarks
       → espelhamento + suavização → desenho do esqueleto → tela + métricas
```

---

## Como executar

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate          # Windows
```

```bash
source .venv/bin/activate        # Linux / macOS
```

```bash
pip install -r requirements.txt
```

```bash
python espelho.py
```

Na primeira execução o modelo escolhido é baixado para `modelos/`
(Lite 6 MB, Full 9 MB, Heavy 31 MB). Os avisos `INFO:`/`WARNING:` que o
MediaPipe imprime ao carregar o modelo são normais.

Opções úteis:

```bash
python espelho.py --modelo lite --camera 1
```

```bash
python espelho.py --registrar --duracao 30
```

```bash
python espelho.py --video caminho/video.mp4 --sem-janela --registrar
```

## Se o FPS ficar baixo

O painel mostra, ao lado do FPS, a **espera da câmera**: o tempo que o
programa fica parado aguardando o próximo quadro. Se ela for bem maior que a
latência, quem limita o FPS é a webcam (pouca luz, formato sem compressão ou
resolução alta), não o modelo. Para descobrir a melhor configuração da câmera:

```bash
python testar_camera.py
```

O script mede o FPS real em várias combinações (sem salvar imagens) e imprime
o comando `python espelho.py ...` recomendado. No Windows testa DirectShow e
Media Foundation; no Linux, V4L2. Com pouca luz, a exposição automática da
webcam aumenta e o FPS pode cair; acender a luz de frente para a pessoa
resolve. No Linux a fila da câmera usa 2 buffers: com 1 só, o V4L2 descarta
um quadro a cada dois e a webcam fica presa em 15 FPS mesmo com luz boa.

## Gravar e analisar as métricas

O jeito recomendado é o **roteiro guiado**, que faz tudo sozinho:

```bash
python espelho.py --roteiro
```

Ele percorre Lite, Full e Heavy automaticamente (cerca de 1 min 20 s). Uma
faixa na tela diz o que fazer e conta o tempo: `PREPARE-SE` (não é gravado),
`FIQUE PARADO` (10 s) e `MOVIMENTE-SE` (10 s) em cada modelo. Fique de corpo
inteiro no quadro. No fim a janela fecha e a análise é gerada só com essa
gravação (`resultados/sessoes/roteiro_<data>.csv`).

Também dá para gravar livremente com a tecla `R` (ou `--registrar`) e
analisar depois tudo o que estiver em `resultados/sessoes/`:

```bash
python analise.py
```

```bash
python analise.py --arquivos resultados/sessoes/roteiro_20261116_230000.csv
```

Saídas: `resultados/metricas/RESUMO.md`, `resultados/metricas/resumo_por_modelo.csv`
e, em `resultados/graficos/`, `desempenho_por_modelo.png`, `tempo_por_quadro.png`,
`latencia_no_tempo.png`, `estabilidade.png` e `deteccao_por_modelo.png`.
Gravações com menos de 30 quadros são ignoradas.

## Medir a taxa de acerto

```bash
python espelho.py --gravar-video
```

Grava o vídeo bruto da câmera (sem esqueleto, sem espelhar) enquanto uma
faixa na tela pede 8 poses de 5 s cada: braços para cima, braços abertos,
braços cruzados, agachamento, de perfil, em uma perna, andando de lado e
movimento livre (cerca de 50 s com o preparo). Fique de corpo inteiro, com
ombros e quadris sempre no quadro. No fim, o mesmo vídeo é processado pelos
três modelos e a taxa de acerto é calculada automaticamente. Para refazer a
conta sobre um vídeo já gravado:

```bash
python acuracia.py
```

Como não há gabarito anotado à mão, o **Heavy serve de referência**: mede-se
o quanto os pontos do Lite e do Full concordam com os dele.

- **Taxa de acerto = PCK@0,2**: % de landmarks a menos de 0,2 × comprimento
  do tronco (ombros → quadris) do ponto do Heavy. PCK@0,1 é a versão rigorosa.
- Também por parte do corpo, por pose e um exemplo visual do pior quadro.
- Saídas: uma pasta por vídeo, `resultados/acuracia/<nome do vídeo>/`, com
  `RESUMO_acuracia.md` e os gráficos `acuracia_*.png`; gravar de novo não
  apaga os resultados anteriores. O vídeo fica em `resultados/videos/`
  (fora do git, por ser grande).
- **Enquadramento:** cabeça e pés precisam aparecer. Parte do corpo fora do
  quadro não é avaliada, e com a cabeça cortada até o Heavy erra no início
  da detecção, o que contamina a referência.
- **Limitação:** os três modelos têm a mesma arquitetura e treino, então erros
  que os três cometem juntos não aparecem. O número mede concordância com o
  modelo mais preciso, não acerto absoluto.

## Controles

| Tecla | Ação |
|---|---|
| `1` `2` `3` `4` | modo: sobreposto · avatar · lado a lado · rastro |
| `L` `F` `H` | troca o modelo: Lite · Full · Heavy (imprime o resumo do anterior) |
| `S` | liga/desliga a suavização (filtro One Euro) |
| `N` | mostra o número de cada landmark (0–32) |
| `E` | liga/desliga o espelhamento |
| `M` | mostra/esconde o painel de métricas |
| `R` | inicia/encerra a gravação das métricas em CSV |
| `P` | salva uma foto da tela em `resultados/capturas/` |
| `Q` / `Esc` | sair |

Cores do esqueleto: **laranja** = lado esquerdo da pessoa, **azul** = lado
direito, **branco** = ligações centrais (ombros, quadril).

## Como cada critério de avaliação é medido

| Métrica | Como o programa mede |
|---|---|
| **FPS** | quadros apresentados por segundo (média móvel de 30 quadros no painel; média da sessão no resumo). |
| **Latência** | tempo entre o quadro chegar da câmera (`cap.read()`) e estar visível na janela (após `imshow` + `waitKey`). Inclui inferência, desenho e exibição; não inclui o atraso interno do sensor/driver da webcam. |
| **Taxa de acerto** | PCK@0,2 de Lite e Full contra o Heavy num vídeo gravado com 8 poses (ver "Medir a taxa de acerto"). Ao vivo, o painel mostra também a taxa de detecção (% de quadros com pose encontrada). |
| **Estabilidade** | *jitter*: deslocamento médio dos landmarks entre quadros consecutivos (px) — com a pessoa parada, é o tremor da detecção; e nº de trocas detectou ↔ perdeu. No roteiro guiado os trechos parados são marcados pelo próprio roteiro; na gravação livre, "parado" é estimado pelos dados (mediana móvel do deslocamento abaixo de 4 px), o que é só uma aproximação. |

Ao sair (ou ao trocar de modelo) o resumo da sessão é impresso no terminal.
Com `R` ou `--registrar`, cada quadro vira uma linha em
`resultados/sessoes/sessao_<data>.csv`.

## Apresentação

Slides em [`apresentacao/apresentacao.pdf`](apresentacao/apresentacao.pdf) e
texto de cada slide, com o roteiro da demonstração, em
[`apresentacao/script_de_fala.md`](apresentacao/script_de_fala.md).

Duas figuras dos slides (`modos.png` e `lite_vs_full.png`) são feitas a partir
de um vídeo de poses de corpo inteiro (o vídeo fica fora do git; as figuras já
estão em `apresentacao/figuras/`). Para refazê-las com outro vídeo, grave,
gere as figuras e recompile:

```bash
python espelho.py --gravar-video
```

```bash
python apresentacao/gerar_figuras.py
```

```bash
cd apresentacao && latexmk -xelatex apresentacao.tex
```

## Estrutura

```
espelho.py          aplicação principal (laço captura → detecção → desenho)
analise.py          tabelas e gráficos a partir das sessões gravadas
acuracia.py         taxa de acerto (PCK) de Lite e Full contra o Heavy num vídeo
testar_camera.py    mede o FPS real da webcam em várias configurações
src/camera.py       abertura e configuração da webcam
src/roteiro.py      roteiros guiados (modelos e poses) com a instrução na tela
src/gravador.py     grava o vídeo bruto da câmera e a pose de cada quadro
src/detector.py     MediaPipe Pose Landmarker (Lite/Full/Heavy), devolve matriz 33×4
src/desenho.py      esqueleto, modos de visualização e painel (só OpenCV)
src/suavizacao.py   filtro One Euro
src/metricas.py     FPS, latência, detecção, jitter e gravação em CSV
modelos/            arquivos .task (baixados automaticamente)
resultados/         CSVs das sessões, tabelas, gráficos e capturas de tela
```

## Detalhes de implementação

- **Espelhamento correto.** A detecção roda no quadro original e só depois a
  imagem é invertida e o `x` de cada landmark vira `1 − x`. Se a imagem fosse
  invertida antes, o MediaPipe trocaria esquerda e direita.
- **Visibilidade.** Landmarks com visibilidade < 0,5 (fora do quadro ou
  encobertos) não são desenhados.
- **Suavização.** O MediaPipe já filtra os pontos internamente; o One Euro
  adicional reduz o tremor restante quando a pessoa está parada e quase não
  atrasa em movimentos rápidos (o corte do filtro cresce com a velocidade).
