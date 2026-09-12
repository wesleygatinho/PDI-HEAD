# 5. Roteiro de apresentação — divisão para o trio

> Duração total sugerida: **35–40 min** (≈ 11 min por aluno + 5 min de fechamento
> conjunto). Ajustar proporcionalmente se o tempo for menor.

---

## 5.1 Divisão do trabalho

| Aluno | Tema | Documento | Fonte oficial |
|---|---|---|---|
| **A** | Limiarização global e método de Otsu | `01_alunoA_limiarizacao_global_otsu.md` | scikit-image — *plot_thresholding* |
| **B** | Limiarização multinível (Multi-Otsu) | `02_alunoB_multi_otsu.md` | scikit-image — *plot_multiotsu* |
| **C** | Limiarização adaptativa (local) | `03_alunoC_limiarizacao_adaptativa.md` | OpenCV — *py_thresholding* |

Os três leem antes a **fundamentação comum** (`00_fundamentacao_teorica.md`) e o
**comparativo** (`04_analise_comparativa.md`), que é apresentado em conjunto.

**Fio condutor que amarra os três blocos** (combinar antes de apresentar):

> A → "Otsu escolhe o melhor limiar **único**, e é imune a mudança global de luz."
> B → "Mas e se a cena tiver **mais de duas** populações? E note: a escolha de
> *qual* limiar usar muda tudo."
> C → "E se a luz variar **dentro** da imagem? Aí nenhum limiar único serve."
> Todos → "Nenhum venceu as quatro fases. A pergunta certa não é qual é melhor."

---

## 5.2 Roteiro minuto a minuto

### Abertura conjunta — 3 min

1. O que é limiarização (`00`, §0.1).
2. **O modelo $I = R \cdot L + n$** (`00`, §0.2). *Insistir nisto*: é a chave que
   explica todos os resultados do trabalho.
3. Apresentar a cena de teste e as 4 fases do vídeo.
   → Mostrar `resultados/video/mosaico_comparativo.mp4` **em velocidade normal,
   sem comentar**, só para criar a expectativa. 20 segundos.

---

### Aluno A — Otsu (11 min)

| Min | Conteúdo | Apoio visual |
|---|---|---|
| 0–2 | Os 5 tipos de `cv2.threshold`. "Mas qual T?" | `F_tipos_opencv_01_camera.png` |
| 2–4 | Otsu como agrupamento do histograma; $\omega_k$, $\mu_k$ | — |
| 4–7 | **A identidade $\sigma_T^2=\sigma_W^2+\sigma_B^2$** e por que maximizar $\sigma_B^2$ | **`B_otsu_04_sintetica.png`** |
| 7–8 | Custo $O(N)$ — uma passada; 0,16 ms no OpenCV | tabela §4.6 |
| 8–9 | A separabilidade $\eta$ como diagnóstico | tabela §1.2.6 |
| 9–11 | Os 7 métodos do `try_all_threshold` | `A_try_all_02_moedas.png` |

**Encerrar com o gancho:** mostrar o gráfico `V1_limiar_no_tempo.png` e dizer:
> "Na fase 3 a cena escurece e o Otsu baixa o limiar sozinho de 147 para 47 — o
> IoU nem se mexe, 0,944 → 0,947. Otsu é **invariante a mudança global de luz**.
> Agora olhem a fase 2: 0,139. O que mudou? A luz passou a variar **dentro** do
> quadro. Guardem isso — o colega C volta nesse ponto."

---

### Aluno B — Multi-Otsu (11 min)

| Min | Conteúdo | Apoio visual |
|---|---|---|
| 0–2 | Por que 2 classes não bastam (documento, radiografia, satélite) | `X2_histogramas_das_fases.png` |
| 2–4 | Critério generalizado $\sigma_B^2=\sum_k \omega_k(\mu_k-\mu_T)^2$ | — |
| 4–6 | **A explosão combinatória** — tabela de $\binom{255}{M-1}$ | tabela §2.2.2 |
| 6–8 | LUT $H(u,v)=S^2/P$ + **programação dinâmica** $O(ML^2)$ | — |
| 8–9 | Efeito de $M$; por que **não** se escolhe $M$ maximizando $\sigma_B^2$ | `C2_multiotsu_M_04_sintetica.png` |
| 9–11 | Resultados: $t_1$ contra $t_2$ | tabela §2.5 |

**História que funciona bem em sala** (é verdadeira):
> "A primeira versão do nosso código usava busca exaustiva. Rodou para $M=3$ em
> milissegundos. Para $M=4$ demorou minutos. Para $M=5$ **travou** — são 172
> milhões de combinações. Trocamos por programação dinâmica: caiu para 1,5 ms, e
> conferimos que dá **exatamente** os mesmos limiares do scikit-image em 12 casos
> de teste."

**Encerrar com o ponto alto:** na fase 2 do vídeo, $t_1 \to$ IoU 0,731 e
$t_2 \to$ IoU 0,138. Mesmo método, mesmo quadro, 5,3× de diferença, decidido só
por qual limiar se usa. Explicar a margem de segurança: o fundo cai para ~76,
que está acima de $t_1{\approx}52$ mas muito abaixo de $t_2{\approx}131$.

---

### Aluno C — Adaptativa (11 min)

| Min | Conteúdo | Apoio visual |
|---|---|---|
| 0–2 | Retomar $I=R\cdot L$: **nenhum $T$ único serve** | — |
| 2–4 | $T \to T(x,y)$; as fórmulas da média e da gaussiana | **`D_adaptativa_04_sintetica.png`** (mostrar o mapa $T(x,y)$ — é uma estimativa visual de $L$!) |
| 4–5 | **O papel de $C$** — a linha $C=0$ é ruído puro | `E_sensibilidade_04_sintetica.png` |
| 5–7 | Imagem integral: $O(1)$ por pixel, independente de $b$ | — |
| 7–8 | Sauvola: o desvio local substituindo o $C$ manual | — |
| 8–10 | **A regra do tamanho do bloco** e a tabela de IoU × bloco | `X3_efeito_bloco_e_C.png` |
| 10–11 | Contrapartida honesta: pior método na fase 4 (ruído) | tabela §4.5 |

**Encerrar com o ponto alto:** voltar ao mosaico em $t \approx 7{,}5$ s, pausado.
> "Mesmo quadro. Otsu: 0,139. Adaptativa ajustada: 0,956. Sete vezes melhor.
> Mas na fase 4, com ruído, a adaptativa é a **pior de todas** — 22 % dos
> pixels piscando a cada quadro. O motivo é o mesmo nos dois casos: ela decide
> olhando poucos pixels."

---

### Fechamento conjunto — 5 min

1. Mostrar `X4_sintese.png` e a tabela de guia de escolha (§4.7).
2. **As três frases** (§4.7):
   - o limiar global falha com variação **espacial**, não temporal;
   - a adaptativa troca robustez à luz por fragilidade ao ruído;
   - **nenhum método venceu as quatro fases**.
3. Limitações declaradas (§4.8) — mostra maturidade e antecipa a pergunta do
   professor.
4. Reproduzir o `mosaico_comparativo.mp4` inteiro (20 s), agora com a plateia
   sabendo o que olhar.

---

## 5.3 Perguntas prováveis e respostas

**"Por que vocês usaram um vídeo sintético em vez de um real?"**
Para ter o **gabarito exato dos 600 quadros** e poder medir IoU quadro a quadro,
e para controlar um fator de degradação por vez — sem isso só daria para comparar
visualmente. O código aceita vídeo real com `--video`; sem gabarito, continuam
válidos os gráficos de limiar, área e estabilidade temporal. A limitação está
declarada na §4.8.

**"Otsu não falha com iluminação variável?"**
Depende do tipo. Com variação **temporal/global** ele é praticamente invariante —
fase 3, IoU 0,943, com o limiar caindo de 147 para 47 sozinho. Com variação
**espacial** ele colapsa — fase 2, IoU 0,139. A distinção está na §4.4.

**"Por que a adaptativa foi pior que o Otsu em algumas fases?"**
Duas razões distintas. (i) Com $b=35$ o bloco era menor que os objetos e esvaziava
o interior deles — corrigido com $b=151$, que levou o IoU de 0,513 para 0,863.
(ii) Com ruído forte (fase 4), estimar o limiar com $b^2$ pixels é muito mais
sensível que estimá-lo com os 307 200 da imagem inteira. A segunda razão é
intrínseca ao método, a primeira era só ajuste.

**"Como escolher o número de classes $M$?"**
Não há critério automático: $\sigma_B^2$ cresce monotonicamente com $M$, então
maximizá-lo levaria a $M = 256$. Usa-se conhecimento do domínio, inspeção do
histograma, o "cotovelo" da curva, ou critérios de seleção de modelo (AIC/BIC).
Seção §2.4.

**"Vocês implementaram ou só chamaram a biblioteca?"**
Ambos. Otsu, Multi-Otsu (com programação dinâmica), adaptativa (com imagem
integral) e Sauvola estão implementados do zero em NumPy, em `src/limiarizacao.py`,
e foram **validados contra** scikit-image e OpenCV: limiares idênticos em Otsu e
Multi-Otsu, e 99,3–99,9 % de concordância na adaptativa — sendo que 100 % das
discordâncias ocorrem em pixels com $|I - T| < 0{,}5$, ou seja, puro arredondamento
(o OpenCV calcula em inteiro, nós em `float64`).

**"Dá para rodar em tempo real?"**
Sim, todos. De 0,16 ms (Otsu no OpenCV, ~6400 fps) a 22,8 ms (adaptativa
$b=151$, 44 fps) por quadro de 640×480. Tabela §4.6.

**"E se juntar os métodos?"**
Boa direção: está na §4.8. As duas ideias mais promissoras são suavizar antes do
Otsu (recomendação do próprio tutorial do OpenCV, deve recuperar bastante da fase
4) e dar memória temporal ao limiar,
$T_k = \alpha T_{k-1} + (1-\alpha) T_k^{\text{Otsu}}$, para cortar a cintilação.

---

## 5.4 Checklist antes de apresentar

- [ ] Rodar `python executar_tudo.py` e conferir que `resultados/` foi gerado
- [ ] Testar o `mosaico_comparativo.mp4` **no computador da apresentação**
      (é H.264; se o player reclamar, usar VLC)
- [ ] Ter as figuras-chave exportadas para os slides:
      `B_otsu_04_sintetica.png`, `C2_multiotsu_M_04_sintetica.png`,
      `D_adaptativa_04_sintetica.png`, `E_sensibilidade_04_sintetica.png`,
      `V1_limiar_no_tempo.png`, `V2_iou_no_tempo.png`, `X3_efeito_bloco_e_C.png`,
      `X4_sintese.png`
- [ ] Cada aluno sabe **derivar no quadro** a fórmula central do seu tema
      (A: $\sigma_T^2=\sigma_W^2+\sigma_B^2$ · B: $\sigma_B^2=\sum H - \mu_T^2$ ·
      C: $T(x,y)=m(x,y)-C$ e a imagem integral)
- [ ] Combinar as três frases de transição entre os blocos (§5.1)
- [ ] Ensaiar o cronômetro: 11 min passa muito rápido
