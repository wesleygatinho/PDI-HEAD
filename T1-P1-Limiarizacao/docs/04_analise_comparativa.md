# 4. Análise comparativa — diferença de comportamento entre as técnicas

> Esta é a seção que responde diretamente ao enunciado:
> *"apliquem um threshold em uma imagem e em um vídeo curto de 20 s, analisem a
> diferença de comportamento entre as técnicas."*

---

## 4.1 Metodologia

### Material

| Item | Descrição |
|---|---|
| **Imagens** | 3 reais do acervo scikit-image (`camera`, `coins`, `page`) + 1 sintética 640×480 com **gabarito conhecido**, degradada por gradiente diagonal de iluminação ($L$ de 1,00 a 0,38) e ruído gaussiano $\sigma = 3$ |
| **Vídeo** | 640×480, 30 fps, **20 s = 600 quadros**, com gabarito analítico quadro a quadro |

### Por que um vídeo sintético

Usar uma cena gerada permite ter o **gabarito exato de cada um dos 600 quadros**,
o que torna possível medir IoU/F1 quadro a quadro em vez de apenas comparar
visualmente. Além disso, permite **controlar isoladamente** cada fator de
degradação — o que é essencial para atribuir causas, e não só observar efeitos.

> Os scripts aceitam um vídeo próprio via `python src/exp02_video.py --video meu.mp4`.
> Nesse caso os gráficos de limiar, área e estabilidade continuam válidos; apenas
> as métricas que dependem de gabarito (IoU/F1) ficam indisponíveis.

### As quatro fases do vídeo

Cada fase dura 5 s e isola **um** fator:

| Fase | Tempo | O que muda | Hipótese testada |
|---|---|---|---|
| **1 — Uniforme** | 0–5 s | nada (referência) | comportamento em condições ideais |
| **2 — Holofote** | 5–10 s | $L$ varia **no espaço** (holofote varrendo) | quebra a hipótese dos métodos globais |
| **3 — Escurecimento** | 10–15 s | $L$ varia **no tempo** (ganho de 1,00 → 0,32) | separa "limiar automático" de "limiar fixo" |
| **4 — Ruído** | 15–20 s | $\sigma$ de 4 → 30 e contraste de 1,00 → 0,45 | penaliza os métodos locais |

O ponto metodológico importante é a **separação das fases 2 e 3**: ambas mudam a
iluminação, mas apenas uma delas (a 2) muda $L$ *dentro* do quadro. É essa
distinção que explica todos os resultados.

---

## 4.2 Resultado 1 — Imagem estática

`resultados/metricas/RESUMO_imagem.md` · figura `G_comparativo_04_sintetica.png`

| Método | Acurácia | Precisão | Revocação | F1/Dice | **IoU** | MCC |
|---|---|---|---|---|---|---|
| Limiar fixo T=127 | 0,808 | 0,388 | 0,950 | 0,551 | 0,380 | 0,529 |
| Otsu global (T=125) | 0,825 | 0,410 | 0,948 | 0,572 | 0,401 | 0,550 |
| Multi-Otsu, classe escura (T=74) | 0,950 | **1,000** | 0,593 | 0,744 | 0,593 | 0,749 |
| Multi-Otsu, escura+média (T=148) | 0,620 | 0,242 | 0,971 | 0,387 | 0,240 | 0,356 |
| **Adaptativa média (b=35, C=10)** | **0,975** | 0,998 | **0,803** | **0,890** | **0,801** | **0,882** |
| Adaptativa gaussiana (b=35, C=10) | 0,949 | 0,996 | 0,586 | 0,738 | 0,585 | 0,743 |
| Sauvola (b=35, k=0,2) | 0,973 | 1,000 | 0,780 | 0,877 | 0,780 | 0,870 |

**Leituras:**

1. **Otsu quase não melhora sobre o limiar fixo** (0,401 contra 0,380). Escolher o
   limiar de forma ótima não adianta quando **nenhum limiar único é adequado**.
   Este é o argumento mais forte contra os métodos globais.
2. **A precisão separa as famílias.** Os métodos locais têm precisão ~1,000 (quase
   nenhum falso positivo) e revocação em torno de 0,8; os globais têm revocação
   ~0,95 mas precisão ~0,4 — eles classificam todo o canto escuro da imagem como
   objeto. São **modos de falha qualitativamente diferentes**, e a acurácia
   sozinha esconderia isso.
3. **Gaussiana pior que média** (0,585 contra 0,801) porque, com $b=35$, o $\sigma$
   padrão do OpenCV dá uma vizinhança efetiva de apenas ~11 px, esvaziando mais o
   interior dos objetos (ver seção 3.2.2).

---

## 4.3 Resultado 2 — Vídeo: o limiar em movimento

Figura `V1_limiar_no_tempo.png`

Este gráfico é o mais informativo do trabalho. Ele mostra o valor do limiar
escolhido por cada método ao longo dos 20 s.

| Fase | Comportamento observado |
|---|---|
| 1 | Todas as curvas planas — nada muda na cena. |
| 2 | Queda abrupta em $t = 5$ s: o holofote reduz a luminância global, e todos os métodos automáticos reagem. |
| 3 | **Descida linear e limpa**: o Otsu cai de **147 para 47** ao longo de 5 s, acompanhando o escurecimento. O limiar fixo permanece na horizontal em 127. |
| 4 | Subida com degraus: o ruído alarga o histograma e desloca o ótimo. |

O trecho da fase 3 é o que melhor demonstra o valor de um método automático:
**o limiar variou por um fator de 3× e a qualidade não mudou** (IoU do Otsu:
0,944 no início da fase, 0,947 no fim). Enquanto isso, o limiar fixo, que não se
move, produz a "torre" bem visível em `V2_iou_no_tempo.png`: ele atinge
**IoU 0,997 em $t = 12{,}73$ s** — o melhor resultado de todo o experimento! — no
instante exato em que o brilho da cena, descendo, coincide por acaso com
$T = 127$; e dois segundos e meio depois está em **0,120**.

Esse pico é a melhor ilustração possível do argumento: um limiar fixo não é
"ruim", ele é **certo por acidente e errado o resto do tempo**. O que um método
automático oferece não é um limiar melhor em um dado instante, é a garantia de
continuar certo quando as condições mudam.

---

## 4.4 Resultado 3 — Vídeo: qualidade por fase

`resultados/metricas/RESUMO_video.md` · figuras `V2_iou_no_tempo.png`, `V5_resumo_por_fase.png`

**IoU médio por fase** (o melhor de cada coluna em negrito):

| Método | Fase 1<br>uniforme | Fase 2<br>holofote | Fase 3<br>escurecim. | Fase 4<br>ruído | Global |
|---|---|---|---|---|---|
| Limiar fixo T=127 | 0,657 | 0,140 | 0,636 | 0,634 | 0,516 |
| Otsu global | 0,944 | 0,139 | 0,943 | **0,834** | 0,715 |
| Multi-Otsu $t_1$ (classe escura) | 0,592 | 0,731 | 0,600 | 0,639 | 0,641 |
| Multi-Otsu $t_2$ (escura+média) | 0,966 | 0,138 | **0,965** | 0,626 | 0,674 |
| Adaptativa (b=35) | 0,621 | 0,528 | 0,561 | 0,341 | 0,513 |
| **Adaptativa ajustada (b=151)** | **0,971** | **0,956** | 0,950 | 0,575 | **0,863** |
| Sauvola (b=35) | 0,787 | 0,757 | 0,756 | 0,627 | 0,732 |

**As quatro conclusões centrais:**

### (a) Variação ESPACIAL de iluminação destrói os métodos globais

Fase 2: Otsu cai de 0,944 para **0,139** — uma perda de 85 % da qualidade. E o
limiar fixo cai para 0,140, praticamente o mesmo valor. **A escolha ótima do
limiar global não compra nada** quando a hipótese de uniformidade é violada; o
problema não está em *qual* $T$, está em existir um $T$ único.

No mesmo quadro, a adaptativa ajustada mantém **0,956** — uma diferença de **6,9×**.
Ver o mosaico em $t \approx 7{,}5$ s: onde o Otsu produz uma mancha preta cobrindo
toda a região do holofote, a adaptativa reproduz o gabarito quase exatamente.

### (b) Variação TEMPORAL de iluminação não é problema para nenhum método automático

Fase 3: Otsu 0,943 e Multi-Otsu $t_2$ 0,965 — praticamente os mesmos valores da
fase 1. Apenas o **limiar fixo** sofre (0,636).

Esta é a distinção conceitual mais importante do trabalho e costuma ser
confundida: dizer que "Otsu falha com iluminação variável" é **impreciso**.
Otsu falha com iluminação **espacialmente** não uniforme; a mudanças globais de
brilho ele é essencialmente **invariante**, porque recalcula o limiar a cada
quadro sobre o histograma daquele quadro.

### (c) Ruído inverte o ranking

Fase 4 é a única em que os métodos **globais** vencem: Otsu 0,834 contra 0,575
da adaptativa ajustada e 0,341 da adaptativa padrão.

A razão é estrutural. O limiar global é estimado a partir de $N = 307\,200$
pixels; pela lei dos grandes números, o ruído praticamente se cancela na
estimativa. O limiar local é estimado a partir de $b^2$ pixels — para $b = 35$,
apenas 1 225 — e herda a variância do ruído. Além disso, em regiões homogêneas
$I \approx m$, e a decisão passa a depender essencialmente do ruído.

### (d) A escolha do limiar dentro do Multi-Otsu muda tudo

Compare as duas linhas do Multi-Otsu na fase 2: $t_1$ dá 0,731 e $t_2$ dá 0,138.
**O mesmo método, o mesmo quadro, o mesmo histograma** — 5,3× de diferença,
decidido apenas por qual dos dois limiares se usa como corte binário.

A explicação é quantitativa: o holofote rebaixa o fundo até
$0{,}34 \times 225 \approx 76$, valor que fica **acima** de $t_1 \approx 52$ (o
fundo escapa de ser classificado como objeto) mas bem **abaixo** de
$t_2 \approx 131$ (todo o fundo escuro vira objeto). A classe mais escura tem
uma margem de segurança que as outras não têm.

---

## 4.5 Resultado 4 — Estabilidade temporal (só existe em vídeo)

Figura `V3_instabilidade_no_tempo.png`

Fração média de pixels que trocam de classe entre quadros consecutivos:

| Método | Fase 1 | Fase 2 | Fase 3 | **Fase 4** | Global |
|---|---|---|---|---|---|
| Limiar fixo | 1,01 % | 1,27 % | 1,86 % | 2,53 % | 1,67 % |
| Otsu global | 0,28 % | 1,29 % | 0,78 % | 3,57 % | 1,48 % |
| **Multi-Otsu $t_1$** | **0,23 %** | **0,29 %** | **0,30 %** | **1,81 %** | **0,66 %** |
| Multi-Otsu $t_2$ | 0,29 % | 1,23 % | 0,81 % | 19,38 % | 5,44 % |
| Adaptativa (b=35) | 0,63 % | 0,70 % | 0,70 % | **21,95 %** | 6,00 % |
| Adaptativa ajustada (b=151) | 0,29 % | 0,30 % | 0,38 % | 16,58 % | 4,39 % |
| Sauvola | 0,51 % | 0,78 % | 0,69 % | 4,34 % | 1,58 % |

Como o conteúdo da cena muda muito pouco entre dois quadros vizinhos, esses
valores medem **cintilação do método**, não movimento real.

**Este é um critério que a análise de imagem estática não revela.** Na fase 4, a
adaptativa faz **21,95 %** dos pixels piscarem a cada quadro — um resultado
visualmente insuportável, que inviabiliza qualquer processamento posterior
(rastreamento, contagem de objetos, OCR). O Multi-Otsu $t_1$, com 1,81 %, é
**doze vezes** mais estável.

Note também que Sauvola tem instabilidade 5,1× menor que a adaptativa do OpenCV
(4,34 % contra 21,95 %) no mesmo cenário, exatamente pelo mecanismo previsto na
seção 3.4: em região homogênea $s \to 0$ e o limiar desce para $m(1-k)$, criando
sozinho a banda morta que impede a decisão de seguir o ruído.

---

## 4.6 Resultado 5 — Custo computacional

Figura `V6_custo_por_quadro.png` · `resultados/metricas/tempos_execucao.csv`

Tempo médio por quadro de 640×480 (implementações próprias em NumPy, exceto onde
indicado):

| Método | Tempo | Taxa máxima | Tempo real a 30 fps? |
|---|---|---|---|
| Limiar fixo | 0,34 ms | 2 970 fps | sim |
| Otsu (OpenCV, C++) | 0,16 ms | 6 370 fps | sim |
| Otsu (próprio) | 0,88 ms | 1 140 fps | sim |
| Multi-Otsu $M=3$ | 1,20 ms | 830 fps | sim |
| Multi-Otsu $M=4$ (skimage) | 32,8 ms | 31 fps | no limite |
| Adaptativa $b=35$ | 4,62 ms | 216 fps | sim |
| **Adaptativa $b=151$** | **22,8 ms** | **44 fps** | sim, com folga pequena |
| Sauvola | 3,73 ms | 268 fps | sim |

Observações:

- Todos os métodos são compatíveis com processamento em tempo real a 30 fps.
- O salto de $M=3$ para $M=4$ no Multi-Otsu é de **31×** — a complexidade
  $O(M L^2)$ da programação dinâmica é bem-comportada, mas a implementação do
  scikit-image usa busca com LUT, cujo custo cresce muito mais rápido.
- A adaptativa gaussiana escala com $b$ (filtro separável, $O(Nb)$). Se o custo
  incomodar, a versão da **média** com imagem integral é $O(N)$ **independente de
  $b$** — mesma robustez à iluminação, custo constante.

---

## 4.7 Síntese: guia de escolha

Figura `X4_sintese.png`

| Situação | Método recomendado | Justificativa |
|---|---|---|
| Histograma bimodal, luz uniforme | **Otsu** | IoU 0,944 a 0,88 ms; nada supera o custo/benefício |
| Brilho global varia (dia/noite, auto-exposição) | **Otsu** ou **Multi-Otsu** por quadro | invariantes a mudança global (fase 3) |
| 3+ populações, luz uniforme | **Multi-Otsu** | melhor global do trabalho (IoU 0,966) |
| **Sombra, holofote, vinheta, documento digitalizado** | **Adaptativa** com $b$ ajustado | único que segue $L(x,y)$: 0,956 contra 0,139 |
| Objetos grandes e maciços | **Global**, ou adaptativa com $b >$ objeto | a adaptativa esvazia o interior |
| Ruído forte / baixo contraste | **Global**, ou **Sauvola** | a média de $N$ pixels cancela ruído; $b^2$ não |
| Vídeo, estabilidade é crítica | **Multi-Otsu $t_1$** | 0,66 % de instabilidade, a menor de todas |
| Texto sobre fundo irregular | **Sauvola** | equilibra robustez à luz e ao ruído |

### As três frases que resumem o trabalho

1. **O limiar global falha com variação espacial de iluminação, não com variação
   temporal.** Fase 2 (0,139) contra fase 3 (0,943) para o mesmo método.
2. **A limiarização adaptativa troca robustez à iluminação por fragilidade ao
   ruído.** Fase 2: 0,956. Fase 4: 0,575. O ganho e a perda vêm da mesma causa —
   estimar o limiar a partir de poucos pixels.
3. **Nenhum método venceu as quatro fases.** O melhor variou em cada uma
   (adaptativa ajustada, adaptativa ajustada, Multi-Otsu $t_2$, Otsu). A escolha
   não é sobre qual método é "melhor", é sobre **qual hipótese a sua cena
   satisfaz**.

---

## 4.8 Limitações deste estudo e trabalhos futuros

**Limitações, declaradas abertamente:**

- O vídeo é **sintético**. Isso foi uma escolha deliberada (permite gabarito
  exato e controle isolado dos fatores), mas cenas reais têm textura, desfoque
  de movimento, sombras suaves e compressão mais agressiva. Os *mecanismos*
  identificados valem em geral; os *números* são específicos desta cena.
- As comparações usam **um único ajuste de parâmetros por método**. A adaptativa
  ajustada ($b=151$) foi otimizada nesta mesma cena, o que a favorece um pouco.
- Não há **pós-processamento morfológico** (abertura/fechamento), que na prática
  removeria boa parte do ruído da fase 4 e mudaria o ranking dessa fase.

**Extensões naturais:**

1. Pré-filtro gaussiano antes do Otsu (recomendado pelo tutorial do OpenCV) e
   medir quanto ele recupera na fase 4.
2. Limiar com **memória temporal**: $T_k = \alpha T_{k-1} + (1-\alpha)T_k^{\text{Otsu}}$,
   para reduzir a cintilação sem perder a adaptação.
3. Abertura morfológica 3×3 após a adaptativa, e recomputar a tabela 4.5.
4. Comparar com métodos baseados em região (crescimento de regiões, watershed) e
   com segmentação por aprendizado profundo, para quantificar o que a
   limiarização deixa na mesa.
