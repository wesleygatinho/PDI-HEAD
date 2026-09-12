# 1. ALUNO A — Limiarização global e o método de Otsu

> **Fonte base:** <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html>
> **Tempo sugerido de apresentação:** 10–12 min
> **Figuras de apoio:** `resultados/graficos/B_otsu_*.png`, `A_try_all_*.png`, `F_tipos_opencv_*.png`

---

## 1.1 Limiarização simples: os cinco tipos do OpenCV

Antes do problema de *escolher* o limiar, vale fixar o que se faz *com* ele.
O OpenCV implementa cinco operações de limiarização em
`cv2.threshold(src, T, maxval, tipo)`:

| Tipo | Definição | Uso típico |
|---|---|---|
| `THRESH_BINARY` | $dst = maxval$ se $src > T$, senão $0$ | segmentação padrão |
| `THRESH_BINARY_INV` | $dst = 0$ se $src > T$, senão $maxval$ | objeto escuro sobre fundo claro |
| `THRESH_TRUNC` | $dst = T$ se $src > T$, senão $src$ | saturação / limitação de brilho |
| `THRESH_TOZERO` | $dst = src$ se $src > T$, senão $0$ | preserva intensidade do objeto |
| `THRESH_TOZERO_INV` | $dst = 0$ se $src > T$, senão $src$ | máscara complementar |

Apenas os dois primeiros produzem imagem binária; os outros três são
transformações pontuais que preservam níveis de cinza. Ver
`resultados/graficos/F_tipos_opencv_*.png`.

**O problema central:** em todos eles $T$ é um número que o usuário precisa
fornecer. Um valor escolhido à mão funciona para *uma* imagem e falha na
seguinte. É isso que os métodos automáticos resolvem.

---

## 1.2 O método de Otsu (1979)

### 1.2.1 A ideia

Otsu reformula a escolha do limiar como um problema de **análise discriminante**:

> O melhor limiar é aquele que separa os pixels em duas classes
> **internamente mais homogêneas** e **mutuamente mais distintas** possível.

Trata-se o histograma normalizado $p(i)$ como uma distribuição de probabilidade
e procura-se o corte que melhor a particiona em dois grupos — é literalmente um
algoritmo de agrupamento (*clustering*) unidimensional com $k = 2$.

### 1.2.2 Formulação

Para um limiar candidato $t$, definem-se as classes
$C_0 = \{0, \dots, t\}$ e $C_1 = \{t+1, \dots, L-1\}$.

**Pesos (probabilidades a priori das classes):**

$$
\omega_0(t) = \sum_{i=0}^{t} p(i)
\qquad\qquad
\omega_1(t) = \sum_{i=t+1}^{L-1} p(i) = 1 - \omega_0(t)
$$

**Médias das classes:**

$$
\mu_0(t) = \frac{1}{\omega_0(t)}\sum_{i=0}^{t} i\,p(i)
\qquad\qquad
\mu_1(t) = \frac{1}{\omega_1(t)}\sum_{i=t+1}^{L-1} i\,p(i)
$$

**Variâncias das classes:**

$$
\sigma_0^2(t) = \frac{1}{\omega_0(t)}\sum_{i=0}^{t}(i-\mu_0)^2 p(i)
\qquad
\sigma_1^2(t) = \frac{1}{\omega_1(t)}\sum_{i=t+1}^{L-1}(i-\mu_1)^2 p(i)
$$

Note a identidade que será usada adiante:

$$
\mu_T = \omega_0\mu_0 + \omega_1\mu_1
$$

### 1.2.3 As duas variâncias e a identidade fundamental

**Variância intraclasse** (*within-class*) — mede a dispersão **dentro** de cada
classe; queremos que seja **pequena**:

$$
\boxed{\;\sigma_W^2(t) = \omega_0(t)\,\sigma_0^2(t) + \omega_1(t)\,\sigma_1^2(t)\;}
$$

**Variância interclasse** (*between-class*) — mede o quanto as médias das classes
se afastam da média global; queremos que seja **grande**:

$$
\boxed{\;\sigma_B^2(t) = \omega_0(t)\bigl(\mu_0(t)-\mu_T\bigr)^2
                       + \omega_1(t)\bigl(\mu_1(t)-\mu_T\bigr)^2\;}
$$

**Teorema (decomposição da variância).** Para todo $t$:

$$
\boxed{\;\sigma_T^2 = \sigma_W^2(t) + \sigma_B^2(t) = \text{constante}\;}
$$

*Demonstração.* Parte-se da variância total e separa-se o somatório nas duas
classes:

$$
\sigma_T^2 = \sum_{i \in C_0}(i-\mu_T)^2 p(i) + \sum_{i \in C_1}(i-\mu_T)^2 p(i)
$$

No primeiro somatório escreve-se $i - \mu_T = (i - \mu_0) + (\mu_0 - \mu_T)$ e
expande-se o quadrado:

$$
\sum_{i \in C_0}(i-\mu_T)^2 p(i)
= \underbrace{\sum_{i \in C_0}(i-\mu_0)^2 p(i)}_{\omega_0 \sigma_0^2}
+ 2(\mu_0-\mu_T)\underbrace{\sum_{i \in C_0}(i-\mu_0)p(i)}_{=\,\omega_0\mu_0 - \omega_0\mu_0 = 0}
+ (\mu_0-\mu_T)^2\underbrace{\sum_{i \in C_0}p(i)}_{\omega_0}
$$

O termo cruzado se anula **por construção**, pois $\mu_0$ é justamente a média de
$C_0$. Logo o primeiro somatório vale $\omega_0\sigma_0^2 + \omega_0(\mu_0-\mu_T)^2$.
Repetindo para $C_1$ e somando:

$$
\sigma_T^2 = \underbrace{\bigl[\omega_0\sigma_0^2 + \omega_1\sigma_1^2\bigr]}_{\sigma_W^2}
           + \underbrace{\bigl[\omega_0(\mu_0-\mu_T)^2 + \omega_1(\mu_1-\mu_T)^2\bigr]}_{\sigma_B^2}
\qquad \blacksquare
$$

**Consequência prática — e é o coração do método:** $\sigma_T^2$ depende só da
imagem, **não de $t$**. Portanto

$$
\text{maximizar } \sigma_B^2(t) \;\equiv\; \text{minimizar } \sigma_W^2(t)
$$

Como $\sigma_B^2$ envolve apenas médias (momentos de 1ª ordem) e $\sigma_W^2$
exigiria calcular variâncias de cada classe, **maximizar $\sigma_B^2$ é muito mais
barato**. Essa é a escolha de Otsu.

> A figura `B_otsu_*.png` mostra as duas curvas: o pico de $\sigma_B^2$ coincide
> exatamente com o vale de $\sigma_W^2$, e a soma é a reta horizontal $\sigma_T^2$.

### 1.2.4 Forma simplificada e o critério final

Usando $\mu_T = \omega_0\mu_0 + \omega_1\mu_1$ e $\omega_0 + \omega_1 = 1$,
obtém-se $\mu_0 - \mu_T = \omega_1(\mu_0-\mu_1)$ e $\mu_1 - \mu_T = \omega_0(\mu_1-\mu_0)$.
Substituindo:

$$
\sigma_B^2 = \omega_0\omega_1^2(\mu_0-\mu_1)^2 + \omega_1\omega_0^2(\mu_0-\mu_1)^2
           = \omega_0\omega_1(\omega_0+\omega_1)(\mu_0-\mu_1)^2
$$

$$
\boxed{\;\sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\,\bigl[\mu_0(t)-\mu_1(t)\bigr]^2\;}
$$

Leitura da fórmula: o limiar ótimo equilibra **dois fatores em conflito** —
o produto $\omega_0\omega_1$ (máximo quando as classes têm o mesmo tamanho) e a
separação $(\mu_0-\mu_1)^2$ (máxima quando as médias estão distantes).

**Critério de Otsu:**

$$
\boxed{\;t^{*} = \arg\max_{0 \le t < L-1} \sigma_B^2(t)\;}
$$

### 1.2.5 Cálculo eficiente

Definindo o momento acumulado $\mu(t) = \sum_{i \le t} i\,p(i)$, chega-se a uma
forma que usa apenas somas acumuladas:

$$
\sigma_B^2(t) = \frac{\bigl[\mu_T\,\omega_0(t) - \mu(t)\bigr]^2}
                     {\omega_0(t)\,\bigl[1-\omega_0(t)\bigr]}
$$

Com `np.cumsum` obtêm-se $\omega_0(t)$ e $\mu(t)$ para **todos** os $t$ de uma
vez. O custo total é:

| Etapa | Custo |
|---|---|
| Construir o histograma | $O(N)$, $N$ = nº de pixels |
| Avaliar $\sigma_B^2$ em todos os $t$ | $O(L)$, $L = 256$ |
| **Total** | $O(N + L) \approx O(N)$ |

Ou seja: **uma única passada pela imagem**. Medido no experimento: **0,53 ms**
para uma imagem 640×480 na implementação própria em NumPy (0,16 ms na do
OpenCV, que é em C++). Ver `resultados/metricas/tempos_execucao.csv`.

### 1.2.6 Medida de separabilidade

Otsu propõe, no mesmo artigo, um indicador de **confiabilidade** do limiar:

$$
\eta = \frac{\sigma_B^2(t^{*})}{\sigma_T^2} \in [0,\,1]
$$

$\eta \to 1$: histograma claramente bimodal, limiar confiável.
$\eta$ baixo: histograma unimodal, e o limiar global é pouco significativo —
o método devolve *algum* valor, mas sem sentido físico.

Valores medidos neste trabalho:

| Imagem | $t^{*}$ | $\eta$ | Leitura |
|---|---|---|---|
| `01_camera` | 102 | **0,857** | bimodal, ótimo caso |
| `02_moedas` | 107 | 0,756 | bom |
| `03_pagina` | 157 | 0,719 | razoável |
| `04_sintetica` (luz desigual) | 125 | **0,596** | o gradiente já degrada a bimodalidade |

O $\eta$ é a forma de **diagnosticar antecipadamente** que o limiar global vai
falhar — um recurso valioso que raramente é usado na prática.

### 1.2.7 Hipóteses e limitações

Otsu assume, implicitamente:

1. histograma **bimodal**;
2. as duas classes têm **variâncias comparáveis** (o critério penaliza classes de
   dispersão muito diferente);
3. as classes têm **tamanhos comparáveis** — o fator $\omega_0\omega_1$ empurra o
   limiar na direção de partições equilibradas, e por isso o método **enviesa**
   quando o objeto é muito pequeno;
4. **iluminação espacialmente uniforme**.

A hipótese 4 é a que quebra de forma mais espetacular no vídeo (seção 4).

---

## 1.3 Os outros métodos globais (`try_all_threshold`)

A função `skimage.filters.try_all_threshold` aplica sete métodos globais de uma
vez — é a ferramenta de diagnóstico rápido recomendada pelo tutorial básico.
Ver `resultados/graficos/A_try_all_*.png`.

| Método | Critério | Comentário |
|---|---|---|
| **Mean** | $T = \mu_T$ | o mais simples possível; referência ingênua |
| **Isodata** (Ridler–Calvard) | ponto fixo de $T = \tfrac{\mu_0(T)+\mu_1(T)}{2}$ | iterativo; geralmente muito próximo de Otsu |
| **Otsu** | $\max \sigma_B^2$ | análise discriminante |
| **Li** | mínima **entropia cruzada** entre a imagem original e a binarizada | bom para objetos pequenos |
| **Yen** | máxima **correlação entrópica** entre as classes | baseado em teoria da informação |
| **Minimum** | suaviza o histograma até restarem dois máximos; $T$ = vale entre eles | exige histograma realmente bimodal |
| **Triangle** | traça a reta do pico ao extremo do histograma; $T$ = ponto de **máxima distância perpendicular** à reta | geométrico; bom para histogramas **unimodais** com cauda |

**Quando divergem:** em histogramas bimodais e equilibrados, Otsu, Isodata e Li
caem praticamente no mesmo valor. As diferenças aparecem quando o objeto é
pequeno (Li e Triangle tendem a se sair melhor, pois não são atraídos pelo
equilíbrio de tamanhos) ou quando o histograma é unimodal (Triangle é o único
projetado para esse caso).

---

## 1.4 Otsu no OpenCV e o pré-filtro gaussiano

```python
# Otsu no OpenCV: o T passado é ignorado, o método calcula o seu
T, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

O tutorial do OpenCV destaca uma prática importante: **suavizar antes de
limiarizar**.

```python
suave = cv2.GaussianBlur(img, (5, 5), 0)
T, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

**Justificativa matemática:** o ruído alarga os dois modos do histograma e
preenche o vale entre eles. A filtragem gaussiana reduz a variância do ruído por
um fator aproximado de $1/(4\pi\sigma^2)$ por pixel, o que **estreita os modos**,
**aprofunda o vale** e aumenta $\eta$ — tornando o limiar mais estável. O custo é
a perda de detalhes finos e o arredondamento de cantos.

Esse efeito aparece na **fase 4** do vídeo: com ruído forte, o IoU do Otsu cai
de 0,944 para 0,834 e sua instabilidade temporal salta de 0,28 % para
**3,57 %** — exatamente o cenário em que o pré-filtro seria indicado.

---

## 1.5 O que apresentar (resumo do Aluno A)

1. O problema do limiar fixo: mostrar `F_tipos_opencv_*.png` e perguntar "qual T?"
2. A ideia de Otsu como agrupamento do histograma.
3. **A identidade $\sigma_T^2 = \sigma_W^2 + \sigma_B^2$** — mostrar
   `B_otsu_04_sintetica.png`: pico e vale coincidentes, soma constante.
4. Custo $O(N)$: por que isso importa em vídeo (0,88 ms/quadro → 1100 fps).
5. O $\eta$ como diagnóstico.
6. **O ponto alto — o resultado mais bonito do trabalho:** na fase 3 do vídeo a
   cena escurece progressivamente e o limiar de Otsu **desce sozinho de 147 para
   47** (um fator de 3×), enquanto o IoU permanece praticamente inalterado
   (0,944 → 0,947). Ou seja, Otsu é **invariante a mudanças globais de
   iluminação**. Mas na fase 2, em que a luz varia **no espaço**, ele colapsa
   para IoU 0,139. Gancho para o Aluno C.
