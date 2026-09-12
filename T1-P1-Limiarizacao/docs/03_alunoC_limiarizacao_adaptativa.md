# 3. ALUNO C — Limiarização adaptativa (local)

> **Fonte base:** <https://docs.opencv.org/4.4.0/d7/d4d/tutorial_py_thresholding.html>
> **Tempo sugerido de apresentação:** 10–12 min
> **Figuras de apoio:** `resultados/graficos/D_adaptativa_*.png`, `E_sensibilidade_*.png`, `X3_efeito_bloco_e_C.png`

---

## 3.1 A motivação, em uma linha

Retomando o modelo de formação da imagem:

$$
I(x,y) = R(x,y)\cdot L(x,y)
$$

Se a iluminação $L$ **varia dentro da cena**, não existe nenhum valor único de
$T$ que funcione: o fundo de uma região sombreada pode ser mais escuro que o
objeto de uma região iluminada. Nenhum método global resolve isso — **nem Otsu,
nem Multi-Otsu, nem nenhum outro**, porque todos decidem a partir do histograma,
que já descartou a informação espacial.

A solução é abandonar a ideia de um limiar único:

$$
\boxed{\;T \;\longrightarrow\; T(x,y)\;}
$$

Calcula-se **um limiar para cada pixel**, a partir apenas da sua vizinhança.
Como $L(x,y)$ varia devagar no espaço, dentro de uma janela pequena ela é
aproximadamente constante — e ali um limiar local é válido.

> Em linguagem de frequências: $\log I = \log R + \log L$, e a iluminação é a
> componente de **baixa frequência espacial**. A limiarização adaptativa estima
> essa componente com um filtro passa-baixas (a média local) e a subtrai. É,
> essencialmente, um **filtro passa-altas seguido de um limiar fixo**.

---

## 3.2 Formulação

### 3.2.1 Definição geral

$$
g(x,y) =
\begin{cases}
\text{maxValue} & \text{se } I(x,y) > T(x,y)\\
0 & \text{caso contrário}
\end{cases}
\qquad\quad
T(x,y) = f\bigl(B_{b}(x,y)\bigr) - C
$$

onde $B_b(x,y)$ é a janela $b \times b$ centrada em $(x,y)$, $f$ é uma estatística
local e $C$ é uma constante.

### 3.2.2 Os dois métodos do OpenCV

```python
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
```

**`ADAPTIVE_THRESH_MEAN_C`** — média aritmética da vizinhança:

$$
T(x,y) = \frac{1}{b^2}\sum_{(i,j)\in B_b(x,y)} I(i,j) \;-\; C
$$

**`ADAPTIVE_THRESH_GAUSSIAN_C`** — média ponderada por uma gaussiana:

$$
T(x,y) = \sum_{(i,j)\in B_b(x,y)} w(i,j)\,I(i,j) \;-\; C,
\qquad
w(i,j) \propto \exp\!\left(-\frac{(i-x)^2+(j-y)^2}{2\sigma^2}\right)
$$

com $\sum w = 1$. Quando $\sigma$ não é informado, o OpenCV usa a regra padrão
de `getGaussianKernel`:

$$
\sigma = 0{,}3\cdot\left(\frac{b-1}{2} - 1\right) + 0{,}8
$$

Para $b = 35$ isso dá $\sigma = 5{,}6$ — note que a **vizinhança efetiva** da
versão gaussiana é bem **menor** que a janela nominal de 35 px. Essa é uma
armadilha prática: `blockSize=35` com a gaussiana não "enxerga" 35 px, enxerga
cerca de $\pm 2\sigma \approx 11$ px. Isso explica por que, nos experimentos, a
versão gaussiana esvaziou os objetos grandes mais do que a versão da média
(IoU 0,585 contra 0,801 na imagem estática).

**Restrições:** `blockSize` deve ser **ímpar** e $\ge 3$; `thresholdType` deve ser
`THRESH_BINARY` ou `THRESH_BINARY_INV`.

### 3.2.3 O papel da constante $C$

$C$ não é um detalhe de implementação — é **essencial**. Sem ele ($C = 0$), em
uma região homogênea vale $I(x,y) \approx m(x,y)$, e o teste $I > m$ vira
essencialmente o **sinal do ruído**: o resultado é ruído binário puro.

$C$ estabelece uma **banda morta**: só é declarado objeto o pixel que for pelo
menos $C$ níveis mais escuro que sua vizinhança.

$$
\text{objeto} \iff m(x,y) - I(x,y) > C
$$

A figura `E_sensibilidade_*.png` mostra isso de forma inequívoca: a linha $C = 0$
é ruído em toda a imagem; a partir de $C \approx 5$ o fundo limpa.

---

## 3.3 Implementação eficiente: a imagem integral

A média em janela $b \times b$ custaria $O(N b^2)$ se calculada ingenuamente —
para $b = 151$ seriam ~23 000 somas **por pixel**. A **imagem integral**
(*summed-area table*, Crow 1984; aplicada à limiarização por Bradley & Roth 2007)
reduz isso a $O(1)$ por pixel:

$$
II(x,y) = \sum_{x' \le x}\;\sum_{y' \le y} I(x',y')
$$

A soma em qualquer retângulo sai com **quatro acessos**:

$$
\sum_{\substack{x_1 \le x \le x_2 \\ y_1 \le y \le y_2}} I
= II(x_2,y_2) - II(x_1{-}1,y_2) - II(x_2,y_1{-}1) + II(x_1{-}1,y_1{-}1)
$$

| Abordagem | Custo total | Depende de $b$? |
|---|---|---|
| Soma direta | $O(N b^2)$ | sim, quadraticamente |
| **Imagem integral** | $O(N)$ | **não** |

É por isso que a limiarização adaptativa pela média é viável em tempo real:
o tamanho da janela **não afeta o tempo de execução**. (A versão gaussiana é
separável, $O(Nb)$, e por isso fica mais cara quando $b$ cresce — medido: 4,6 ms
com $b = 35$ contra 22,8 ms com $b = 151$.)

---

## 3.4 Variantes que usam também o desvio local

A média sozinha ignora o **contraste** local. Duas variantes clássicas corrigem
isso usando o desvio-padrão local $s(x,y)$:

**Niblack (1986):**

$$
T(x,y) = m(x,y) + k\,s(x,y), \qquad k \approx -0{,}2
$$

**Sauvola (2000)** — padrão em binarização de documentos:

$$
\boxed{\;T(x,y) = m(x,y)\left[\,1 + k\left(\frac{s(x,y)}{R} - 1\right)\right]\;}
\qquad R = 128,\;\; k \approx 0{,}2
$$

**Por que Sauvola é melhor em fundo liso.** Em uma região homogênea $s \to 0$, e
então

$$
T \to m\,(1-k)
$$

isto é, o limiar cai **proporcionalmente** abaixo da média local, criando
automaticamente a margem que, no método do OpenCV, teria de ser fornecida à mão
pela constante $C$. Já em regiões de alto contraste ($s \to R$), $T \to m$, e o
método recupera a sensibilidade.

Os dois momentos necessários saem da mesma técnica de imagem integral, usando
$\operatorname{Var}(X) = E[X^2] - (E[X])^2$ com **duas** tabelas ($I$ e $I^2$).

Isso se confirma nos números: no vídeo, a instabilidade temporal média da
adaptativa foi de **6,00 %**, enquanto a de Sauvola ficou em **1,58 %** — quase
quatro vezes menos cintilação, pela mesma razão. Na fase de ruído a diferença é
ainda maior: 21,95 % contra 4,34 %.

---

## 3.5 A escolha do tamanho do bloco — a regra prática mais importante

Este foi o achado experimental mais instrutivo do trabalho.

Com o valor "de livro-texto" $b = 35$, a adaptativa teve desempenho **medíocre**
(IoU médio 0,513 no vídeo) e chegou a ser pior que o Otsu global. A figura
`D_adaptativa_04_sintetica.png` mostra a causa de imediato: os círculos e
retângulos grandes aparecem **ocos**, apenas com o contorno.

**Por quê.** Um pixel no centro de um objeto grande e uniforme tem uma vizinhança
$b \times b$ inteiramente contida no objeto. Logo $m(x,y) \approx I(x,y)$, e o
teste $I > m - C$ falha: o interior do objeto é classificado como fundo. O método
só detecta o que está **próximo de uma borda**.

$$
\boxed{\;b \;>\; \text{maior dimensão do objeto}\;}
$$

Mas o bloco também não pode ser grande demais: se $b$ se aproxima da escala em
que a iluminação varia, a média local volta a ser uma média global e o método
perde exatamente a propriedade que o justifica.

$$
\boxed{\;\underbrace{\text{maior objeto}}_{\approx\,130\text{ px}}
\;<\; b \;<\;
\underbrace{\text{escala de variação da iluminação}}_{\approx\,300\text{ px}}\;}
$$

Na cena usada, o maior objeto é o retângulo deslizante de 130 × 48 px (os discos
têm até 66 px de diâmetro) e o holofote tem cerca de 300 px de diâmetro efetivo.
A janela precisa ser larga o bastante para nunca caber **inteira** dentro de um
objeto, e estreita o bastante para que a iluminação seja aproximadamente
constante dentro dela. Isso prevê $130 < b < 300$ — e é exatamente onde a curva
experimental satura.

**Verificação experimental** (`X3_efeito_bloco_e_C.png`), IoU em quadros das
quatro fases:

| bloco | 35 | 71 | 101 | 131 | **151** | 201 | 301 |
|---|---|---|---|---|---|---|---|
| fase 1 | 0,627 | 0,876 | 0,964 | 0,968 | **0,971** | 0,975 | 0,980 |
| fase 2 | 0,532 | 0,761 | 0,914 | 0,957 | **0,959** | 0,963 | 0,969 |
| fase 3 | 0,570 | 0,784 | 0,926 | 0,941 | **0,946** | 0,955 | 0,970 |
| fase 4 | 0,240 | 0,309 | 0,354 | 0,386 | **0,401** | 0,425 | 0,451 |

O ganho satura por volta de $b \approx 101$–$151$. Adotou-se $b = 151$ como
"bloco ajustado", e o resultado no vídeo saltou de **IoU 0,513 para 0,863**.

> **Lição para a apresentação:** um método pode parecer inferior simplesmente
> porque seus hiperparâmetros não foram ajustados à escala do problema. Foi
> exatamente o que quase aconteceu neste trabalho — e é por isso que as duas
> configurações ($b=35$ e $b=151$) foram mantidas lado a lado nos resultados.

---

## 3.6 Validação da implementação própria

A implementação em NumPy deste trabalho foi comparada com `cv2.adaptiveThreshold`:

| Imagem | Concordância |
|---|---|
| `01_camera` | 99,48 % |
| `02_moedas` | 99,27 % |
| `03_pagina` | 99,88 % |
| `04_sintetica` | 99,90 % |

A diferença **não é um erro**. Investigando os 293 pixels discordantes na imagem
sintética: **em 100 % deles vale $|I(x,y) - T(x,y)| < 0{,}5$**, com desvio máximo de
0,499 — contra uma diferença média de 16,7 níveis nos pixels concordantes. Ou
seja, a discordância ocorre exclusivamente onde o limiar cai a menos de meio
nível de cinza da intensidade do pixel. A causa é que o OpenCV calcula a média
local em **aritmética inteira** (arredondando para `uint8`), enquanto a
implementação própria trabalha em `float64`.

---

## 3.7 Limitações

1. **Dois hiperparâmetros** ($b$ e $C$), ambos dependentes da escala do conteúdo
   — e não há como estimá-los a partir do histograma.
2. **Esvazia objetos maiores que o bloco** (seção 3.5).
3. **Amplifica ruído em regiões homogêneas.** É a pior fragilidade em vídeo:
   na fase 4, a instabilidade temporal chegou a **21,95 %** — mais de um quinto
   dos pixels piscando entre quadros consecutivos. Mitigações: aumentar $C$,
   pré-filtrar com gaussiana, ou usar Sauvola.
4. **Sem significado global.** Cada região é binarizada em relação a si mesma;
   a informação de que uma área é absolutamente mais escura que outra é perdida.
5. **Custo maior.** 4,6 ms ($b=35$) a 22,8 ms ($b=151$) por quadro, contra
   0,88 ms do Otsu — ainda assim compatível com tempo real (44 fps).

---

## 3.8 O que apresentar (resumo do Aluno C)

1. Retomar $I = R \cdot L$ e enunciar o problema: nenhum $T$ único serve.
2. $T \to T(x,y)$: a imagem de limiares. Mostrar o mapa $T(x,y)$ em
   `D_adaptativa_04_sintetica.png` — ele é visivelmente uma versão borrada da
   imagem, isto é, uma estimativa de $L$.
3. As duas fórmulas (média e gaussiana) e **o papel de $C$** — mostrar a linha
   $C = 0$ em `E_sensibilidade_*.png`.
4. A imagem integral: $O(1)$ por pixel, independente de $b$.
5. Sauvola: como o desvio local substitui o $C$ manual.
6. **O ponto alto:** a tabela de bloco da seção 3.5 — o mesmo método,
   IoU 0,532 com $b=35$ e 0,959 com $b=151$ no quadro do holofote, contra
   **0,139 do Otsu** no mesmo quadro. Mostrar o mosaico
   (`resultados/video/mosaico_comparativo.mp4`, t ≈ 7,5 s).
7. Fechar com a contrapartida honesta: na fase 4 (ruído), a adaptativa é a
   **pior** de todas (IoU 0,341, contra 0,834 do Otsu). Não existe método
   universalmente melhor.
