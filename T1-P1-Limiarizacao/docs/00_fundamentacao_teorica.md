# 0. Fundamentação teórica comum

> Base compartilhada pelos três seminários. Cada aluno deve dominar esta seção,
> porque ela estabelece a notação e o modelo físico usados nas três técnicas.

---

## 0.1 O que é limiarização

**Segmentação** é o processo de particionar uma imagem em regiões
significativas. A **limiarização** (*thresholding*) é a forma mais simples de
segmentação: classifica-se cada pixel comparando sua intensidade com um valor
de corte $T$, chamado **limiar**.

Para uma imagem em tons de cinza $I(x,y) \in \{0, 1, \dots, L-1\}$ (tipicamente
$L = 256$), a limiarização binária produz:

$$
g(x,y) =
\begin{cases}
1 & \text{se } I(x,y) > T \quad \text{(objeto)}\\[2pt]
0 & \text{se } I(x,y) \le T \quad \text{(fundo)}
\end{cases}
$$

O problema inteiro se reduz a **como escolher $T$**. É nessa escolha que as três
técnicas do trabalho diferem.

> **Convenção deste trabalho.** Nas cenas usadas, os objetos são **escuros sobre
> fundo claro** (texto em papel, peças sobre mesa). Por isso o código usa
> `inverso=True`, que equivale a $g = 1 \iff I \le T$. Isso não muda nada da
> matemática, apenas a polaridade da saída.

### Por que limiarizar?

| Vantagem | Consequência |
|---|---|
| Custo computacional baixíssimo | Viável em tempo real, em vídeo e em hardware embarcado |
| Saída binária | Habilita morfologia matemática, rotulagem de componentes conexas, OCR |
| Sem treinamento | Não exige dados rotulados nem GPU |

E as limitações, que este trabalho explora experimentalmente: a limiarização usa
**apenas a intensidade** do pixel, ignorando textura, forma e contexto. Dois
pixels com o mesmo cinza recebem o mesmo rótulo, ainda que um pertença a um
objeto e o outro a uma sombra do fundo.

---

## 0.2 O modelo de formação da imagem — a origem de todo o problema

Uma imagem é o produto de duas grandezas físicas distintas:

$$
I(x,y) \;=\; R(x,y) \cdot L(x,y) \;+\; n(x,y)
$$

| Símbolo | Nome | Significado |
|---|---|---|
| $R(x,y)$ | **refletância** | propriedade do *objeto*: quanto da luz incidente ele reflete. É **isto** que queremos segmentar. |
| $L(x,y)$ | **iluminação** | propriedade da *cena*: quanta luz chega àquele ponto. É um **estorvo**. |
| $n(x,y)$ | ruído | sensor, quantização, compressão |

Esta decomposição explica de forma limpa todos os resultados do trabalho:

- **Um limiar global só é válido se $L(x,y) \approx \text{constante}$.** Quando a
  iluminação é uniforme, comparar $I$ com $T$ equivale a comparar $R$ com
  $T/L$ — um limiar fixo na refletância, que é o que se deseja.
- Se $L$ varia **no tempo mas não no espaço** (uma lâmpada que escurece), o
  limiar global correto simplesmente *se desloca*. Um método automático como
  Otsu recalcula $T$ a cada quadro e **continua funcionando**.
- Se $L$ varia **no espaço** (sombra, holofote, vinheta), nenhum valor único de
  $T$ serve para a imagem toda: o fundo de uma região escura pode ser mais
  escuro que o objeto de uma região clara. **Nenhum método global resolve isso.**

Essa é a fronteira conceitual entre as técnicas 1–2 (globais) e a técnica 3
(local). O vídeo do experimento foi construído para separar exatamente esses
dois casos: a **fase 3** varia $L$ só no tempo, a **fase 2** varia $L$ no espaço.

> **Observação.** Tomando o logaritmo, $\log I = \log R + \log L$: a iluminação
> vira uma componente **aditiva** de baixa frequência espacial. A limiarização
> adaptativa é, em essência, um filtro passa-altas que estima e remove essa
> componente localmente.

---

## 0.3 O histograma

O histograma é a ferramenta de decisão de todos os métodos globais. Sendo $n_i$
o número de pixels com nível $i$ e $N$ o total de pixels:

$$
p(i) \;=\; \frac{n_i}{N}, \qquad p(i) \ge 0, \qquad \sum_{i=0}^{L-1} p(i) = 1
$$

Ele é tratado como a **distribuição de probabilidade** da intensidade. Momentos
importantes:

$$
\mu_T = \sum_{i=0}^{L-1} i\,p(i)
\qquad\qquad
\sigma_T^2 = \sum_{i=0}^{L-1} (i - \mu_T)^2\,p(i)
$$

O histograma **descarta toda a informação espacial**: duas imagens muito
diferentes podem ter histogramas idênticos. Essa é a raiz da fragilidade dos
métodos globais.

### Bimodalidade

A hipótese clássica é que uma cena com objeto e fundo gera um histograma
**bimodal**: um modo para cada população, separados por um vale. O limiar ideal
fica no vale.

Na prática a bimodalidade se perde quando:

- a iluminação é desigual (os modos se alargam e se fundem);
- o objeto ocupa uma fração muito pequena da imagem (seu modo some);
- o ruído é forte ou o contraste é baixo (os modos se sobrepõem);
- há mais de duas populações (daí o Multi-Otsu).

---

## 0.4 Taxonomia e posicionamento das três técnicas

Sezgin & Sankur (2004) organizam mais de 40 métodos de limiarização em seis
famílias. As três técnicas do trabalho se encaixam assim:

| # | Técnica | Família | Nº de limiares | Escopo |
|---|---|---|---|---|
| 1 | **Otsu** | agrupamento (*clustering*) | 1 | **global** |
| 2 | **Multi-Otsu** | agrupamento | $M-1$ | **global** |
| 3 | **Adaptativa** | local / espacial | 1 **por pixel** | **local** |

A diferença essencial em uma frase:

> Otsu e Multi-Otsu produzem **um número** (ou uns poucos números) válido para a
> imagem inteira; a limiarização adaptativa produz uma **imagem de limiares**
> $T(x,y)$, do mesmo tamanho da entrada.

---

## 0.5 Como medimos a qualidade

Comparando a máscara predita com o gabarito (*ground truth*), contam-se:

- **VP** verdadeiro positivo (objeto corretamente detectado)
- **VN** verdadeiro negativo (fundo corretamente rejeitado)
- **FP** falso positivo (fundo classificado como objeto)
- **FN** falso negativo (objeto perdido)

$$
\text{Acurácia} = \frac{VP + VN}{VP+VN+FP+FN}
\qquad
\text{Precisão} = \frac{VP}{VP + FP}
\qquad
\text{Revocação} = \frac{VP}{VP + FN}
$$

$$
F_1 = \text{Dice} = \frac{2\,VP}{2\,VP + FP + FN}
\qquad\qquad
\text{IoU} = \text{Jaccard} = \frac{VP}{VP + FP + FN}
$$

**Por que IoU e não acurácia?** Nas cenas do trabalho o objeto ocupa cerca de
**12 %** da área. Um classificador que respondesse "tudo é fundo" teria
**87,6 % de acurácia** e IoU igual a **zero**. IoU e $F_1$ não se deixam enganar por
desbalanceamento; por isso são a métrica principal aqui.

Também é reportado o **MCC** (coeficiente de correlação de Matthews), que usa as
quatro células da matriz de confusão:

$$
\text{MCC} = \frac{VP\cdot VN - FP \cdot FN}
{\sqrt{(VP{+}FP)(VP{+}FN)(VN{+}FP)(VN{+}FN)}} \in [-1, 1]
$$

### Métrica específica de vídeo: instabilidade temporal

Um critério que **só existe em vídeo** e que se mostrou decisivo neste trabalho:

$$
\text{Inst}(k) = \frac{1}{N}\,\bigl|\{(x,y) : g_k(x,y) \ne g_{k-1}(x,y)\}\bigr|
$$

isto é, a fração de pixels que trocaram de classe entre quadros consecutivos.
Como a cena muda pouco de um quadro para o outro, um valor alto indica
**cintilação** (*flicker*) causada pelo método, não movimento real. Um método
pode ter IoU razoável e ainda assim ser inutilizável em vídeo por instabilidade.

---

## 0.6 Referências

1. **Otsu, N.** (1979). *A Threshold Selection Method from Gray-Level
   Histograms*. IEEE Trans. SMC, 9(1), 62–66.
2. **Liao, P.-S., Chen, T.-S., Chung, P.-C.** (2001). *A Fast Algorithm for
   Multilevel Thresholding*. J. Information Science and Engineering, 17, 713–727.
3. **Sauvola, J., Pietikäinen, M.** (2000). *Adaptive Document Image
   Binarization*. Pattern Recognition, 33(2), 225–236.
4. **Bradley, D., Roth, G.** (2007). *Adaptive Thresholding using the Integral
   Image*. Journal of Graphics Tools, 12(2), 13–21.
5. **Sezgin, M., Sankur, B.** (2004). *Survey over image thresholding techniques
   and quantitative performance evaluation*. J. Electronic Imaging, 13(1), 146–165.
6. **Gonzalez, R. C., Woods, R. E.** *Digital Image Processing*, cap. 10
   (Image Segmentation).
7. scikit-image — <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html>
8. scikit-image — <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html>
9. OpenCV — <https://docs.opencv.org/4.4.0/d7/d4d/tutorial_py_thresholding.html>
