# 2. ALUNO B — Limiarização multinível (Multi-Otsu)

> **Fonte base:** <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html>
> **Tempo sugerido de apresentação:** 10–12 min
> **Figuras de apoio:** `resultados/graficos/C_multiotsu_*.png`, `C2_multiotsu_M_*.png`, `X2_histogramas_das_fases.png`

---

## 2.1 Por que generalizar

O método de Otsu clássico parte de uma hipótese forte: **a cena tem exatamente
duas populações de intensidade**. O mundo raramente coopera:

| Cena | Populações reais |
|---|---|
| Documento digitalizado | papel, tinta, **sombra da dobra / mancha** |
| Radiografia | fundo, tecido mole, **osso** |
| Imagem de satélite | água, vegetação, **solo exposto**, construção |
| Peças em esteira | esteira, peça metálica, **sombra projetada** |

Forçar duas classes onde há três significa, necessariamente, **fundir duas
populações distintas** em um único rótulo. A informação perdida é irrecuperável
nas etapas seguintes.

**Multi-Otsu** generaliza o critério para $M$ classes, produzindo $M-1$ limiares.

---

## 2.2 Formulação matemática

### 2.2.1 Generalização do critério

Buscam-se $M-1$ limiares ordenados

$$
0 \le t_1 < t_2 < \dots < t_{M-1} < L-1
$$

que definem $M$ classes disjuntas que cobrem toda a faixa de níveis:

$$
C_0 = [0,\,t_1], \quad
C_1 = [t_1{+}1,\,t_2], \quad \dots, \quad
C_{M-1} = [t_{M-1}{+}1,\,L{-}1]
$$

Para cada classe $C_k$, exatamente como no caso binário:

$$
\omega_k = \sum_{i \in C_k} p(i)
\qquad\qquad
\mu_k = \frac{1}{\omega_k}\sum_{i \in C_k} i\,p(i)
$$

O critério é a extensão **direta** da variância interclasse:

$$
\boxed{\;
\sigma_B^2(t_1,\dots,t_{M-1}) = \sum_{k=0}^{M-1} \omega_k\,(\mu_k - \mu_T)^2
\;}
$$

$$
\boxed{\;
(t_1^{*},\dots,t_{M-1}^{*}) = \arg\max\; \sigma_B^2(t_1,\dots,t_{M-1})
\;}
$$

A identidade $\sigma_T^2 = \sigma_W^2 + \sigma_B^2$ continua valendo para $M$
classes (a demonstração da seção 1.2.3 se repete classe a classe), então
maximizar $\sigma_B^2$ continua equivalente a minimizar a dispersão interna.

### 2.2.2 O problema: explosão combinatória

A busca exaustiva testa todas as combinações de $M-1$ limiares entre $L-1$
níveis:

$$
\binom{L-1}{M-1} \quad \text{combinações}, \quad
\text{cada uma com } O(L) \text{ operações}
$$

Com $L = 256$:

| $M$ | Limiares | Combinações | Viável? |
|---|---|---|---|
| 2 | 1 | 255 | trivial |
| 3 | 2 | 32 385 | sim |
| 4 | 3 | 2 731 135 | lento (minutos em Python) |
| 5 | 4 | 172 061 505 | **inviável** |
| 6 | 5 | ~8,6 × 10⁹ | impossível |

> **Isto não é hipotético.** A primeira versão do código deste trabalho usava
> busca combinatória (`itertools.combinations`) e **travou** ao chegar em
> $M = 4$ e $M = 5$. Foi preciso substituí-la — ver seção 2.2.4.

### 2.2.3 Tabelas de momentos (Liao, Chen & Chung, 2001)

O truque central é **pré-computar** os momentos de todo intervalo possível
$[u, v]$:

$$
P(u,v) = \sum_{i=u}^{v} p(i)
\qquad\qquad
S(u,v) = \sum_{i=u}^{v} i\,p(i)
$$

Ambas as tabelas $L \times L$ saem de somas acumuladas em $O(L^2)$:

$$
P(u,v) = \text{cum}_p[v{+}1] - \text{cum}_p[u]
\qquad
S(u,v) = \text{cum}_s[v{+}1] - \text{cum}_s[u]
$$

Define-se então a **variância interclasse modificada**:

$$
\boxed{\;H(u,v) = \frac{S(u,v)^2}{P(u,v)}\;}
$$

**Proposição.** $\displaystyle \sigma_B^2 = \sum_{k=0}^{M-1} H(u_k, v_k) - \mu_T^2$,
onde $[u_k, v_k]$ são os intervalos das classes.

*Demonstração.* Expandindo o quadrado no critério:

$$
\sigma_B^2 = \sum_k \omega_k(\mu_k-\mu_T)^2
           = \sum_k \omega_k\mu_k^2 - 2\mu_T\underbrace{\sum_k \omega_k\mu_k}_{=\,\mu_T}
             + \mu_T^2\underbrace{\sum_k \omega_k}_{=\,1}
           = \sum_k \omega_k\mu_k^2 - \mu_T^2
$$

Como $\omega_k = P(u_k,v_k)$ e $\mu_k = S(u_k,v_k)/P(u_k,v_k)$:

$$
\omega_k\,\mu_k^2 = P \cdot \left(\frac{S}{P}\right)^2 = \frac{S^2}{P} = H(u_k,v_k)
\qquad \blacksquare
$$

Como $\mu_T^2$ é constante, maximizar $\sigma_B^2$ equivale a maximizar

$$
J = \sum_{k=0}^{M-1} H(t_k{+}1,\; t_{k+1})
$$

Cada avaliação de uma combinação cai de $O(L)$ para $O(M)$ — um ganho grande,
mas o número de combinações continua explodindo.

### 2.2.4 Programação dinâmica: eliminando a explosão

A função objetivo $J$ é **aditiva sobre intervalos disjuntos e contíguos**, o que
é exatamente a estrutura que a programação dinâmica explora. Defina

$$
D[m][v] = \text{melhor valor de } J \text{ usando } m \text{ intervalos para cobrir } [0,\,v]
$$

Recorrência:

$$
\boxed{
\begin{aligned}
D[1][v] &= H(0,\,v) \\[4pt]
D[m][v] &= \max_{u \,<\, v} \Bigl( D[m{-}1][u] \;+\; H(u{+}1,\; v) \Bigr)
\end{aligned}}
$$

A resposta é $D[M][L{-}1]$, e os limiares saem por **retro-rastreamento** dos
argumentos que realizaram cada máximo.

| | Busca exaustiva | Programação dinâmica |
|---|---|---|
| Complexidade | $O\!\left(\binom{L}{M-1} \cdot M\right)$ | $\;O(M \cdot L^2)\;$ |
| $M = 5$, $L = 256$ | ~1,7 × 10⁸ avaliações | ~3,3 × 10⁵ operações |
| Tempo medido | horas | **1,5 ms** |

**Validação.** A implementação por programação dinâmica deste trabalho foi
comparada com `skimage.filters.threshold_multiotsu` em 3 imagens × 4 valores de
$M$ (12 casos) e produziu **limiares idênticos em todos eles**:

| Imagem | $M{=}2$ | $M{=}3$ | $M{=}4$ | $M{=}5$ |
|---|---|---|---|---|
| camera | [102] | [87, 176] | [69, 134, 180] | [46, 100, 145, 182] |
| moedas | [107] | [77, 139] | [63, 107, 156] | [58, 95, 134, 173] |
| página | [157] | [114, 186] | [93, 150, 199] | [71, 119, 161, 203] |

---

## 2.3 Aplicação dos limiares

Obtidos os limiares, a rotulação é uma busca binária por pixel — `np.digitize`:

```python
from skimage.filters import threshold_multiotsu
import numpy as np

limiares = threshold_multiotsu(imagem, classes=3)   # ex.: [87, 176]
regioes  = np.digitize(imagem, bins=limiares)       # rótulos 0, 1, 2
```

O resultado **não é uma imagem binária**, e sim um **mapa de rótulos**. Para
obter uma máscara binária é preciso decidir quais classes compõem o objeto — e
essa decisão é do projetista, não do algoritmo. Este trabalho avalia as duas
opções naturais:

- `multiotsu_t1`: objeto $=$ apenas a classe mais escura ($I \le t_1$);
- `multiotsu_t2`: objeto $=$ classes escura **e** média ($I \le t_2$).

Como se verá, elas se comportam de forma **radicalmente diferente** sob
iluminação desigual.

---

## 2.4 Como escolher $M$

Não há resposta automática. Critérios úteis:

1. **Conhecimento do domínio** — quantas populações físicas a cena tem? É o
   critério mais confiável.
2. **Inspeção do histograma** — contar os modos visíveis
   (`C2_multiotsu_M_*.png` mostra o efeito de $M = 2, 3, 4, 5$).
3. **Ganho marginal de $\sigma_B^2$** — aumentar $M$ **sempre** aumenta
   $\sigma_B^2$ (mais classes sempre explicam mais variância, no limite
   $M = L$ com $\sigma_B^2 = \sigma_T^2$). Portanto **não se pode escolher $M$
   maximizando $\sigma_B^2$** — seria como escolher o número de parâmetros de um
   modelo maximizando o ajuste. Procura-se o "cotovelo" da curva.
4. **Critérios de seleção de modelo** (AIC/BIC, ou índices como Calinski–Harabasz),
   que penalizam explicitamente a complexidade.

> Este é o ponto teórico mais delicado do Multi-Otsu e vale ser destacado na
> apresentação: **$M$ é um hiperparâmetro, não um resultado do método.**

---

## 2.5 Resultados obtidos

### Na imagem estática com gabarito (`04_sintetica`)

| Máscara | IoU | Precisão | Revocação |
|---|---|---|---|
| Otsu binário ($T = 125$) | 0,401 | 0,410 | 0,948 |
| **Multi-Otsu, só classe escura ($t_1 = 74$)** | **0,593** | **1,000** | 0,593 |
| Multi-Otsu, escura + média ($t_2 = 148$) | 0,240 | 0,242 | 0,971 |

Observação importante: usando apenas a classe mais escura, a **precisão é 1,000**
— *nenhum* pixel de fundo foi classificado como objeto. O Multi-Otsu isolou uma
classe que o gradiente de iluminação não conseguiu contaminar. O preço é a
revocação de 0,593: os objetos de cinza médio foram perdidos.

### No vídeo (600 quadros, IoU médio por fase)

| Fase | Otsu | Multi-Otsu $t_1$ | Multi-Otsu $t_2$ |
|---|---|---|---|
| 1 — iluminação uniforme | 0,944 | 0,592 | **0,966** |
| 2 — holofote móvel | 0,139 | **0,731** | 0,138 |
| 3 — escurecimento global | 0,943 | 0,600 | **0,965** |
| 4 — ruído + baixo contraste | **0,834** | 0,639 | 0,626 |

**Este é o resultado mais interessante do Multi-Otsu**, e merece destaque na
apresentação:

- Com iluminação bem comportada (fases 1 e 3), $t_2$ é o **melhor método global
  do trabalho** (IoU 0,966), superando o Otsu binário — porque a terceira classe
  permite acomodar os objetos de cinza médio sem arrastar o fundo junto.
- Sob o holofote (fase 2), $t_2$ colapsa junto com o Otsu (0,138), mas $t_1$
  **resiste** (0,731). A razão é geométrica: o holofote rebaixa o fundo até cerca
  de $0{,}34 \times 225 \approx 76$, valor que ainda está **acima** de $t_1 \approx 52$,
  mas muito **abaixo** de $t_2 \approx 131$. A classe mais escura tem uma
  "margem de segurança" que as demais não têm.
- Na fase 4, $t_1$ é o **método mais estável de todos** (instabilidade temporal
  de apenas 1,81 %, contra 19,4 % de $t_2$ e 22,0 % da adaptativa) — uma
  diferença de **12×** para a adaptativa.

**Conclusão do Aluno B:** o Multi-Otsu não é apenas "Otsu com mais cortes". Ao
fornecer vários limiares, ele oferece ao projetista a **escolha de qual fatia do
histograma usar** — e essa liberdade pode ser explorada para ganhar robustez a
variações de iluminação que derrubam o Otsu binário.

---

## 2.6 Limitações

1. **Continua sendo global.** Herda integralmente a fragilidade à variação
   espacial de iluminação (fase 2). A resiliência de $t_1$ é um efeito
   colateral feliz, não uma propriedade garantida.
2. **$M$ precisa ser escolhido** e não há critério interno para isso.
3. **Custo cresce com $M$.** Medido: 1,05 ms para $M = 3$ contra 32,8 ms para
   $M = 4$ no scikit-image — um salto de 31×, relevante em vídeo.
4. **Classes vazias.** Se a cena não tiver de fato $M$ populações, o método
   fatia artificialmente um modo único, produzindo fronteiras sem significado.
5. **A saída não é binária**, exigindo uma decisão adicional de agrupamento das
   classes.

---

## 2.7 O que apresentar (resumo do Aluno B)

1. Motivação: mostrar `X2_histogramas_das_fases.png` — histogramas com três modos.
2. O critério generalizado $\sigma_B^2 = \sum_k \omega_k(\mu_k-\mu_T)^2$.
3. **A explosão combinatória** — a tabela de $\binom{255}{M-1}$, e contar que o
   código realmente travou em $M = 5$.
4. A solução: LUT $H(u,v) = S^2/P$ + programação dinâmica, $O(M L^2)$, ~1,3 ms.
5. `C2_multiotsu_M_*.png`: efeito de $M$, e por que **não se escolhe $M$
   maximizando $\sigma_B^2$**.
6. **O ponto alto:** na fase 2 do vídeo, $t_2$ fica em IoU 0,138 e $t_1$ em
   0,731 — *o mesmo método, o mesmo quadro*, resultados opostos apenas por qual
   limiar se usa. Explicar a margem de segurança da classe escura.
