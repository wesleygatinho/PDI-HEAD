# Falas — Simulador de Espelho Corporal (slides novos)

Roteiro curto para os 9 slides novos (as mesmas falas estão nas notas do
apresentador de cada slide). Leiam duas ou três vezes antes; na hora, falem
com as próprias palavras.

Onde aparece `[A]`, `[B]` e `[C]`, fale o nome da pessoa.

| Quem | Slides | Tempo |
|---|---|---|
| **Aluno A** | capa, 01 Como funciona, 02 O espelho | ~2 min |
| **Aluno B** | 03 Desempenho, 04 Estabilidade | ~1 min 45 s |
| **Aluno C** | 05 Taxa de acerto, 06 Onde o Lite falha, 07 Conclusão | ~2 min 10 s |
| **Todos** | 08 Demonstração | ~2 min |

Total: cerca de 8 minutos.

---

## Aluno A

### Capa · 15 s

> Bom dia. Nosso tema é o simulador de espelho corporal: o programa olha pela
> webcam, encontra o corpo da pessoa e redesenha o esqueleto dela na tela, em
> tempo real, como um espelho. Eu mostro como ele funciona, o [B] mostra o
> desempenho e o [C] mostra se ele acerta. No fim a gente faz a demonstração ao vivo.

### 01 · Como funciona · 1 min

> Cada quadro faz esse caminho, da esquerda para a direita. A webcam entrega
> trinta quadros por segundo, o OpenCV captura e passa para o MediaPipe Pose,
> uma rede neural que já vem treinada pelo Google. Ela devolve 33 pontos do
> corpo — nariz, ombros, cotovelos, punhos, quadris, joelhos, pés — cada um com
> posição, profundidade e uma visibilidade, que é a confiança de que o ponto
> está mesmo aparecendo. Ela vem em três tamanhos, Lite, Full e Heavy, e o
> programa troca entre eles com uma tecla.
>
> Dois cuidados. Primeiro: se a gente invertesse a imagem antes de mandar para
> a rede, ela trocaria o braço esquerdo pelo direito. Então detecta no quadro
> original e só depois espelha. Segundo: ponto com visibilidade baixa é um
> chute da rede, então a gente não desenha.

### 02 · O espelho · 40 s

> Aqui estão três dos quatro modos. O primeiro desenha o esqueleto por cima da
> câmera. O segundo mostra só o esqueleto, como um avatar. O terceiro é o
> rastro: ele deixa um traço que vai apagando, então dá para ver o movimento.
> O quarto, câmera e avatar lado a lado, a gente mostra ao vivo.
>
> As cores seguem o corpo da pessoa: laranja é o lado esquerdo, azul o
> direito. Agora o [B] mostra se isso tudo roda rápido o suficiente.

---

## Aluno B

### 03 · Desempenho · 1 min

> Esse gráfico mostra para onde vai o tempo de cada quadro. Em cinza, o
> programa parado esperando a câmera mandar o próximo quadro. Em laranja, a
> rede neural. Em claro, desenhar e mostrar. A linha azul tracejada é o
> intervalo entre dois quadros da câmera: 33 milissegundos.
>
> No Lite e no Full sobra cinza: o programa termina antes de o próximo quadro
> chegar. Os dois rodam a trinta quadros por segundo, que é o máximo da
> câmera. Quem limita é a webcam, não o modelo.
>
> No Heavy, só a rede já leva 41 milissegundos e passa da linha. Ele não dá
> conta e cai para 22 quadros por segundo. A latência, do quadro chegar até
> aparecer na tela, fica em 14, 18 e 45 milissegundos.

### 04 · Estabilidade · 45 s

> Para medir estabilidade, a pessoa fica parada e a gente mede quanto os
> pontos tremem de um quadro para o outro, em pixels. A barra cinza é o que
> sai direto da rede; a azul, depois do filtro One Euro.
>
> A ideia do filtro é suavizar forte quando o ponto está parado e quase nada
> quando ele se mexe rápido. Assim tira o tremor sem deixar o esqueleto
> atrasado. Ele reduziu o tremor em 19, 35 e 12 por cento, e o Full com filtro
> ficou com o menor tremor de todos. E a pose foi detectada em cem por cento
> dos quadros.
>
> Agora o [C] mostra se esses pontos estão no lugar certo.

---

## Aluno C

### 05 · Taxa de acerto · 1 min

> Para medir acerto de verdade, alguém teria que marcar à mão os 33 pontos em
> cada quadro. A gente não tinha isso, então usou o Heavy, o modelo maior e
> mais preciso, como régua.
>
> Gravamos um vídeo com oito poses e passamos o mesmo vídeo pelos três
> modelos. Um ponto conta como certo se cair a menos de vinte por cento do
> tamanho do tronco do ponto do Heavy. Essa métrica se chama PCK e é a usada
> em artigos de estimativa de pose.
>
> Com a pessoa perto, só a parte de cima, os dois concordam quase cem por
> cento. Com o corpo inteiro, o Full fica em 92 e o Lite cai para 84. E a
> diferença aparece nas poses difíceis: de perfil o Lite acerta 56, em uma
> perna só 35, enquanto o Full fica entre 97 e 100.
>
> Um cuidado: isso é concordância com o melhor modelo, não acerto absoluto.

### 06 · Onde o Lite falha · 30 s

> Um exemplo do nosso vídeo. A pessoa está de lado, em uma perna. O esqueleto
> branco é o Heavy. À esquerda, o Lite: ele junta a perna levantada com a de
> apoio, e o joelho e o pé ficam no lugar errado. À direita, o Full,
> praticamente em cima do Heavy.

### 07 · Conclusão · 40 s

> Juntando tudo: o Full é o melhor para esse uso. Trinta quadros por segundo,
> 18 milissegundos de atraso, o menor tremor com o filtro e 92 por cento de
> acerto com o corpo inteiro. O Heavy é o mais preciso, mas não aguenta o
> tempo real. O Lite é o mais rápido, mas se perde de lado e em uma perna.
>
> E uma lição prática: o enquadramento pesou tanto quanto o modelo. Quando a
> cabeça ou os pés saem do quadro, até o Heavy erra.
>
> Agora a demonstração.

---

## 08 · Demonstração — todos · 2 min

Uma pessoa fica no ponto marcado, de corpo inteiro. Outra opera o teclado e
fala. A terceira fica de reserva para perguntas.

```bash
python espelho.py
```

1. **Entrar no quadro** (modo 1, Full) e mexer braços e pernas.
   > No canto, o painel mostra o FPS, a latência e quantos pontos estão visíveis.
2. **Levantar o braço direito.**
   > A imagem levanta do mesmo lado que um espelho mostraria.
3. **Teclas `2` e `4`**: avatar e rastro, com um movimento amplo dos braços.
4. **Tecla `1`, depois `L`**: virar de lado e ficar em uma perna.
   > Aqui o Lite costuma se perder.

   Depois **`F`**:
   > O Full acompanha.
5. **Tecla `H`**:
   > O FPS cai: o Heavy demora mais que o intervalo entre dois quadros.
6. **`F` e depois `S`**, com a pessoa parada:
   > Sem o filtro, os pontos tremem.

   **`S`** de novo para religar.
7. **`Q`** para sair.
   > Obrigado. Perguntas?

Se o esqueleto sumir, a pessoa saiu do quadro: dê um passo para trás.

O notebook da apresentação pode ser mais lento que o das medições; se o FPS
ao vivo ficar abaixo de 30, diga que depende do hardware e da luz da sala.

---

## Perguntas prováveis

**Por que não treinaram um modelo?**
> O tema pedia o MediaPipe, que já vem treinado. O trabalho foi montar o
> pipeline em tempo real, corrigir o espelhamento, suavizar e medir.

**Por que não o Heavy, se é o mais preciso?**
> Porque não fecha trinta quadros por segundo. No espelho, um esqueleto
> atrasado incomoda mais que alguns pixels de erro.

**Como sabem que o Heavy está certo?**
> Não sabemos com certeza; por isso dizemos concordância, não acerto. Com a
> cabeça cortada, vimos o próprio Heavy errar.

**O que o One Euro faz?**
> É um passa-baixa cuja frequência de corte cresce com a velocidade do ponto.
> Parado, corta muito e tira o tremor; rápido, corta pouco e não atrasa.
