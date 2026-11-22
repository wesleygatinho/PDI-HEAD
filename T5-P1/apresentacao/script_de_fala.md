# Script de fala — Simulador de Espelho Corporal

Texto para falar em cada slide de `apresentacao.pdf`. Leia duas ou três vezes
antes; não é para ler no papel durante a apresentação.

Onde aparece `[A]`, `[B]` e `[C]`, fale o nome da pessoa.

---

## Divisão

| Quem | Slides | Tempo |
|---|---|---|
| **Aluno A** | capa, 1, 2 | 2 min 30 s |
| **Aluno B** | 3, 4 | 2 min 00 s |
| **Aluno C** | 5, 6, 7 | 3 min 00 s |
| **Todos** | demonstração | 3 min |

Total: cerca de 7 min 30 s de slides e 3 min de demonstração.

---

## Antes de começar

- Notebook ligado na tomada e com a webcam livre: nenhuma outra janela do
  espelho aberta.
- Rode tudo no **mesmo notebook** em que os modelos já foram baixados
  (pasta `modelos/`); assim não depende de internet na sala. Os números dos
  slides foram medidos em outra máquina: num notebook mais lento o FPS ao vivo
  fica abaixo de 30, e tudo bem, é só falar que depende do hardware.
- Faça um teste rápido da câmera na sala (com a luz que vai ser usada; pouca
  luz derruba o FPS da webcam):

```bash
python testar_camera.py
```

- Marque no chão o ponto onde quem vai demonstrar fica de corpo inteiro no
  quadro, cabeça e pés visíveis (uns 3 m da câmera).
- Deixe um terminal aberto na pasta `T5-P1`, com o ambiente ativado, pronto
  para rodar `python espelho.py`.

---

# ABERTURA — Aluno A

## Capa · 15 s

> Bom dia. Nosso tema é o simulador de espelho corporal: o programa olha pela
> webcam, encontra o corpo da pessoa e desenha o esqueleto dela na tela, em
> tempo real. Eu explico como funciona, o [B] mostra o desempenho e o [C] a taxa
> de acerto. No fim a gente faz a demonstração ao vivo.

---

## Slide 1 · Objetivo e arquitetura · 1 min 10 s

> O caminho de cada quadro é esse aqui, da esquerda para a direita.
>
> A webcam entrega trinta quadros por segundo. O OpenCV captura cada quadro e
> passa para o MediaPipe Pose, que é uma rede neural já treinada pelo Google. Ela
> devolve 33 pontos do corpo, os landmarks: nariz, olhos, ombros, cotovelos,
> punhos, quadris, joelhos, tornozelos e pés. Cada ponto vem com a posição x e y
> na imagem, uma profundidade z e uma visibilidade, que é a confiança de que
> aquele ponto está mesmo aparecendo.
>
> Depois a gente espelha, suaviza e desenha o esqueleto.
>
> O MediaPipe tem três tamanhos de modelo, o Lite, o Full e o Heavy, e o programa
> troca entre eles com uma tecla. Ele também mede sozinho as quatro métricas que
> a avaliação pede.

---

## Slide 2 · O espelho · 1 min 05 s

> Aqui estão três dos quatro modos de visualização. O primeiro desenha o
> esqueleto em cima da câmera. O segundo mostra só o esqueleto, como um avatar.
> E o rastro deixa um traço que vai apagando, para ver o movimento. O quarto
> modo, que não está aqui, é a câmera e o avatar lado a lado.
>
> Dois detalhes de implementação. Primeiro, o espelhamento. Se a gente
> inverter a imagem antes de mandar para a rede, ela troca o braço esquerdo
> pelo direito. Então a detecção roda no quadro original, e só depois a gente
> inverte a imagem e cada coordenada x vira um menos x.
>
> Segundo, pontos com visibilidade abaixo de meio não são desenhados. São
> pontos fora do quadro ou escondidos atrás do corpo, e a posição deles é um
> chute da rede.

*Passe para o [B].*

---

# RESULTADOS — Aluno B

## Slide 3 · Desempenho: FPS e latência · 1 min 10 s

> Esse gráfico mostra para onde vai o tempo de cada quadro nos três modelos.
>
> Em azul é o tempo que o programa fica parado esperando a câmera entregar o
> próximo quadro. Em laranja, o tempo da rede neural. Em verde, desenhar e
> mostrar na tela.
>
> No Lite e no Full sobra azul: o programa termina antes de o próximo quadro
> chegar. Os dois rodam a trinta quadros por segundo, que é o máximo da câmera.
> O gargalo é a webcam, não o modelo.
>
> No Heavy não tem azul. A rede leva 41 milissegundos, e a câmera manda um
> quadro a cada 33. Ele não dá conta e cai para 22 quadros por segundo.
>
> A latência, que é do quadro chegar até aparecer na tela, fica em 14, 18 e 45
> milissegundos.

---

## Slide 4 · Estabilidade · 50 s

> Para a estabilidade, a gente pediu para a pessoa ficar parada e mediu quanto
> os pontos tremem de um quadro para o outro, em pixels. As barras azuis são o
> que sai direto da rede. As laranjas, depois de um filtro de suavização chamado
> One Euro.
>
> A ideia desse filtro é suavizar forte quando o ponto está parado e quase nada
> quando ele se mexe rápido. Assim tira o tremor sem deixar o esqueleto
> atrasado. Ele reduziu o tremor em 19, 35 e 12 por cento.
>
> E a pose foi detectada em cem por cento dos quadros, sem nenhuma perda.

*Passe para o [C].*

---

# TAXA DE ACERTO — Aluno C

## Slide 5 · Taxa de acerto · 1 min 20 s

> Para medir acerto precisaria de um gabarito, alguém marcando à mão os 33
> pontos em cada quadro. A gente não tinha isso. Então usou o Heavy, que é o
> modelo maior e mais preciso, como referência.
>
> Gravamos um vídeo com oito poses, e o mesmo vídeo passou pelos três modelos.
> Um ponto conta como certo se cair perto do ponto do Heavy, a menos de vinte
> por cento do tamanho do tronco. Essa é a métrica PCK, usada em artigos de
> estimativa de pose.
>
> Com a pessoa perto da câmera, só a parte de cima do corpo, os dois modelos
> concordam quase cem por cento. Com o corpo inteiro, o Full fica em 92 e o Lite
> cai para 84. E a diferença aparece nas poses difíceis: de perfil o Lite acerta
> 56, em uma perna só 35. O Full fica entre 97 e 100.
>
> Um cuidado: isso mede concordância com o melhor modelo, não acerto absoluto.
> Se os três errarem juntos, a gente não vê.

---

## Slide 6 · Onde o Lite falha · 30 s

> Um exemplo. A pessoa está de lado, em uma perna. Em branco, o Heavy. À
> esquerda, o Lite: ele junta a perna levantada com a de apoio, e o joelho
> e o pé dessa perna ficam no lugar errado. À direita, o Full, em cima do
> Heavy.

---

## Slide 7 · Conclusões · 1 min 10 s

> Juntando tudo. O Full é o melhor para esse uso: roda a trinta quadros por
> segundo, com 18 milissegundos de latência, o menor tremor e 92 por cento de
> acerto com o corpo inteiro. O Heavy é o mais preciso, mas não aguenta o tempo
> real nessa máquina. O Lite é o mais rápido, mas se perde de lado e em uma perna.
>
> E o que mais pesou foi o enquadramento. Na primeira gravação, com a pessoa
> perto, as pernas nem apareceram. Na segunda a cabeça saiu pelo topo. E quando
> a cabeça sai do quadro, até o Heavy erra.
>
> As limitações: foi uma gravação por cenário, a referência é um modelo e não um
> gabarito, e numa das poses a própria referência falhou.
>
> Agora a demonstração.

---

# DEMONSTRAÇÃO — todos · 3 min

Uma pessoa fica no ponto marcado, de corpo inteiro. Outra opera o teclado e
fala. A terceira fica de reserva para responder perguntas.

```bash
python espelho.py
```

Roteiro, nesta ordem:

1. **Entrar no quadro** (modo 1, Full). Mexer os braços e as pernas devagar.
   > Esse é o esqueleto em tempo real. No canto, o painel mostra o FPS, a
   > latência e quantos pontos estão visíveis.
2. **Levantar o braço direito.**
   > Repara que o lado desenhado bate com o espelho: a pessoa levanta o direito
   > e a imagem levanta do mesmo lado que um espelho mostraria.
3. **Tecla `2`, depois `4`.** Avatar e rastro. No rastro, fazer um movimento
   amplo com os braços.
4. **Tecla `1`, depois `L`.** Lite. Virar de lado e ficar em uma perna.
   > Aqui é onde o Lite costuma se perder.
5. **Tecla `F`.** Voltar para o Full na mesma pose.
6. **Tecla `H`.** Heavy.
   > Olhem o FPS caindo: o Heavy demora mais que o intervalo entre dois
   > quadros da câmera.
7. **Tecla `F` e depois `S`** para desligar a suavização, com a pessoa parada.
   > Sem o filtro, dá para ver os pontos tremendo. Com o `S` de novo, param.
8. **`Q`** para sair.

Se algo der errado: se a câmera não abrir, feche qualquer outra janela que
esteja usando a câmera e rode de novo. Se o esqueleto sumir, a pessoa saiu do
quadro: dê um passo para trás.

---

## Perguntas prováveis

**Por que não treinaram um modelo?**
> O tema pedia usar o MediaPipe, que já vem treinado. O trabalho foi montar o
> pipeline em tempo real, corrigir o espelhamento, suavizar e medir.

**Por que o Heavy não é o escolhido, se é o mais preciso?**
> Porque não fecha trinta quadros por segundo nessa máquina. No espelho, um
> esqueleto atrasado incomoda mais que alguns pixels de erro.

**Como sabem que o Heavy está certo?**
> Não sabemos com certeza; por isso dizemos que é concordância, não acerto. Num
> caso, braços para cima com a cabeça cortada, vimos o próprio Heavy errar.

**O que é a visibilidade?**
> É a confiança que a rede dá de que o ponto está aparecendo na imagem. Abaixo
> de 0,5 a gente não desenha.

**O que o One Euro faz exatamente?**
> É um passa-baixa cuja frequência de corte cresce com a velocidade do ponto.
> Parado, corta muito e tira o tremor. Em movimento rápido, corta pouco e não
> atrasa.
