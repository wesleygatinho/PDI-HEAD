# Script de fala — apresentação v2

Texto para falar em cada slide de `apresentacao_v2.pdf`.
Não é para decorar nem para ler no papel. É para você ler duas ou três vezes
antes e saber o que precisa sair da sua boca em cada tela.

São **2 556 palavras**. Em ritmo normal de apresentação isso dá **17 a 18
minutos**; se vocês falarem devagar, encosta em 19 e meio. Cronometrem no ensaio,
porque o tempo real depende muito de quem está falando.

Onde aparece `[A]`, `[B]` e `[C]`, fale o nome da pessoa.

---

## Divisão

| Quem | Slides | Tempo |
|---|---|---|
| **Aluno A** | capa, 1, 2, 3, 4, 5, 6 | 6 min 50 s |
| **Aluno B** | 7, 8, 9 · depois 13, 14 | 6 min 00 s |
| **Aluno C** | 10, 11, 12 · depois 15, 16 | 5 min 40 s |

O B e o C falam duas vezes. A segunda entrada de cada um é o bloco de
comparação, e é proposital: quem apresentou o método comenta o resultado dele.

Os slides de seção (só o nome do bloco, fundo claro) são a deixa da troca.
Passe rápido por eles, sem falar nada.

---

# ABERTURA — Aluno A

## Capa · 15 s

> Boa tarde. Nosso trabalho é sobre limiarização, que é segmentar uma imagem
> escolhendo um valor de corte de intensidade. A gente aplicou três técnicas
> numa imagem e num vídeo de vinte segundos e comparou o comportamento delas.
> Eu vou falar do método de Otsu, o [B] fala do Multi-Otsu e o [C] fala da
> limiarização adaptativa.

---

## Slide 1 · Limiarização · 1 min

> A ideia da limiarização é essa aqui. Para cada pixel, se a intensidade for
> maior que um valor T, ele vira objeto. Se for menor ou igual, vira fundo. É
> uma comparação por pixel, só isso. Não tem treinamento, não tem rede neural,
> roda em qualquer máquina.
>
> O problema todo está em escolher esse T. E é aí que as três técnicas se
> separam.
>
> O Otsu escolhe um único T para a imagem inteira, olhando o histograma. O
> Multi-Otsu escolhe vários limiares, para quando a cena tem mais de duas
> populações de cinza. E a adaptativa escolhe um T diferente para cada pixel.
>
> Guardem essa diferença, porque ela explica quase todos os resultados que a
> gente vai mostrar.

---

## Slide 2 · Modelo de formação da imagem · 1 min 10 s

*Este é o slide que amarra a apresentação inteira. Não corra nele.*

> Antes de entrar nos métodos, esse slide aqui é o mais importante.
>
> Uma imagem é o produto de duas coisas. A refletância, que é propriedade do
> objeto, e é o que a gente quer segmentar. E a iluminação, que é quanta luz
> chega em cada ponto da cena. A câmera entrega o produto dos dois, não a
> refletância sozinha.
>
> Olhem as três imagens. À esquerda a refletância, a cena ideal. No meio a
> iluminação, que aqui é um gradiente, mais clara em cima e mais escura embaixo.
> À direita o que a câmera registra.
>
> Por que isso importa: um limiar global só faz sentido se a iluminação for
> aproximadamente constante na imagem toda. Se a luz varia dentro da cena, o
> fundo de uma região escura pode ficar mais escuro que o objeto de uma região
> clara. E aí não existe valor de T nenhum que resolva.

---

## Slide 3 · Material e método · 55 s

> Para testar isso, a gente gerou um vídeo de vinte segundos, seiscentos
> quadros, dividido em quatro fases.
>
> Na primeira a iluminação é uniforme, é o caso ideal. Na segunda tem um
> holofote varrendo a cena, então a luz varia no espaço. Na terceira a cena
> escurece por inteiro, a luz varia no tempo. E na quarta a gente aumenta o
> ruído e derruba o contraste.
>
> A vantagem de gerar o vídeo é que a gente sabe exatamente qual é a resposta
> certa em cada quadro. Isso deixa a gente medir, e não só achar que ficou
> melhor.
>
> A métrica é o IoU, interseção sobre união. Não usamos acurácia porque o objeto
> ocupa só doze por cento da área. Quem respondesse "é tudo fundo" teria oitenta
> e sete por cento de acurácia e IoU zero.

---

# BLOCO 1: OTSU — Aluno A

## Slide 4 · Método de Otsu · 1 min 20 s

*Slide de fórmula. Fale devagar e aponte cada termo.*

> O Otsu é de 1979, e a ideia é tratar o histograma como uma distribuição de
> probabilidade.
>
> Um limiar t separa os pixels em duas classes. Cada classe tem um peso, que é a
> fração de pixels que caiu nela, e uma média.
>
> A partir disso ele define duas variâncias. A variância dentro das classes mede
> o quanto os pixels estão espalhados dentro de cada grupo, e a gente quer que
> ela seja pequena. A variância entre as classes mede o quanto as médias dos
> dois grupos estão distantes uma da outra, e essa a gente quer grande.
>
> Parece que são dois objetivos separados, mas não são. Existe uma identidade: a
> variância total da imagem é a soma das duas. E a variância total não depende de
> t, ela é uma propriedade da imagem.
>
> Então maximizar uma é exatamente a mesma coisa que minimizar a outra. O Otsu
> escolhe maximizar a variância entre classes, porque essa fórmula só precisa das
> médias, enquanto a outra precisaria calcular a variância de cada classe. Sai
> mais barato.
>
> O custo total é O de N, uma passada pela imagem. Deu zero vírgula dezesseis
> milissegundos por quadro.

---

## Slide 5 · Otsu: histograma e critério · 1 min

> Aqui dá para ver o método funcionando.
>
> À esquerda, o histograma da imagem, com o limiar que o Otsu escolheu, que foi
> 125.
>
> À direita, as duas curvas. A azul é a variância entre classes, a vermelha é a
> variância dentro das classes.
>
> Reparem que o pico de uma cai exatamente no vale da outra. E a soma das duas é
> aquela linha tracejada lá em cima, que é constante, exatamente como a
> identidade do slide anterior diz.
>
> O limiar escolhido é o do pico. Não tem chute e não tem parâmetro para ajustar.
> O método lê o histograma e devolve o número.

---

## Slide 6 · Otsu no vídeo · 1 min

*Fecha o bloco do A. Termine olhando para o [B].*

> E o que acontece no vídeo.
>
> Na fase três, em que a cena inteira escurece, o Otsu baixa o limiar sozinho de
> 147 para 47. É uma queda de três vezes. E o IoU fica em zero vírgula noventa e
> quatro, praticamente sem mudar. Ou seja, o Otsu é invariante a mudança global
> de iluminação, porque ele recalcula o limiar em cada quadro.
>
> Agora a fase dois, com o holofote. O IoU cai para zero vírgula cento e trinta e
> nove. O método continua escolhendo o melhor T possível. Só que o melhor T não
> resolve.
>
> Então a frase correta não é que o Otsu falha com iluminação variável. Ele falha
> com iluminação que varia *dentro* da imagem. Essa distinção é o fio da
> apresentação inteira.
>
> Passo para o [B].

---

# BLOCO 2: MULTI-OTSU — Aluno B

## Slide 7 · Multi-Otsu · 1 min 5 s

> O Otsu parte de uma hipótese forte, que é a cena ter exatamente duas
> populações de cinza. Na prática isso quase nunca acontece.
>
> Um documento digitalizado tem papel, tinta e sombra. Uma radiografia tem fundo,
> tecido mole e osso. Uma imagem de satélite tem água, vegetação e solo. Se você
> força duas classes onde tem três, você funde duas populações diferentes no
> mesmo rótulo, e essa informação não volta mais.
>
> O Multi-Otsu generaliza o critério. Em vez de duas classes, são M classes e M
> menos um limiares. A fórmula é a mesma soma de antes, agora percorrendo todas
> as classes.
>
> Tem uma diferença prática importante: a saída não é uma imagem binária, é um
> mapa de rótulos. Se você quiser uma máscara binária, precisa decidir em qual
> limiar cortar. E essa decisão é sua, não do algoritmo. Isso vai importar daqui
> a dois slides.

---

## Slide 8 · O custo da busca · 1 min 20 s

*O slide mais técnico do bloco. A história do código travando funciona bem,
não pule.*

> Só que generalizar cria um problema de custo.
>
> Com dois limiares são trinta e dois mil combinações, tranquilo. Com três
> limiares já são dois milhões e setecentos mil. Com quatro limiares são cento e
> setenta e dois milhões de combinações.
>
> E isso não é teórico. A nossa primeira versão do código usava busca exaustiva.
> Para M igual a três rodou em milissegundos. Para M igual a quatro demorou
> minutos. Para M igual a cinco travou, e a gente teve que matar o processo.
>
> A solução tem duas partes. A primeira é pré-computar os momentos de todo
> intervalo possível numa tabela, que é o trabalho do Liao, de 2001. Com a tabela
> pronta, avaliar uma combinação fica barato.
>
> A segunda parte é perceber que a função que a gente quer maximizar é uma soma
> sobre intervalos que não se sobrepõem. Essa é exatamente a estrutura que
> programação dinâmica resolve: a gente monta a solução para m intervalos
> reaproveitando a solução para m menos um.
>
> O custo cai para O de M vezes L ao quadrado. Na prática, um milissegundo e
> pouco. E a gente conferiu que dá exatamente os mesmos limiares da biblioteca,
> em doze casos de teste.

---

## Slide 9 · Multi-Otsu: resultado · 1 min 10 s

*Fecha o bloco do B. Termine olhando para o [C].*

> Aqui está o efeito de mudar o número de classes. Com M igual a dois é o Otsu
> normal, um limiar só. Com três e com quatro o histograma vai sendo fatiado em
> mais pedaços, e embaixo dá para ver as regiões que saem disso.
>
> Um cuidado importante: aumentar M sempre aumenta o critério. Se você levasse ao
> extremo, cada nível de cinza viraria uma classe e o critério seria o máximo
> possível. Então não dá para escolher M maximizando o próprio critério. M é um
> hiperparâmetro, você decide olhando a cena.
>
> E aqui está o resultado que eu queria mostrar. Com luz uniforme, cortando no
> limiar de cima, o Multi-Otsu dá IoU zero vírgula novecentos e sessenta e seis.
> É o melhor entre todos os métodos globais que a gente testou.
>
> Mas sob o holofote esse mesmo corte cai para zero vírgula cento e trinta e
> oito, enquanto cortar no limiar de baixo mantém zero vírgula setecentos e
> trinta e um. Mesmo método, mesmo quadro, mesmo histograma. Cinco vezes de
> diferença só na escolha de onde cortar. A classe mais escura aguenta melhor
> porque o holofote não consegue empurrar o fundo até lá embaixo.
>
> Passo para o [C].

---

# BLOCO 3: ADAPTATIVA — Aluno C

## Slide 10 · Limiarização adaptativa · 1 min 10 s

> Voltando ao problema que o [A] levantou no começo: quando a luz varia dentro da
> imagem, nenhum valor único de T funciona.
>
> A saída é parar de procurar um número.
>
> Olhem os três painéis. No meio, o que o Otsu produz: o número 128, um só,
> válido para a imagem inteira. À direita, o que a adaptativa produz: uma imagem
> inteira de limiares, do mesmo tamanho da entrada. Reparem que ela é
> basicamente uma estimativa da iluminação da cena.
>
> O cálculo é simples. Para cada pixel, você olha a vizinhança dele, tira a
> média, e subtrai uma constante C. Pronto, esse é o limiar daquele pixel.
>
> E funciona porque a iluminação varia devagar no espaço. Dentro de uma janela
> pequena ela é praticamente constante, então ali um limiar local faz sentido.
>
> O OpenCV oferece duas versões, uma com média simples e outra com média
> ponderada por uma gaussiana. A diferença entre elas é só o peso que cada
> vizinho recebe.

---

## Slide 11 · Os dois parâmetros · 1 min 20 s

> A adaptativa tem dois parâmetros, e os dois importam muito.
>
> O primeiro é o tamanho do bloco. Olhem a figura. No meio, com bloco de trinta e
> cinco pixels, os quadrados aparecem ocos, só com o contorno. O motivo é que um
> pixel no centro de um objeto grande tem a vizinhança inteira dentro do próprio
> objeto. Então a média local fica igual à intensidade dele, e o teste falha. Só
> a borda sobrevive.
>
> À direita, com bloco de cento e cinquenta e um, o objeto aparece cheio. A regra
> é essa: o bloco precisa ser maior que o objeto, e ao mesmo tempo menor que a
> escala em que a iluminação varia.
>
> Isso quase enganou a gente. Com o valor padrão de trinta e cinco, o IoU médio no
> vídeo foi zero vírgula quinhentos e treze, pior que o Otsu. Com cento e
> cinquenta e um foi zero vírgula oitocentos e sessenta e três. O método não era
> ruim, o parâmetro é que estava fora da escala do problema.
>
> O segundo parâmetro é a constante C. Numa região lisa, a intensidade do pixel é
> praticamente igual à média da vizinhança dele. Sem o C, o resultado seria o
> sinal do ruído, ou seja, ruído puro. O C cria uma margem: só é objeto o pixel
> que for pelo menos C níveis mais escuro que a vizinhança.

---

## Slide 12 · Custo · 50 s

> Uma pergunta natural é se isso não fica caro, porque agora é uma média por
> pixel, e não uma conta só para a imagem toda.
>
> Calculando direto ficaria caro mesmo. Mas com a imagem integral, que é uma
> tabela de somas acumuladas, a soma de qualquer retângulo sai com quatro
> acessos. O custo fica O de N, independente do tamanho da janela.
>
> Na tabela, todos os métodos rodam em tempo real a trinta quadros por segundo. O
> Otsu é o mais rápido, com zero vírgula dezesseis milissegundos. A adaptativa com
> bloco grande é a mais cara, vinte e dois milissegundos, que ainda dá quarenta e
> quatro quadros por segundo.

---

# COMPARAÇÃO — Aluno B volta

## Slide 13 · O limiar ao longo do vídeo · 1 min

> Esse gráfico mostra o valor do limiar que cada método escolheu ao longo dos
> vinte segundos. As faixas coloridas ao fundo são as quatro fases.
>
> Na fase um está tudo plano, porque nada muda na cena.
>
> A parte interessante é a fase três, a faixa azul. A cena está escurecendo, e
> olhem o que acontece: todos os métodos automáticos descem junto. O Otsu vai de
> cento e quarenta e sete para quarenta e sete. E, como o [A] mostrou, a
> qualidade não cai junto.
>
> A linha cinza é o limiar fixo em cento e vinte e sete. Ela não se mexe. Em
> algum momento ela cruza o valor certo, por acaso, e nesse instante acerta quase
> tudo. Dois segundos depois está completamente errada.
>
> É a diferença entre estar certo por acidente e estar certo porque o método se
> ajusta.

---

## Slide 14 · Resultados no vídeo · 1 min 30 s

*Slide mais importante do fechamento. Dê tempo para a plateia ler a tabela
antes de falar.*

> Essa é a tabela principal. IoU médio de cada método em cada uma das quatro
> fases.
>
> Tem duas coisas para olhar aqui.
>
> A primeira é a coluna da fase dois, o holofote. O limiar fixo dá zero vírgula
> cento e quarenta. O Otsu dá zero vírgula cento e trinta e nove. O Multi-Otsu no
> corte de cima dá zero vírgula cento e trinta e oito. São praticamente o mesmo
> número.
>
> Isso quer dizer que escolher o limiar global de forma ótima não comprou nada. O
> problema não é qual T você escolhe, é o fato de existir um T único. Na mesma
> coluna, a adaptativa dá zero vírgula novecentos e cinquenta e seis.
>
> A segunda coisa é a coluna da fase quatro, o ruído. A ordem se inverte. Agora o
> Otsu é o melhor, com zero vírgula oitocentos e trinta e quatro, e a adaptativa é
> a pior.
>
> E a razão é estrutural. O limiar global é estimado a partir de trezentos mil
> pixels, então o ruído praticamente se cancela na média. O limiar local é
> estimado a partir dos pixels de uma janelinha, e herda o ruído dela.
>
> Reparem que é a mesma característica gerando o ponto forte e o ponto fraco. A
> adaptativa decide olhando pouca coisa. Isso a salva do holofote e a derruba no
> ruído.

---

# FECHAMENTO — Aluno C volta

## Slide 15 · Onde cada método erra · 1 min

*Se sobrar tempo, é aqui que vale mostrar o `mosaico_comparativo.mp4`.
São 20 segundos. Deixe a janela já aberta antes de começar a apresentação.*

> Aqui dá para ver os erros separados por tipo. Vermelho é falso positivo, quando
> o método declara objeto e não é. Azul é falso negativo, quando ele perde o
> objeto.
>
> Linha de cima, o holofote. O Otsu pinta a imagem quase toda de vermelho. Ele
> classificou como objeto todo o fundo que o holofote deixou escuro. As duas
> colunas da direita estão limpas.
>
> Linha de baixo, o ruído. Agora é a adaptativa que está coberta de vermelho, com
> falso positivo espalhado por toda parte, enquanto o Otsu está bem melhor.
>
> É a mesma figura mostrando os dois resultados opostos. A ordem dos métodos se
> inverte de uma linha para a outra.

---

## Slide 16 · Conclusão · 1 min 10 s

> Fechando, três pontos.
>
> Primeiro, e é o que a gente mais quer que fique: o limiar global falha com
> variação espacial de iluminação, não com variação temporal. O mesmo Otsu deu
> zero vírgula novecentos e quarenta e três na fase em que a cena escurece
> inteira, e zero vírgula cento e trinta e nove na fase do holofote.
>
> Segundo, a limiarização adaptativa troca robustez à iluminação por fragilidade
> ao ruído. E as duas coisas vêm da mesma causa, que é decidir a partir de poucos
> pixels.
>
> Terceiro, nenhum método ganhou nas quatro fases. Então a pergunta não é qual
> técnica é melhor, é qual hipótese a sua cena satisfaz.
>
> Sobre limitações: o vídeo é sintético, e isso foi uma escolha, porque é o que
> permite ter a resposta certa em todos os seiscentos quadros. A gente usou um
> único ajuste de parâmetros por método. E não fizemos pós-processamento
> morfológico, que provavelmente melhoraria bastante o resultado da fase de
> ruído.

---

## Obrigado · 10 s

> É isso. Obrigado, e a gente está à disposição para perguntas.

*O slide 17, de referências, é o último do PDF e não se fala nada nele. Deixe
aberto durante as perguntas, ou avance até ele se alguém pedir a fonte de
alguma coisa.*

---

# Colas

## Números que não podem sair errados

| Onde | Número |
|---|---|
| Otsu na fase 3 (escurece) | limiar cai de **147 para 47**, IoU fica **0,94** |
| Otsu na fase 2 (holofote) | IoU **0,139** |
| Adaptativa na fase 2 | IoU **0,956** |
| Adaptativa na fase 4 (ruído) | IoU **0,575** · Otsu na mesma fase: **0,834** |
| Multi-Otsu, corte de cima / de baixo na fase 2 | **0,138** contra **0,731** |
| Bloco 35 contra bloco 151 | IoU médio **0,513** contra **0,863** |
| Combinações com 4 limiares | **172 milhões** |
| Objeto ocupa | **12 %** da área |

Se esquecer um número, fale a ordem de grandeza e siga. Ninguém vai conferir a
terceira casa decimal, mas todo mundo percebe quando alguém trava.

## Se estiver atrasado

Corte nesta ordem, que o argumento continua de pé:

1. **Slide 12 (Custo)** — diga só "e tudo isso roda em tempo real" e passe.
2. **Slide 13 (limiar ao longo do vídeo)** — a tabela do slide 14 já mostra a
   conclusão.
3. **Slide 5 (histograma e critério)** — se o slide 4 foi bem explicado, esse é
   reforço visual.

Isso economiza uns 3 minutos.

## Se sobrar tempo

Toque o `../resultados/video/mosaico_comparativo.mp4` inteiro, 20 segundos, e
comente enquanto roda: os seis painéis são o mesmo quadro segmentado por métodos
diferentes, e o IoU de cada um aparece atualizando no canto.

## Perguntas prováveis

As respostas estão em [`../docs/05_roteiro_apresentacao.md`](../docs/05_roteiro_apresentacao.md),
seção 5.3. As três mais prováveis:

**"Por que vídeo sintético e não real?"**
Para ter a resposta certa em cada um dos 600 quadros e poder medir IoU quadro a
quadro, e para controlar um fator de degradação por vez. O código aceita vídeo
real, só que aí não dá para calcular IoU.

**"Vocês implementaram ou só chamaram a biblioteca?"**
Os dois. Otsu, Multi-Otsu, adaptativa e Sauvola estão implementados do zero em
NumPy, e foram validados contra o scikit-image e o OpenCV. Otsu e Multi-Otsu dão
limiar idêntico. Na adaptativa a concordância fica entre 99,3 e 99,9 %, e a diferença
está em pixels onde o limiar cai a menos de meio nível de cinza da intensidade,
porque o OpenCV calcula em inteiro e a gente em ponto flutuante.

**"Otsu não falha com iluminação variável?"**
Depende do tipo. Com variação global ele é praticamente invariante, deu 0,943 na
fase 3. Com variação espacial ele colapsa, 0,139 na fase 2.
