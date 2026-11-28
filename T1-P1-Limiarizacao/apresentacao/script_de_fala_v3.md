# Script de fala — apresentação v3

Texto para falar em cada slide de `apresentacao_v3.pdf`.
Leia duas ou três vezes antes; não é para ler no papel durante a apresentação.

Onde aparece `[A]`, `[B]` e `[C]`, fale o nome da pessoa.

---

## Divisão

| Quem | Slides | Tempo |
|---|---|---|
| **Aluno A** | capa, 1, 2, 3, 4, 5, 6 | 6 min 40 s |
| **Aluno B** | 7, 8, 9 · depois 13, 14 | 6 min 00 s |
| **Aluno C** | 10, 11, 12 · depois 15, 16 | 5 min 50 s |

B e C falam duas vezes: a segunda entrada é o bloco de comparação, e quem
apresentou o método é quem comenta o resultado dele.

Os slides de seção (fundo claro, só o nome do bloco) são a deixa da troca.
Passe rápido, sem falar nada.

---

# ABERTURA — Aluno A

## Capa · 15 s

> Boa tarde. Nosso trabalho é sobre limiarização, que é segmentar uma imagem
> escolhendo um valor de corte de intensidade. A gente aplicou três técnicas em
> imagens e num vídeo de vinte segundos e comparou o comportamento delas. Eu vou
> falar do método de Otsu, o [B] fala do Multi-Otsu e o [C] da limiarização
> adaptativa.

---

## Slide 1 · Limiarização · 1 min

> A ideia é essa aqui. Para cada pixel, se a intensidade for maior que um valor
> T, ele vira objeto. Se for menor ou igual, vira fundo. É uma comparação por
> pixel, só isso. Não tem treinamento, não tem rede neural, roda em qualquer
> máquina.
>
> O problema todo está em escolher esse T. E é aí que as três técnicas se
> separam.
>
> O Otsu escolhe um único T para a imagem inteira, olhando o histograma. O
> Multi-Otsu escolhe vários limiares, para quando a cena tem mais de duas
> populações de cinza. E a adaptativa escolhe um T diferente para cada pixel.

---

## Slide 2 · Modelo de formação da imagem · 1 min 10 s

*O slide que amarra a apresentação inteira. Não corra.*

> Antes dos métodos, esse slide é o mais importante.
>
> Uma imagem é o produto de duas coisas. A refletância, que é propriedade do
> objeto, e é o que a gente quer segmentar. E a iluminação, que é quanta luz
> chega em cada ponto. A câmera entrega o produto dos dois, nunca a refletância
> sozinha.
>
> Olhem as três imagens. À esquerda a refletância, a cena ideal. No meio a
> iluminação, que aqui é um gradiente. À direita o que a câmera registra.
>
> Por que isso importa: um limiar global só faz sentido se a iluminação for
> aproximadamente constante na imagem toda. Se a luz varia dentro da cena, o
> fundo de uma região escura pode ficar mais escuro que o objeto de uma região
> clara. E aí não existe valor de T nenhum que resolva.

---

## Slide 3 · A mídia usada · 1 min

> Sobre o material. As três primeiras são imagens reais, todas de licença livre.
>
> A primeira é a foto de sudoku do próprio tutorial de limiarização do OpenCV,
> que é uma das páginas que a gente estudou. A segunda é uma ressonância
> magnética do Wikimedia Commons. A terceira é um quadro do vídeo, um time-lapse
> de pôr do sol no Grand Canyon, produzido pelo serviço de parques americano e
> em domínio público.
>
> A quarta é uma cena que a gente gerou. Ela existe por um motivo específico:
> é a única em que sabemos a resposta certa em cada pixel.
>
> Isso importa porque as métricas que a gente usa, IoU e F1, comparam contra um
> gabarito. Em imagem real ninguém sabe a resposta pixel a pixel. Então nas
> imagens reais a comparação é visual e pelo comportamento do limiar, e na cena
> sintética a comparação é numérica.

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
> dois grupos estão distantes, e essa a gente quer grande.
>
> Parece que são dois objetivos separados, mas não são. Existe uma identidade: a
> variância total da imagem é a soma das duas. E a variância total não depende de
> t, ela é uma propriedade da imagem.
>
> Então maximizar uma é exatamente a mesma coisa que minimizar a outra. O Otsu
> maximiza a variância entre classes, porque essa fórmula só precisa das médias,
> enquanto a outra precisaria da variância de cada classe. Sai mais barato.
>
> O custo total é O de N, uma passada pela imagem. Deu zero vírgula dezesseis
> milissegundos por quadro.

---

## Slide 5 · Otsu: histograma e critério · 1 min

> Aqui dá para ver o método funcionando, sobre um quadro do nosso vídeo.
>
> À esquerda, o histograma, com o limiar que o Otsu escolheu, que foi 70.
>
> À direita, as duas curvas. A azul é a variância entre classes, a vermelha é a
> variância dentro das classes.
>
> Reparem que o pico de uma cai exatamente no vale da outra. E a soma das duas é
> aquela linha tracejada lá em cima, que é constante, exatamente como a
> identidade do slide anterior diz.
>
> O limiar escolhido é o do pico. Não tem chute e não tem parâmetro para
> ajustar. O método lê o histograma e devolve o número.

---

## Slide 6 · Otsu no vídeo real · 1 min 10 s

*Fecha o bloco do A. Termine olhando para o [B].*

> Agora o vídeo. São vinte segundos de um pôr do sol, e a cena vai escurecendo.
>
> Olhem os quatro quadros. O brilho médio cai de 106 para 62, quase pela metade.
> E o Otsu, sem ninguém mexer em nada, desce o limiar de 70 para 46.
>
> Isso é o ponto: nenhum parâmetro foi ajustado entre um quadro e outro. É
> exatamente a mesma linha de código, lendo um histograma diferente a cada
> quadro. O método acompanha a mudança de luz sozinho.
>
> Guardem isso, porque no fim a gente vai mostrar que essa mesma facilidade
> desaparece quando a luz varia dentro do quadro, e não ao longo do tempo.
>
> Passo para o [B].

---

# BLOCO 2: MULTI-OTSU — Aluno B

## Slide 7 · Multi-Otsu · 1 min

> O Otsu parte de uma hipótese forte, que é a cena ter exatamente duas
> populações de cinza. Na prática isso quase nunca acontece.
>
> Um documento digitalizado tem papel, tinta e sombra. Uma ressonância tem
> fundo, tecido e osso. Se você força duas classes onde tem três, funde duas
> populações diferentes no mesmo rótulo, e essa informação não volta.
>
> O Multi-Otsu generaliza o critério. Em vez de duas classes, são M classes e M
> menos um limiares, com a mesma soma de antes percorrendo todas as classes.
>
> Uma diferença prática importante: a saída não é uma imagem binária, é um mapa
> de rótulos. Se você quiser uma máscara binária, precisa decidir em qual limiar
> cortar. E essa decisão é sua, não do algoritmo.

---

## Slide 8 · O custo da busca · 1 min 20 s

*A história do código travando funciona bem em sala. Não pule.*

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
> A segunda é perceber que a função que a gente quer maximizar é uma soma sobre
> intervalos que não se sobrepõem. Essa é exatamente a estrutura que programação
> dinâmica resolve: monta a solução para m intervalos reaproveitando a solução
> para m menos um.
>
> O custo cai para O de M vezes L ao quadrado. Na prática, um milissegundo e
> pouco. E a gente conferiu que dá exatamente os mesmos limiares da biblioteca.

---

## Slide 9 · Multi-Otsu na ressonância · 1 min 10 s

*Fecha o bloco do B. Termine olhando para o [C].*

> E aqui está o método numa imagem de verdade.
>
> Essa é uma ressonância magnética cerebral, corte axial. No histograma, no
> segundo painel, dá para ver três populações: um pico enorme perto do zero, que
> é o fundo preto; uma região larga no meio, que é o tecido cerebral; e uma cauda
> à direita, que é a gordura e o osso do crânio.
>
> No terceiro painel está o Otsu, com um limiar e duas classes. Ele responde bem
> uma pergunta só: cérebro ou não-cérebro.
>
> No quarto está o Multi-Otsu com três classes. Agora o fundo, o tecido e a
> gordura do crânio saem separados, cada um com sua cor. É a mesma imagem e o
> mesmo histograma, só que particionado em três pedaços em vez de dois.
>
> Um cuidado importante: aumentar M sempre aumenta o critério. Se levasse ao
> extremo, cada nível de cinza viraria uma classe. Então não dá para escolher M
> maximizando o próprio critério. M é um hiperparâmetro, você decide olhando a
> cena.
>
> Passo para o [C].

---

# BLOCO 3: ADAPTATIVA — Aluno C

## Slide 10 · Limiarização adaptativa · 1 min 5 s

> Voltando ao problema que o [A] levantou no começo: quando a luz varia dentro da
> imagem, nenhum valor único de T funciona.
>
> A saída é parar de procurar um número.
>
> Olhem os três painéis. No meio, o que o Otsu produz para essa foto: o número
> 96, um só, válido para a imagem inteira. À direita, o que a adaptativa produz:
> uma imagem inteira de limiares, do mesmo tamanho da entrada.
>
> E reparem numa coisa: esse mapa de limiares é praticamente uma foto da
> iluminação da cena. Dá para ver onde a luz bateu e onde ficou sombra. É
> exatamente isso que permite ao método acompanhar a sombra.
>
> O cálculo é simples. Para cada pixel, você olha a vizinhança dele, tira a
> média, e subtrai uma constante C.

---

## Slide 11 · Os dois parâmetros · 1 min 15 s

> A adaptativa tem dois parâmetros, e os dois importam.
>
> O primeiro é o tamanho do bloco. No segundo painel o bloco é pequeno demais, e
> a grade sai grossa e borrada, porque a vizinhança cai quase toda dentro do
> próprio traço. No terceiro, com bloco de trinta e um, fica limpo. A regra é
> que o bloco precisa ser maior que o traço que você quer detectar, e ao mesmo
> tempo menor que a escala em que a iluminação varia.
>
> O segundo parâmetro é a constante C. No quarto painel eu zerei o C, e vejam o
> que acontece: a imagem inteira vira ruído.
>
> O motivo é direto. Numa região lisa, a intensidade do pixel é praticamente
> igual à média da vizinhança dele. Sem o C, o teste "sou mais escuro que a minha
> vizinhança" vira o sinal do ruído, e o resultado é aleatório. O C cria uma
> margem: só é objeto o pixel que for pelo menos C níveis mais escuro que a
> vizinhança.

---

## Slide 12 · A demonstração · 1 min 10 s

*O slide mais forte do bloco. Deixe a plateia olhar antes de falar.*

> E essa é a demonstração.
>
> A foto é a do próprio tutorial de limiarização do OpenCV. É um sudoku de
> jornal fotografado, e o canto superior esquerdo está na sombra.
>
> No meio, o Otsu. Ele escolhe o melhor limiar global possível, aquele que
> maximiza a variância entre classes, e mesmo assim perde a região inteira da
> sombra. A grade some, os números somem.
>
> E não é falha de implementação. É que com um único limiar não tem saída: o
> papel na sombra é mais escuro que a tinta preta do lado iluminado. Qualquer
> número que você escolher vai errar de um lado ou do outro.
>
> À direita, a adaptativa, com bloco trinta e um e C igual a sete. A grade
> inteira volta, e todos os dígitos ficam legíveis, tanto na parte clara quanto
> na sombra.

---

# COMPARAÇÃO — Aluno B volta

## Slide 13 · O limiar ao longo do vídeo · 1 min

> Esse gráfico mostra o limiar que cada método escolheu ao longo dos vinte
> segundos do vídeo.
>
> As três linhas coloridas são os métodos automáticos. Todas descem juntas,
> acompanhando o pôr do sol. O Otsu, em azul, vai de setenta para quarenta e
> seis.
>
> A linha cinza reta é o limiar fixo em cento e vinte e sete. Ela não se mexe.
>
> E olhem onde ela está: acima de todas as outras o tempo inteiro. Na prática,
> no começo do vídeo ela já classifica sessenta e sete por cento da imagem como
> objeto, e no fim classifica noventa e nove vírgula seis por cento. Ou seja,
> quase a imagem toda.
>
> É a diferença entre um método que se ajusta e um número que alguém chutou uma
> vez.

---

## Slide 14 · Para medir, é preciso gabarito · 1 min 30 s

*Dê tempo para a plateia ler a tabela antes de falar.*

> Até aqui a comparação foi visual. Para medir de verdade a gente precisa saber
> a resposta certa, e em imagem real ninguém sabe.
>
> Por isso geramos também uma cena sintética de vinte segundos, com gabarito em
> cada quadro, dividida em quatro fases. Na primeira a luz é uniforme. Na segunda
> tem uma sombra varrendo a cena, então a luz varia no espaço. Na terceira a cena
> escurece inteira, a luz varia no tempo. E na quarta tem ruído forte.
>
> Duas coisas para olhar na tabela.
>
> A coluna da fase dois, a da sombra. O limiar fixo dá zero vírgula cento e
> quarenta, o Otsu dá zero vírgula cento e trinta e nove, e o Multi-Otsu no corte
> de cima dá zero vírgula cento e trinta e oito. Praticamente o mesmo número.
> Escolher o limiar global de forma ótima não comprou nada. O problema não é qual
> T, é existir um T único. Na mesma coluna, a adaptativa dá zero vírgula
> novecentos e cinquenta e seis.
>
> A coluna da fase quatro, a do ruído. Aí a ordem se inverte. O Otsu é o melhor,
> com zero vírgula oitocentos e trinta e quatro, e a adaptativa é a pior.
>
> A razão é estrutural. O limiar global é estimado a partir de trezentos mil
> pixels, então o ruído se cancela na média. O local é estimado a partir de uma
> janelinha, e herda o ruído dela.

---

# FECHAMENTO — Aluno C volta

## Slide 15 · Onde cada método erra · 1 min

*Se sobrar tempo, é aqui que vale mostrar o vídeo do mosaico.*

> Aqui dá para ver os erros separados por tipo. Vermelho é falso positivo, quando
> o método declara objeto e não é. Azul é falso negativo, quando ele perde o
> objeto.
>
> Linha de cima, a fase da sombra. O Otsu pinta a imagem quase toda de vermelho.
> Ele classificou como objeto todo o fundo que a sombra deixou escuro. As duas
> colunas da direita estão limpas.
>
> Linha de baixo, a fase do ruído. Agora é a adaptativa que está coberta de
> vermelho, com falso positivo espalhado por toda parte, enquanto o Otsu está
> bem melhor.
>
> É a mesma figura mostrando os dois resultados opostos. A ordem dos métodos se
> inverte de uma linha para a outra.

---

## Slide 16 · Conclusão · 1 min 10 s

> Fechando, três pontos.
>
> Primeiro, e é o que a gente mais quer que fique: o limiar global falha com
> variação espacial de iluminação, não com variação temporal. No vídeo real a
> cena escureceu por completo e o Otsu acompanhou sem perder qualidade. Na cena
> com sombra ele caiu de zero vírgula novecentos e quarenta e quatro para zero
> vírgula cento e trinta e nove.
>
> Segundo, a adaptativa troca robustez à iluminação por fragilidade ao ruído. E
> as duas coisas vêm da mesma causa, que é decidir a partir de poucos pixels.
>
> Terceiro, nenhum método ganhou em todas as situações. A pergunta não é qual
> técnica é melhor, é qual hipótese a sua cena satisfaz.
>
> Sobre limitações: a parte quantitativa usa cena sintética, porque é o que
> permite ter gabarito. Usamos um único ajuste de parâmetros por método. E não
> fizemos pós-processamento morfológico, que provavelmente melhoraria bastante o
> resultado da fase de ruído.

---

## Obrigado · 10 s

> É isso. Obrigado, e a gente está à disposição para perguntas.

*O slide 17, de créditos e referências, não se fala nada. Deixe aberto durante
as perguntas.*

---

# Colas

## Números que não podem sair errados

| Onde | Número |
|---|---|
| Vídeo real: brilho médio | cai de **106 para 62** |
| Vídeo real: limiar do Otsu | desce de **70 para 46** |
| Vídeo real: limiar fixo T=127 | classifica de **67 %** a **99,6 %** da imagem como objeto |
| Sudoku: limiar do Otsu | **96** |
| Cena sintética, fase da sombra | Otsu **0,139** · adaptativa **0,956** |
| Cena sintética, fase do ruído | Otsu **0,834** · adaptativa **0,575** |
| Combinações com 4 limiares | **172 milhões** |
| Ressonância | Otsu **39** · Multi-Otsu **35 e 109** |

Se esquecer um número, fale a ordem de grandeza e siga.

## Se estiver atrasado

Corte nesta ordem, que o argumento continua de pé:

1. **Slide 13** (limiar ao longo do vídeo) — o slide 6 já mostrou o Otsu
   acompanhando a luz.
2. **Slide 5** (histograma e critério) — se o slide 4 foi bem explicado, é
   reforço visual.
3. **Slide 11** (os dois parâmetros) — mencione o bloco e o C de passagem no
   slide 12.

Isso economiza uns 3 minutos.

## Se sobrar tempo

Toque `../resultados/video/mosaico_comparativo_canyon.mp4`. São 20 segundos.
São oito painéis: o quadro original, o histograma ao vivo com os limiares se
movendo, e os seis métodos. Comente enquanto roda que a linha cinza do
histograma é o limiar fixo, e que ela vai ficando cada vez mais longe de onde
os dados estão.

## Perguntas prováveis

**"Por que vocês não usaram só imagem real?"**
Porque IoU e F1 exigem gabarito, e em imagem real ninguém sabe a resposta pixel
a pixel. Usamos as duas coisas: real para demonstrar o comportamento, sintética
para medir.

**"De onde vieram as imagens?"**
O sudoku é do próprio repositório do OpenCV, é a imagem do tutorial. A
ressonância é do Wikimedia Commons, CC BY-SA. O vídeo é do National Park
Service americano, domínio público. Está tudo no slide de créditos.

**"Vocês implementaram ou só chamaram a biblioteca?"**
Os dois. Otsu, Multi-Otsu, adaptativa e Sauvola estão implementados do zero em
NumPy e validados contra scikit-image e OpenCV. Otsu e Multi-Otsu dão limiar
idêntico. Na adaptativa a concordância fica entre 99,3 e 99,9 %, e a diferença
toda está em pixels onde o limiar cai a menos de meio nível de cinza da
intensidade, porque o OpenCV calcula em inteiro e a gente em ponto flutuante.

**"Otsu não falha com iluminação variável?"**
Depende do tipo. Com variação global ele é praticamente invariante: no vídeo
real a cena escureceu pela metade e ele acompanhou. Com variação espacial ele
colapsa, como no sudoku e na fase 2 da cena sintética.
