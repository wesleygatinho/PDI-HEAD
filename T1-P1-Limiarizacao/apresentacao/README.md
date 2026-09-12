# Apresentação (Beamer / LaTeX)

Duas versões. **Use a v2.**

| Arquivo | Slides | Duração | Para quê |
|---|---|---|---|
| **`apresentacao_v2.tex`** | **17** | **20 min** | **a apresentação** |
| `apresentacao.tex` | 36 | 35–40 min | versão longa, serve de material de consulta |

---

## Antes de compilar

Edite as quatro linhas no topo de `apresentacao_v2.tex`:

```latex
\newcommand{\alunoA}{Nome do Aluno A}
\newcommand{\alunoB}{Nome do Aluno B}
\newcommand{\alunoC}{Nome do Aluno C}
\newcommand{\professor}{Prof. Fulano de Tal}
```

## Compilar

Precisa ser XeLaTeX (o tema usa as fontes Fira):

```bash
latexmk -xelatex apresentacao_v2.tex
```

Para limpar os auxiliares depois:

```bash
latexmk -c
```

---

## Divisão dos 20 minutos

| Slides | Conteúdo | Quem | Tempo |
|---|---|---|---|
| 1–3 | Limiarização, modelo $I = R\cdot L$, material e método | abertura | 4 min |
| 4–6 | Método de Otsu, critério, resultado no vídeo | **Aluno A** | 5 min |
| 7–9 | Multi-Otsu, custo da busca, resultado | **Aluno B** | 5 min |
| 10–12 | Limiar local, os dois parâmetros, custo | **Aluno C** | 5 min |
| 13–17 | Comparação e conclusão | fechamento | 4 min |

Os slides de seção (só o nome do bloco) marcam a troca de apresentador.

---

## Se precisar cortar mais

Na ordem em que eu tiraria, sem quebrar o argumento:

1. **Custo** (slide 12) — o dado importante, que todos rodam em tempo real,
   cabe numa frase falada.
2. **O limiar ao longo do vídeo** (slide 13) — o gráfico é bonito, mas a tabela
   de resultados já mostra a conclusão.
3. **Otsu no vídeo** (slide 6) — dá para adiantar o ponto na tabela final.

Isso deixa 14 slides, uns 16 minutos.

---

## As figuras

`figuras/` tem versões feitas para projeção: menos elementos e fonte grande. As
de `../resultados/graficos/` são densas demais para projetor, servem para o
relatório escrito.

Para regerá-las (depois de rodar `executar_tudo.py`):

```bash
python figuras/gerar_figuras_slides.py
```

---

## Na hora de apresentar

- Deixe `../resultados/video/mosaico_comparativo.mp4` aberto numa janela ao
  lado. São 20 segundos e mostram os seis métodos em paralelo com o IoU
  atualizando. Vale mais que qualquer slide estático.
- Se o projetor for 4:3, troque `aspectratio=169` por `aspectratio=43` na
  primeira linha do `.tex`.
