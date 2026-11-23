# Taxa de acerto em relação ao modelo Heavy

Vídeo: `video_20261121_143943.mp4` · 612 quadros · 602 quadros avaliados (ombros e quadris visíveis no Heavy) · 13428 landmarks avaliados.

| Métrica | Lite | Full |
|---|---:|---:|
| **Taxa de acerto (PCK@0,2)** | **99,4 %** | **99,8 %** |
| PCK@0,1 (rigoroso) | 96,8 % | 99,2 % |
| Quadros corretos (≥ 90% dos pontos) | 99,0 % | 99,8 % |
| Erro médio (px) | 10,7 | 7,7 |
| Erro mediano (px) | 7,9 | 5,8 |
| Erro médio (% do tronco) | 3,1 | 2,2 |
| Detecção nos quadros avaliados | 100,0 % | 100,0 % |
| Visibilidade concorda com o Heavy | 96,9 % | 98,5 % |

## Por parte do corpo (PCK@0,2)

| Parte | Pontos visíveis por quadro | Lite | Full |
|---|---:|---:|---:|
| Rosto | 11,0 de 11 | 100,0 % | 100,0 % |
| Tronco | 4,0 de 4 | 99,9 % | 100,0 % |
| Braços | 3,3 de 4 | 99,7 % | 99,9 % |
| Mãos | 3,9 de 6 | 99,2 % | 99,7 % |
| Pernas | 0,1 de 4 | – | – |
| Pés | 0,0 de 4 | – | – |

"–" = parte do corpo fora do quadro (menos de 1 s de dados): não avaliada.

## Por pose (PCK@0,2)

| Métrica | Lite | Full |
|---|---:|---:|
| Braços para cima | 100,0 % | 100,0 % |
| Braços abertos | 100,0 % | 100,0 % |
| Braços cruzados | 100,0 % | 99,9 % |
| Agachamento | 99,9 % | 100,0 % |
| De perfil | 100,0 % | 100,0 % |
| Em uma perna | 98,3 % | 99,7 % |
| Andando de lado | 97,4 % | 99,1 % |
| Movimento livre | 99,6 % | 99,7 % |

## Como ler

- **PCK@α**: % de landmarks a menos de α × comprimento do tronco do ponto dado pelo Heavy. Com o tronco medindo ~200 px, α = 0,2 equivale a ~40 px e α = 0,1 a ~20 px.
- Só entram landmarks que o Heavy considera visíveis, em quadros com ombros e quadris visíveis.
- **Limitação**: o Heavy é a referência, não um gabarito anotado à mão. Os três modelos têm a mesma arquitetura e treino, então erros que os três cometem juntos não aparecem aqui; o número mede **concordância com o modelo mais preciso**.
