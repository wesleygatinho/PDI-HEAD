# Taxa de acerto em relação ao modelo Heavy

Vídeo: `video_20261121_145614.mp4` · 610 quadros · 610 quadros avaliados (ombros e quadris visíveis no Heavy) · 19429 landmarks avaliados.

| Métrica | Lite | Full |
|---|---:|---:|
| **Taxa de acerto (PCK@0,2)** | **99,6 %** | **99,9 %** |
| PCK@0,1 (rigoroso) | 96,6 % | 99,4 % |
| Quadros corretos (≥ 90% dos pontos) | 98,2 % | 100,0 % |
| Erro médio (px) | 6,4 | 4,9 |
| Erro mediano (px) | 4,9 | 4,2 |
| Erro médio (% do tronco) | 3,0 | 2,3 |
| Detecção nos quadros avaliados | 100,0 % | 100,0 % |
| Visibilidade concorda com o Heavy | 100,0 % | 100,0 % |

## Por parte do corpo (PCK@0,2)

| Parte | Pontos visíveis por quadro | Lite | Full |
|---|---:|---:|---:|
| Rosto | 11,0 de 11 | 100,0 % | 100,0 % |
| Tronco | 4,0 de 4 | 100,0 % | 100,0 % |
| Braços | 3,6 de 4 | 100,0 % | 100,0 % |
| Mãos | 5,4 de 6 | 100,0 % | 100,0 % |
| Pernas | 3,9 de 4 | 98,8 % | 99,7 % |
| Pés | 4,0 de 4 | 97,9 % | 99,8 % |

"–" = parte do corpo fora do quadro (menos de 1 s de dados): não avaliada.

## Por pose (PCK@0,2)

| Métrica | Lite | Full |
|---|---:|---:|
| Braços para cima | 100,0 % | 100,0 % |
| Braços abertos | 100,0 % | 100,0 % |
| Braços cruzados | 100,0 % | 100,0 % |
| Agachamento | 100,0 % | 100,0 % |
| De perfil | 99,9 % | 100,0 % |
| Em uma perna | 96,4 % | 99,5 % |
| Andando de lado | 100,0 % | 100,0 % |
| Movimento livre | 100,0 % | 100,0 % |

## Como ler

- **PCK@α**: % de landmarks a menos de α × comprimento do tronco do ponto dado pelo Heavy. Com o tronco medindo ~200 px, α = 0,2 equivale a ~40 px e α = 0,1 a ~20 px.
- Só entram landmarks que o Heavy considera visíveis, em quadros com ombros e quadris visíveis.
- **Limitação**: o Heavy é a referência, não um gabarito anotado à mão. Os três modelos têm a mesma arquitetura e treino, então erros que os três cometem juntos não aparecem aqui; o número mede **concordância com o modelo mais preciso**.
