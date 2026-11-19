# Taxa de acerto em relação ao modelo Heavy

Vídeo: `video_20261117_224322.mp4` · 1203 quadros · 1203 quadros avaliados (ombros e quadris visíveis no Heavy) · 24376 landmarks avaliados.

| Métrica | Lite | Full |
|---|---:|---:|
| **Taxa de acerto (PCK@0,2)** | **99,8 %** | **99,9 %** |
| PCK@0,1 (rigoroso) | 99,0 % | 99,6 % |
| Quadros corretos (≥ 90% dos pontos) | 99,1 % | 99,4 % |
| Erro médio (px) | 11,6 | 9,2 |
| Erro mediano (px) | 9,7 | 7,8 |
| Erro médio (% do tronco) | 2,9 | 2,3 |
| Detecção nos quadros avaliados | 100,0 % | 100,0 % |
| Visibilidade concorda com o Heavy | 99,6 % | 99,5 % |

## Por parte do corpo (PCK@0,2)

| Parte | Pontos visíveis por quadro | Lite | Full |
|---|---:|---:|---:|
| Rosto | 10,6 de 11 | 100,0 % | 100,0 % |
| Tronco | 4,0 de 4 | 99,9 % | 100,0 % |
| Braços | 2,6 de 4 | 99,5 % | 99,8 % |
| Mãos | 3,1 de 6 | 98,9 % | 99,2 % |
| Pernas | 0,0 de 4 | – | – |
| Pés | 0,0 de 4 | – | – |

"–" = parte do corpo fora do quadro (menos de 1 s de dados): não avaliada.

## Por pose (PCK@0,2)

| Métrica | Lite | Full |
|---|---:|---:|
| Braços para cima | 98,3 % | 98,8 % |
| Braços abertos | 100,0 % | 100,0 % |
| Braços cruzados | 100,0 % | 100,0 % |
| Agachamento | 100,0 % | 100,0 % |
| De perfil | 100,0 % | 100,0 % |
| Em uma perna | 100,0 % | 100,0 % |
| Andando de lado | 100,0 % | 100,0 % |
| Movimento livre | 99,6 % | 100,0 % |

## Como ler

- **PCK@α**: % de landmarks a menos de α × comprimento do tronco do ponto dado pelo Heavy. Com o tronco medindo ~200 px, α = 0,2 equivale a ~40 px e α = 0,1 a ~20 px.
- Só entram landmarks que o Heavy considera visíveis, em quadros com ombros e quadris visíveis.
- **Limitação**: o Heavy é a referência, não um gabarito anotado à mão. Os três modelos têm a mesma arquitetura e treino, então erros que os três cometem juntos não aparecem aqui; o número mede **concordância com o modelo mais preciso**.
