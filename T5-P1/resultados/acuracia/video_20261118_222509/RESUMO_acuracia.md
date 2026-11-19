# Taxa de acerto em relação ao modelo Heavy

Vídeo: `video_20261118_222509.mp4` · 1199 quadros · 1197 quadros avaliados (ombros e quadris visíveis no Heavy) · 27041 landmarks avaliados.

| Métrica | Lite | Full |
|---|---:|---:|
| **Taxa de acerto (PCK@0,2)** | **84,3 %** | **92,2 %** |
| PCK@0,1 (rigoroso) | 74,7 % | 86,1 % |
| Quadros corretos (≥ 90% dos pontos) | 74,4 % | 90,6 % |
| Erro médio (px) | 70,5 | 22,5 |
| Erro mediano (px) | 13,8 | 8,7 |
| Erro médio (% do tronco) | 26,3 | 9,8 |
| Detecção nos quadros avaliados | 99,2 % | 99,3 % |
| Visibilidade concorda com o Heavy | 97,2 % | 97,1 % |

## Por parte do corpo (PCK@0,2)

| Parte | Pontos visíveis por quadro | Lite | Full |
|---|---:|---:|---:|
| Rosto | 9,1 de 11 | 74,0 % | 93,1 % |
| Tronco | 4,0 de 4 | 86,4 % | 93,7 % |
| Braços | 2,4 de 4 | 95,2 % | 91,6 % |
| Mãos | 2,6 de 6 | 92,8 % | 88,7 % |
| Pernas | 2,5 de 4 | 92,3 % | 92,6 % |
| Pés | 2,0 de 4 | 93,7 % | 90,4 % |

"–" = parte do corpo fora do quadro (menos de 1 s de dados): não avaliada.

## Por pose (PCK@0,2)

| Métrica | Lite | Full |
|---|---:|---:|
| Braços para cima | 58,1 % | 36,8 % |
| Braços abertos | 100,0 % | 100,0 % |
| Braços cruzados | 99,8 % | 100,0 % |
| Agachamento | 99,2 % | 99,7 % |
| De perfil | 56,2 % | 96,8 % |
| Em uma perna | 35,2 % | 100,0 % |
| Andando de lado | 89,4 % | 97,1 % |
| Movimento livre | 99,7 % | 99,6 % |

## Como ler

- **PCK@α**: % de landmarks a menos de α × comprimento do tronco do ponto dado pelo Heavy. Com o tronco medindo ~200 px, α = 0,2 equivale a ~40 px e α = 0,1 a ~20 px.
- Só entram landmarks que o Heavy considera visíveis, em quadros com ombros e quadris visíveis.
- **Limitação**: o Heavy é a referência, não um gabarito anotado à mão. Os três modelos têm a mesma arquitetura e treino, então erros que os três cometem juntos não aparecem aqui; o número mede **concordância com o modelo mais preciso**.
