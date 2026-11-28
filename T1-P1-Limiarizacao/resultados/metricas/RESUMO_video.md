# Experimento 2 - Resultados sobre o video de 20 s

Video sintetico: 640x480, 30 fps, 20 s (600 quadros), com gabarito analitico.

## 1. Desempenho global (media sobre os 600 quadros)

| Metodo | IoU medio | Desvio do IoU | IoU minimo | F1 medio | Instabilidade (%) | Tempo (ms) | Desvio do limiar |
|---|---|---|---|---|---|---|---|
| Limiar fixo T=127 | 0.5164 | 0.3001 | 0.1196 | 0.6243 | 1.6675 | 0.3303 | 0.0000 |
| Otsu global (por quadro) | 0.7151 | 0.3412 | 0.1228 | 0.7725 | 1.4813 | 0.8596 | 27.3516 |
| Multi-Otsu (so a classe escura, t1) | 0.6406 | 0.0855 | 0.5760 | 0.7779 | 0.6586 | 1.1983 | 31.6929 |
| Multi-Otsu (escura + media, t2) | 0.6738 | 0.3781 | 0.1224 | 0.7305 | 5.4364 | 1.1983 | 34.9486 |
| Adaptativa gaussiana (b=35, C=10) | 0.5128 | 0.1230 | 0.2092 | 0.6681 | 6.0037 | 4.5890 | 49.4023 |
| Adaptativa AJUSTADA (b=151, C=10) | 0.8630 | 0.1952 | 0.3349 | 0.9117 | 4.3930 | 22.9442 | 49.4665 |
| Sauvola (local, media + desvio) | 0.7318 | 0.0876 | 0.3890 | 0.8418 | 1.5848 | 3.6603 | nan |

## 2. Desempenho por fase

Cada fase dura 5 s e estressa um aspecto diferente.

### Fase 1 - Iluminacao uniforme

| Metodo | IoU medio | F1 medio | Instabilidade (%) | Limiar medio |
|---|---|---|---|---|
| Limiar fixo T=127 | 0.6565 | 0.7927 | 1.0076 | 127.0000 |
| Otsu global (por quadro) | 0.9443 | 0.9714 | 0.2765 | 146.6733 |
| Multi-Otsu (so a classe escura, t1) | 0.5918 | 0.7435 | 0.2297 | 82.0000 |
| Multi-Otsu (escura + media, t2) | 0.9661 | 0.9827 | 0.2864 | 176.0000 |
| Adaptativa gaussiana (b=35, C=10) | 0.6210 | 0.7662 | 0.6284 | 195.6635 |
| Adaptativa AJUSTADA (b=151, C=10) | 0.9712 | 0.9854 | 0.2907 | 195.8714 |
| Sauvola (local, media + desvio) | 0.7873 | 0.8810 | 0.5145 | nan |

### Fase 2 - Holofote movel

| Metodo | IoU medio | F1 medio | Instabilidade (%) | Limiar medio |
|---|---|---|---|---|
| Limiar fixo T=127 | 0.1398 | 0.2451 | 1.2683 | 127.0000 |
| Otsu global (por quadro) | 0.1393 | 0.2444 | 1.2871 | 124.7533 |
| Multi-Otsu (so a classe escura, t1) | 0.7312 | 0.8398 | 0.2878 | 52.1667 |
| Multi-Otsu (escura + media, t2) | 0.1384 | 0.2430 | 1.2345 | 131.0933 |
| Adaptativa gaussiana (b=35, C=10) | 0.5284 | 0.6911 | 0.7028 | 85.3114 |
| Adaptativa AJUSTADA (b=151, C=10) | 0.9560 | 0.9775 | 0.2971 | 85.3139 |
| Sauvola (local, media + desvio) | 0.7567 | 0.8614 | 0.7845 | nan |

### Fase 3 - Escurecimento global

| Metodo | IoU medio | F1 medio | Instabilidade (%) | Limiar medio |
|---|---|---|---|---|
| Limiar fixo T=127 | 0.6357 | 0.6904 | 1.8559 | 127.0000 |
| Otsu global (por quadro) | 0.9425 | 0.9704 | 0.7796 | 96.1333 |
| Multi-Otsu (so a classe escura, t1) | 0.6000 | 0.7499 | 0.3016 | 53.9200 |
| Multi-Otsu (escura + media, t2) | 0.9650 | 0.9822 | 0.8092 | 116.0933 |
| Adaptativa gaussiana (b=35, C=10) | 0.5611 | 0.7180 | 0.7020 | 126.1890 |
| Adaptativa AJUSTADA (b=151, C=10) | 0.9496 | 0.9741 | 0.3804 | 126.3254 |
| Sauvola (local, media + desvio) | 0.7558 | 0.8609 | 0.6937 | nan |

### Fase 4 - Ruido + baixo contraste

| Metodo | IoU medio | F1 medio | Instabilidade (%) | Limiar medio |
|---|---|---|---|---|
| Limiar fixo T=127 | 0.6337 | 0.7691 | 2.5338 | 127.0000 |
| Otsu global (por quadro) | 0.8343 | 0.9040 | 3.5739 | 154.0000 |
| Multi-Otsu (so a classe escura, t1) | 0.6393 | 0.7783 | 1.8123 | 119.9333 |
| Multi-Otsu (escura + media, t2) | 0.6259 | 0.7141 | 19.3811 | 185.8267 |
| Adaptativa gaussiana (b=35, C=10) | 0.3408 | 0.4971 | 21.9460 | 185.7244 |
| Adaptativa AJUSTADA (b=151, C=10) | 0.5753 | 0.7099 | 16.5765 | 185.8531 |
| Sauvola (local, media + desvio) | 0.6275 | 0.7637 | 4.3392 | nan |

