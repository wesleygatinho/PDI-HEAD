# Experimento 1 - Resultados sobre imagens estaticas

## 1. Validacao das implementacoes proprias

Os limiares calculados pelo codigo escrito do zero sao comparados aos das bibliotecas de referencia.

| Imagem | Otsu propria | Otsu skimage | Otsu OpenCV | Multi-Otsu propria | Multi-Otsu skimage | Concord. adapt. (%) | eta |
|---|---|---|---|---|---|---|---|
| 01_camera | 102 | 102 | 102 | [87, 176] | [87, 176] | 99.4797 | 0.8572 |
| 02_moedas | 107 | 107 | 107 | [77, 139] | [77, 139] | 99.2652 | 0.7564 |
| 03_pagina | 157 | 157 | 157 | [114, 186] | [114, 186] | 99.8759 | 0.7189 |
| 04_sintetica | 125 | 125 | 125 | [74, 148] | [74, 148] | 99.9046 | 0.5956 |

## 2. Metricas contra o gabarito (imagem 04_sintetica)

Imagem com iluminacao diagonal nao uniforme e ruido gaussiano (sigma = 3).

| Metodo | Acuracia | Precisao | Revocacao | F1/Dice | IoU | MCC | Ruido isolado |
|---|---|---|---|---|---|---|---|
| Limiar fixo T=127 | 0.8082 | 0.3878 | 0.9496 | 0.5507 | 0.3799 | 0.5287 | 0.0304 |
| Otsu global (T=125) | 0.8247 | 0.4100 | 0.9477 | 0.5723 | 0.4009 | 0.5502 | 0.0296 |
| Multi-Otsu, classe escura (T=74) | 0.9496 | 1.0000 | 0.5925 | 0.7441 | 0.5925 | 0.7485 | 0.0101 |
| Multi-Otsu, escura+media (T=148) | 0.6196 | 0.2418 | 0.9705 | 0.3871 | 0.2400 | 0.3560 | 0.0386 |
| Adaptativa media (b=35, C=10) | 0.9753 | 0.9976 | 0.8025 | 0.8895 | 0.8010 | 0.8824 | 0.0112 |
| Adaptativa gaussiana (b=35, C=10) | 0.9485 | 0.9964 | 0.5862 | 0.7381 | 0.5850 | 0.7426 | 0.0124 |
| Sauvola (b=35, k=0.2) | 0.9728 | 1.0000 | 0.7803 | 0.8766 | 0.7803 | 0.8699 | 0.0123 |

## 3. Custo computacional (imagem 640x480)

| Metodo | Tempo (ms) | Chamadas/s |
|---|---|---|
| Otsu (implementacao propria) | 0.4585 | 2180.8110 |
| Otsu (skimage) | 0.5113 | 1955.8372 |
| Otsu (OpenCV) | 0.1454 | 6879.9449 |
| Multi-Otsu 3 classes (propria) | 1.1388 | 878.0865 |
| Multi-Otsu 3 classes (skimage) | 0.9344 | 1070.2398 |
| Multi-Otsu 4 classes (skimage) | 29.1694 | 34.2825 |
| Adaptativa media (propria, integral) | 1.7046 | 586.6324 |
| Adaptativa gaussiana (propria) | 4.8077 | 208.0016 |
| Adaptativa (OpenCV) | 1.1253 | 888.6361 |
| Sauvola (propria) | 5.9453 | 168.2002 |

