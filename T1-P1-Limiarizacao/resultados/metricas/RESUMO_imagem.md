# Experimento 1 - Resultados sobre imagens estaticas

## 1. Validacao das implementacoes proprias

Os limiares calculados pelo codigo escrito do zero sao comparados aos das bibliotecas de referencia.

| Imagem | Otsu propria | Otsu skimage | Otsu OpenCV | Multi-Otsu propria | Multi-Otsu skimage | Concord. adapt. (%) | eta |
|---|---|---|---|---|---|---|---|
| 10_sudoku | 96 | 96 | 96 | [75, 117] | [75, 117] | 99.7559 | 0.6476 |
| 11_ressonancia | 39 | 39 | 39 | [35, 109] | [35, 109] | 99.7178 | 0.8184 |
| 01_camera | 102 | 102 | 102 | [87, 176] | [87, 176] | 99.4797 | 0.8572 |
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
| Otsu (implementacao propria) | 0.4720 | 2118.7563 |
| Otsu (skimage) | 0.6009 | 1664.0873 |
| Otsu (OpenCV) | 0.2403 | 4160.8588 |
| Multi-Otsu 3 classes (propria) | 1.0472 | 954.8955 |
| Multi-Otsu 3 classes (skimage) | 0.9901 | 1009.9735 |
| Multi-Otsu 4 classes (skimage) | 29.1558 | 34.2985 |
| Adaptativa media (propria, integral) | 1.5062 | 663.9313 |
| Adaptativa gaussiana (propria) | 4.5493 | 219.8150 |
| Adaptativa (OpenCV) | 1.0783 | 927.3943 |
| Sauvola (propria) | 2.9664 | 337.1101 |

