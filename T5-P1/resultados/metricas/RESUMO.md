# Resumo das sessões gravadas

1 gravações analisadas.

| Métrica | Lite | Full | Heavy |
|---|---:|---:|---:|
| Quadros | 601 | 601 | 441 |
| Duração (s) | 20,0 | 20,0 | 20,0 |
| FPS | 30,0 | 30,0 | 22,0 |
| Latência média (ms) | 14,1 | 17,5 | 45,1 |
| Latência p95 (ms) | 16,5 | 18,6 | 46,3 |
| Inferência (ms) | 10,4 | 13,7 | 41,2 |
| Espera da câmera (ms) | 19,1 | 15,6 | 0,2 |
| Detecção (%) | 100,0 | 100,0 | 100,0 |
| Trocas de detecção | 0 | 0 | 0 |
| Landmarks visíveis | 20,8 | 24,1 | 18,5 |
| Tremor parado, bruto (px) | 4,35 | 3,26 | 2,82 |
| Tremor parado, suavizado (px) | 3,53 | 2,13 | 2,48 |

- **FPS**: quadros apresentados por segundo.
- **Latência**: do quadro chegar da câmera até estar visível na janela.
- **Espera da câmera**: tempo parado em `cap.read()` aguardando o próximo quadro.
- **Trocas de detecção**: vezes em que a pose foi perdida ou reencontrada.
- **Tremor parado**: deslocamento médio dos landmarks entre quadros, só com a pessoa parada. Os trechos parados são os marcados pelo roteiro guiado (`--roteiro`).
