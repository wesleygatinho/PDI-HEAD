"""Metricas de avaliacao: FPS, latencia, taxa de deteccao e estabilidade."""
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import numpy as np

LIMIAR_VISIBILIDADE = 0.5

COLUNAS = [
    "quadro", "t_s", "modelo", "suavizacao", "fps_inst", "captura_ms", "latencia_ms",
    "inferencia_ms", "detectado", "n_visiveis", "vis_media",
    "jitter_bruto_px", "jitter_exibido_px", "fase",
]


def _deslocamento_medio(ant, atual, vis_ant, vis_atual) -> float:
    """Deslocamento medio (px) entre dois quadros, so nos landmarks visiveis nos dois."""
    if ant is None or atual is None:
        return float("nan")
    ambos = (vis_ant >= LIMIAR_VISIBILIDADE) & (vis_atual >= LIMIAR_VISIBILIDADE)
    if not ambos.any():
        return float("nan")
    return float(np.linalg.norm(atual[ambos] - ant[ambos], axis=1).mean())


class Metricas:
    """Acumula as medidas quadro a quadro.

    - FPS: quadros apresentados por segundo.
    - Latencia: do instante em que o quadro chega da camera ate ele ser
      mostrado na janela (inferencia + desenho + exibicao).
    - Captura: tempo que o programa fica parado em cap.read() esperando a
      camera entregar o proximo quadro. Se for bem maior que a latencia,
      quem limita o FPS e a camera, nao o processamento.
    - Taxa de deteccao: % de quadros em que a pose foi encontrada.
    - Estabilidade: jitter (tremor medio dos landmarks entre quadros, em px)
      e numero de vezes que a deteccao "piscou" (achou -> perdeu ou vice-versa).
    """

    def __init__(self, janela: int = 30) -> None:
        self._t_fim = deque(maxlen=janela)
        self._lat = deque(maxlen=janela)
        self._inf = deque(maxlen=janela)
        self._cap = deque(maxlen=janela)
        self._jit = deque(maxlen=janela)
        self.quadros = 0
        self.detectados = 0
        self.trocas = 0
        self._t_primeiro = None
        self._t_ultimo = None
        self._det_ant = None
        self._brutos_ant = self._exib_ant = self._vis_ant = None
        self._todas_lat: list[float] = []
        self._todas_inf: list[float] = []
        self._todas_cap: list[float] = []
        self._todos_jit_bruto: list[float] = []
        self._todos_jit_exib: list[float] = []
        self._todas_vis: list[float] = []

    def atualizar(self, t_captura, t_fim, inferencia_ms, brutos_px, exibidos_px, vis,
                  modelo="", suavizacao=False, captura_ms=float("nan"), fase="") -> dict:
        """Registra um quadro ja apresentado e devolve a linha para o CSV."""
        detectado = brutos_px is not None
        latencia = (t_fim - t_captura) * 1000.0
        tem_anterior = self._t_ultimo is not None and t_fim > self._t_ultimo
        fps_inst = 1.0 / (t_fim - self._t_ultimo) if tem_anterior else float("nan")

        if detectado:
            jit_bruto = _deslocamento_medio(self._brutos_ant, brutos_px, self._vis_ant, vis)
            jit_exib = _deslocamento_medio(self._exib_ant, exibidos_px, self._vis_ant, vis)
            n_vis = int((vis >= LIMIAR_VISIBILIDADE).sum())
            vis_media = float(vis.mean())
            self.detectados += 1
            self._todas_vis.append(vis_media)
            self._brutos_ant, self._exib_ant, self._vis_ant = brutos_px.copy(), exibidos_px.copy(), vis.copy()
        else:
            jit_bruto = jit_exib = vis_media = float("nan")
            n_vis = 0
            self._brutos_ant = self._exib_ant = self._vis_ant = None

        if self._det_ant is not None and detectado != self._det_ant:
            self.trocas += 1
        self._det_ant = detectado

        if self._t_primeiro is None:
            self._t_primeiro = t_fim
        self._t_ultimo = t_fim
        self._t_fim.append(t_fim)
        self._lat.append(latencia)
        self._inf.append(inferencia_ms)
        self._todas_lat.append(latencia)
        self._todas_inf.append(inferencia_ms)
        self._cap.append(captura_ms)
        self._todas_cap.append(captura_ms)
        if not np.isnan(jit_exib):
            self._jit.append(jit_exib)
            self._todos_jit_bruto.append(jit_bruto)
            self._todos_jit_exib.append(jit_exib)

        linha = {
            "quadro": self.quadros,
            "t_s": round(t_fim - self._t_primeiro, 4),
            "modelo": modelo,
            "suavizacao": int(suavizacao),
            "fps_inst": round(fps_inst, 2),
            "captura_ms": round(captura_ms, 2),
            "latencia_ms": round(latencia, 2),
            "inferencia_ms": round(inferencia_ms, 2),
            "detectado": int(detectado),
            "n_visiveis": n_vis,
            "vis_media": round(vis_media, 4),
            "jitter_bruto_px": round(jit_bruto, 3),
            "jitter_exibido_px": round(jit_exib, 3),
            "fase": fase,       # "parado"/"movimento" no roteiro guiado; vazio na gravacao livre
        }
        self.quadros += 1
        return linha

    # --- valores da janela movel (usados no painel da tela) ---
    @property
    def fps(self) -> float:
        if len(self._t_fim) < 2:
            return 0.0
        return (len(self._t_fim) - 1) / (self._t_fim[-1] - self._t_fim[0])

    @property
    def latencia_ms(self) -> float:
        return float(np.mean(self._lat)) if self._lat else 0.0

    @property
    def inferencia_ms(self) -> float:
        return float(np.mean(self._inf)) if self._inf else 0.0

    @property
    def captura_ms(self) -> float:
        return float(np.mean(self._cap)) if self._cap else 0.0

    @property
    def jitter_px(self) -> float:
        return float(np.mean(self._jit)) if self._jit else 0.0

    @property
    def taxa_deteccao(self) -> float:
        return 100.0 * self.detectados / self.quadros if self.quadros else 0.0

    # --- resumo da sessao inteira ---
    def resumo(self) -> dict:
        if self.quadros < 2:
            return {"quadros": self.quadros}
        duracao = self._t_ultimo - self._t_primeiro
        media = lambda v: float(np.mean(v)) if v else float("nan")
        return {
            "quadros": self.quadros,
            "duracao_s": duracao,
            "fps_medio": (self.quadros - 1) / duracao if duracao > 0 else float("nan"),
            "latencia_media_ms": media(self._todas_lat),
            "latencia_p95_ms": float(np.percentile(self._todas_lat, 95)),
            "inferencia_media_ms": media(self._todas_inf),
            "captura_media_ms": media(self._todas_cap),
            "taxa_deteccao_pct": self.taxa_deteccao,
            "trocas_deteccao": self.trocas,
            "visibilidade_media": media(self._todas_vis),
            "jitter_bruto_px": media(self._todos_jit_bruto),
            "jitter_exibido_px": media(self._todos_jit_exib),
        }

    def imprimir_resumo(self, titulo: str) -> None:
        r = self.resumo()
        print(f"\n=== {titulo} ===")
        if r["quadros"] < 2:
            print("  (quadros insuficientes)")
            return
        print(f"  Quadros processados : {r['quadros']} em {r['duracao_s']:.1f} s")
        print(f"  FPS medio           : {r['fps_medio']:.1f}")
        print(f"  Latencia            : media {r['latencia_media_ms']:.1f} ms | p95 {r['latencia_p95_ms']:.1f} ms")
        print(f"  Inferencia (modelo) : media {r['inferencia_media_ms']:.1f} ms")
        print(f"  Espera da camera    : media {r['captura_media_ms']:.1f} ms por quadro")
        print(f"  Taxa de deteccao    : {r['taxa_deteccao_pct']:.1f} %  ({r['trocas_deteccao']} trocas de estado)")
        print(f"  Visibilidade media  : {r['visibilidade_media']:.2f}")
        print(f"  Jitter (px/quadro)  : bruto {r['jitter_bruto_px']:.2f} | exibido {r['jitter_exibido_px']:.2f}")


class RegistroCSV:
    """Grava uma linha por quadro para analise posterior (graficos, tabelas)."""

    def __init__(self, caminho: Path) -> None:
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._arquivo = open(self.caminho, "w", newline="", encoding="utf-8")
        self._escritor = csv.DictWriter(self._arquivo, fieldnames=COLUNAS)
        self._escritor.writeheader()

    def escrever(self, linha: dict) -> None:
        self._escritor.writerow(linha)

    def fechar(self) -> None:
        self._arquivo.close()
