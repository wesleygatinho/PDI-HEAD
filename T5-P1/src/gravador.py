"""Grava o video BRUTO da camera (sem esqueleto, sem espelhar) para reprocessar depois.

Junto do video sai um CSV com a fase (pose pedida pelo roteiro) de cada quadro:
    resultados/videos/video_<data>.mp4
    resultados/videos/video_<data>_fases.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2


def caminho_fases(video: Path) -> Path:
    return video.with_name(video.stem + "_fases.csv")


class GravadorVideo:
    def __init__(self, caminho: Path, fps: float = 30.0) -> None:
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.quadros = 0
        self._writer = None
        self._arq_fases = open(caminho_fases(self.caminho), "w", newline="", encoding="utf-8")
        self._fases = csv.writer(self._arq_fases)
        self._fases.writerow(["quadro", "t_s", "fase"])

    def escrever(self, quadro, fase: str, t_s: float) -> None:
        if self._writer is None:        # abre so no primeiro quadro, quando o tamanho e conhecido
            h, w = quadro.shape[:2]
            self._writer = cv2.VideoWriter(str(self.caminho), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"Nao foi possivel criar o video {self.caminho}")
        self._writer.write(quadro)
        self._fases.writerow([self.quadros, round(t_s, 4), fase])
        self.quadros += 1

    def fechar(self) -> None:
        if self._writer is not None:
            self._writer.release()
        self._arq_fases.close()
