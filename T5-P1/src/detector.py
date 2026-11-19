"""Deteccao de pose com o MediaPipe Pose Landmarker (API Tasks).

Encapsula o modelo pre-treinado (Lite / Full / Heavy) e devolve os 33
landmarks de um quadro como uma matriz NumPy, junto com o tempo gasto.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from time import perf_counter

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODELOS = ("lite", "full", "heavy")
N_LANDMARKS = 33

PASTA_MODELOS = Path(__file__).resolve().parent.parent / "modelos"
_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{0}/float16/latest/pose_landmarker_{0}.task"
)


def caminho_modelo(nome: str) -> Path:
    """Devolve o arquivo .task do modelo, baixando-o na primeira vez."""
    if nome not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {nome!r}. Use um de {MODELOS}.")
    destino = PASTA_MODELOS / f"pose_landmarker_{nome}.task"
    if not destino.exists():
        PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
        print(f"[modelo] Baixando pose_landmarker_{nome}.task (so na primeira vez)...")
        parcial = destino.with_suffix(".parcial")
        urllib.request.urlretrieve(_URL.format(nome), parcial)
        parcial.replace(destino)
        print(f"[modelo] Salvo em {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


class DetectorPose:
    """Pose Landmarker em modo VIDEO (sincrono: um quadro entra, um resultado sai).

    O modo VIDEO usa o rastreamento entre quadros do MediaPipe e, por ser
    sincrono, permite medir exatamente o tempo de inferencia de cada quadro.
    """

    def __init__(
        self,
        modelo: str = "full",
        conf_deteccao: float = 0.5,
        conf_presenca: float = 0.5,
        conf_rastreio: float = 0.5,
    ) -> None:
        self.modelo = modelo
        # O modelo e passado como bytes (e nao como caminho) para funcionar
        # mesmo quando a pasta do projeto tem acentos ou espacos no nome.
        dados = caminho_modelo(modelo).read_bytes()
        opcoes = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=dados),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=conf_deteccao,
            min_pose_presence_confidence=conf_presenca,
            min_tracking_confidence=conf_rastreio,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opcoes)
        self._ultimo_ts = -1

    def detectar(self, quadro_bgr: np.ndarray, ts_ms: float) -> tuple[np.ndarray | None, float]:
        """Processa um quadro BGR.

        Devolve (landmarks, tempo_ms). `landmarks` e uma matriz 33x4 com
        [x, y, z, visibilidade], x e y normalizados em [0, 1] em relacao a
        largura e altura da imagem; ou None se nenhuma pessoa foi encontrada.
        """
        # O MediaPipe exige carimbos de tempo estritamente crescentes.
        ts = max(int(ts_ms), self._ultimo_ts + 1)
        self._ultimo_ts = ts

        t0 = perf_counter()
        rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = self._landmarker.detect_for_video(imagem, ts)
        tempo_ms = (perf_counter() - t0) * 1000.0

        if not resultado.pose_landmarks:
            return None, tempo_ms
        pontos = np.array(
            [[p.x, p.y, p.z, p.visibility or 0.0] for p in resultado.pose_landmarks[0]],
            dtype=np.float32,
        )
        return pontos, tempo_ms

    def fechar(self) -> None:
        self._landmarker.close()
