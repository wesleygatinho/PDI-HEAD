"""Abertura e configuracao da webcam com OpenCV."""
from __future__ import annotations

import sys

import cv2

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2, "qualquer": cv2.CAP_ANY}
# "auto" tenta nesta ordem (DirectShow/Media Foundation so existem no Windows)
ORDEM_AUTO = ("dshow", "msmf", "qualquer") if sys.platform == "win32" else ("v4l2", "qualquer")


def _abrir(indice, backend, fourcc, largura, altura):
    cap = cv2.VideoCapture(indice, BACKENDS[backend])
    if not cap.isOpened():
        return None
    # Fila curta = menos atraso. No V4L2 (Linux) o minimo e 2: com 1 buffer o
    # driver nao tem onde gravar o proximo quadro enquanto o programa le o atual
    # e descarta um a cada dois (medido: 15 fps com 1 buffer, 30 fps com 2).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1 if sys.platform == "win32" else 2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if fourcc.lower() != "nenhum":
        # O formato tem de ser o ULTIMO ajuste: no DirectShow, mudar o tamanho
        # depois dele volta a camera para YUY2 (sem compressao), que em HD nao
        # cabe na USB 2.0 a 30 fps (medido: 13 fps em YUY2 contra 30 em MJPG).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))
    if not cap.read()[0]:                  # abriu mas nao entrega imagem
        cap.release()
        return None
    return cap


def abrir_camera(indice, backend="auto", fourcc="MJPG", largura=1280, altura=720):
    """Abre e configura a webcam. Devolve None se nao abrir ou nao entregar imagem.

    backend="auto" tenta, no Windows, DirectShow (abre mais rapido), depois
    Media Foundation; no Linux, V4L2; e por fim o que o OpenCV escolher.
    """
    for nome in (ORDEM_AUTO if backend == "auto" else (backend,)):
        cap = _abrir(indice, nome, fourcc, largura, altura)
        if cap is not None:
            return cap
    return None


def descrever_camera(cap) -> str:
    """Configuracao que a camera realmente aceitou (pode diferir da pedida)."""
    codigo = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((codigo >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")
    if not fourcc.isprintable() or not fourcc:
        fourcc = "?"
    return (f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"{fourcc} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps nominais, backend {cap.getBackendName()}")
