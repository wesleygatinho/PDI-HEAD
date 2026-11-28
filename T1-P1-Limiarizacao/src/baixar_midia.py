# -*- coding: utf-8 -*-
"""
baixar_midia.py
===============
Baixa a midia REAL usada no trabalho e prepara o recorte de 20 s do video.

Toda a midia tem licenca livre e verificada; a procedencia fica registrada em
data/CREDITOS.md, gerado por este script.

    sudoku.png   imagem do proprio tutorial de limiarizacao do OpenCV.
                 Foto de jornal com iluminacao desigual - o caso classico da
                 limiarizacao adaptativa.
                 Fonte: github.com/opencv/opencv (samples/data), licenca Apache 2.0

    mri.jpg      ressonancia magnetica cerebral T1 axial. Tres populacoes de
                 intensidade (fundo, tecido, gordura/osso) - o caso classico do
                 Multi-Otsu.
                 Fonte: Wikimedia Commons, autor 511KeV, licenca CC BY-SA 4.0

    canyon_20s.mp4  por-do-sol no Grand Canyon (time-lapse). Recorte de 20 s de
                 um trecho SEM corte de cena, em que o brilho medio cai de 107
                 para 63 e o limiar de Otsu desce de 71 para 47.
                 Fonte: Wikimedia Commons / National Park Service, dominio publico

Uso:
    python src/baixar_midia.py            baixa o que faltar
    python src/baixar_midia.py --forcar   baixa tudo de novo
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import config
import utils

UA = "UFMA-trabalho-academico/1.0 (limiarizacao; contato via disciplina)"

# ---------------------------------------------------------------------------
# Catalogo da midia, com procedencia
# ---------------------------------------------------------------------------
FONTES = {
    "sudoku": dict(
        url="https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/sudoku.png",
        arquivo="sudoku.png",
        titulo="Sudoku de jornal fotografado",
        autor="Projeto OpenCV",
        licenca="Apache 2.0",
        pagina="https://github.com/opencv/opencv/blob/4.x/samples/data/sudoku.png",
        nota="Imagem usada no tutorial oficial de limiarizacao do OpenCV.",
    ),
    "mri": dict(
        url="https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/"
            "MRI_Brain_T1_Axial_%2812%29.jpg/960px-MRI_Brain_T1_Axial_%2812%29.jpg",
        arquivo="mri.jpg",
        titulo="MRI Brain T1 Axial (12)",
        autor="511KeV",
        licenca="CC BY-SA 4.0",
        pagina="https://commons.wikimedia.org/wiki/File:MRI_Brain_T1_Axial_(12).jpg",
        nota="Exige atribuicao. O credito aparece no slide de referencias.",
    ),
    "canyon": dict(
        url="https://upload.wikimedia.org/wikipedia/commons/6/63/"
            "Grand_Canyon_National_Park-_Colorado_River_Time-lapse_and_Sunset_"
            "%288616673351%29.webm",
        arquivo="canyon_completo.webm",
        titulo="Grand Canyon National Park - Colorado River Time-lapse and Sunset",
        autor="Grand Canyon National Park / National Park Service",
        licenca="Dominio publico (obra do governo dos EUA)",
        pagina="https://commons.wikimedia.org/wiki/File:Grand_Canyon_National_Park-"
               "_Colorado_River_Time-lapse_and_Sunset_(8616673351).webm",
        nota="Do arquivo completo (72 s) e extraido o recorte de 20 s.",
    ),
}

# Recorte escolhido por analise automatica: trecho sem corte de cena e com a
# maior variacao do limiar de Otsu (ver secao 'escolha do recorte' no README).
CANYON_QUADRO_INICIAL = 495          # t = 20,6 s no arquivo original
CANYON_DURACAO_S = 20
NOME_VIDEO_REAL = "canyon_20s.mp4"


def _baixar(url: str, destino: Path, forcar: bool) -> bool:
    if destino.exists() and not forcar:
        print(f"    {destino.name:24s} ja existe ({destino.stat().st_size/1e6:.1f} MB)")
        return True
    destino.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(destino, "wb") as f:
            f.write(r.read())
    except Exception as e:
        print(f"    !! falhou {destino.name}: {e}")
        return False
    print(f"    {destino.name:24s} baixado ({destino.stat().st_size/1e6:.1f} MB)")
    return True


def recortar_video(origem: Path, destino: Path, quadro_inicial: int,
                   duracao_s: int) -> None:
    """Extrai `duracao_s` segundos a partir de `quadro_inicial` e grava em mp4."""
    cap = cv2.VideoCapture(str(origem))
    if not cap.isOpened():
        raise FileNotFoundError(f"nao consegui abrir {origem}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    larg = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_quadros = int(round(duracao_s * fps))

    escritor = utils.abrir_escritor(destino, fps, (larg, alt), colorido=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, quadro_inicial)
    escritos = 0
    while escritos < n_quadros:
        ok, quadro = cap.read()
        if not ok:
            break
        escritor.write(quadro)
        escritos += 1
    cap.release()
    escritor.release()
    print(f"    {destino.name:24s} {escritos} quadros @ {fps:.0f} fps "
          f"= {escritos/fps:.1f} s, {larg}x{alt}")


def escrever_creditos() -> None:
    cam = config.DIR_DADOS / "CREDITOS.md"
    with open(cam, "w", encoding="utf-8") as f:
        f.write("# Créditos da mídia\n\n")
        f.write("Toda a mídia usada neste trabalho tem licença livre. "
                "As imagens foram baixadas por `src/baixar_midia.py`.\n\n")
        for chave, d in FONTES.items():
            f.write(f"## {d['titulo']}\n\n")
            f.write(f"- **Arquivo:** `{d['arquivo']}`\n")
            f.write(f"- **Autor:** {d['autor']}\n")
            f.write(f"- **Licença:** {d['licenca']}\n")
            f.write(f"- **Origem:** {d['pagina']}\n")
            f.write(f"- {d['nota']}\n\n")
        f.write("## Cena sintética\n\n")
        f.write("A cena sintética (`04_sintetica*.png` e `video_teste_20s.mp4`) "
                "foi gerada pelo código deste trabalho, em `src/gerar_midia.py`. "
                "É a única mídia com gabarito conhecido, e por isso é a que "
                "permite medir IoU.\n")
    print(f"    {cam.name:24s} escrito")


def main() -> None:
    ap = argparse.ArgumentParser(description="Baixa a midia real do trabalho")
    ap.add_argument("--forcar", action="store_true", help="rebaixar tudo")
    args = ap.parse_args()

    config.garantir_pastas()
    print("=" * 74)
    print("DOWNLOAD DA MIDIA REAL")
    print("=" * 74)

    print("\n[imagens]")
    ok_sudoku = _baixar(FONTES["sudoku"]["url"],
                        config.DIR_IMAGENS / "10_sudoku.png", args.forcar)
    ok_mri = _baixar(FONTES["mri"]["url"],
                     config.DIR_IMAGENS / "11_ressonancia.jpg", args.forcar)

    print("\n[video]")
    bruto = config.DIR_VIDEOS / FONTES["canyon"]["arquivo"]
    ok_video = _baixar(FONTES["canyon"]["url"], bruto, args.forcar)

    recorte = config.DIR_VIDEOS / NOME_VIDEO_REAL
    if ok_video and (args.forcar or not recorte.exists()):
        recortar_video(bruto, recorte, CANYON_QUADRO_INICIAL, CANYON_DURACAO_S)
    elif recorte.exists():
        print(f"    {recorte.name:24s} ja existe "
              f"({recorte.stat().st_size/1e6:.1f} MB)")

    print("\n[creditos]")
    escrever_creditos()

    if not (ok_sudoku and ok_mri and ok_video):
        print("\n!! Alguma fonte falhou. Verifique a conexao e rode de novo.")
        raise SystemExit(1)
    print("\nConcluido.\n")


if __name__ == "__main__":
    main()
