"""Diagnostico da webcam: quantos quadros por segundo ela REALMENTE entrega.

Testa combinacoes de backend, formato e resolucao, sem MediaPipe e sem
janela, e mostra qual configuracao usar no espelho.py. Nenhuma imagem e salva.

    python testar_camera.py
    python testar_camera.py --camera 1
"""
from __future__ import annotations

import argparse
import sys
from time import perf_counter

from src.camera import abrir_camera, descrever_camera

# backend, fourcc, largura, altura (a primeira e a configuracao padrao do espelho.py)
if sys.platform == "win32":
    COMBINACOES = [
        ("dshow", "MJPG", 1280, 720),
        ("dshow", "nenhum", 1280, 720),
        ("dshow", "MJPG", 640, 480),
        ("dshow", "nenhum", 640, 480),
        ("msmf", "MJPG", 1280, 720),
        ("msmf", "nenhum", 1280, 720),
        ("msmf", "nenhum", 640, 480),
    ]
else:
    COMBINACOES = [
        ("v4l2", "MJPG", 1280, 720),
        ("v4l2", "nenhum", 1280, 720),
        ("v4l2", "MJPG", 640, 480),
        ("v4l2", "nenhum", 640, 480),
    ]
AQUECIMENTO = 15      # quadros descartados (a camera ajusta exposicao no inicio)
SEGUNDOS = 3.0


def medir(camera, backend, fourcc, largura, altura):
    cap = abrir_camera(camera, backend, fourcc, largura, altura)
    if cap is None:
        return None
    try:
        for _ in range(AQUECIMENTO):
            if not cap.read()[0]:
                return None
        n, brilho, t0 = 0, 0.0, perf_counter()
        while perf_counter() - t0 < SEGUNDOS:
            ok, quadro = cap.read()
            if not ok:
                break
            n += 1
            brilho += float(quadro[::8, ::8].mean())
        dt = perf_counter() - t0
        return {"fps": n / dt, "brilho": brilho / max(n, 1), "config": descrever_camera(cap)}
    finally:
        cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description="Mede o FPS real da webcam em varias configuracoes")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    print(f"Testando a webcam {args.camera} (cerca de 1 minuto)...\n")
    print(f"{'backend':8} {'pedido':20} {'aceito pela camera':48} {'FPS real':>8} {'brilho':>7}")
    resultados = []
    for backend, fourcc, larg, alt in COMBINACOES:
        r = medir(args.camera, backend, fourcc, larg, alt)
        pedido = f"{larg}x{alt} {fourcc}"
        if r is None:
            print(f"{backend:8} {pedido:20} {'(nao abriu)':48}")
            continue
        print(f"{backend:8} {pedido:20} {r['config']:48} {r['fps']:8.1f} {r['brilho']:7.0f}")
        resultados.append((r["fps"], backend, fourcc, larg, alt, r["brilho"]))

    if not resultados:
        raise SystemExit("\nNenhuma configuracao funcionou. Feche qualquer programa que esteja usando a camera "
                         "(inclusive a janela do espelho) e rode de novo.")

    melhor_hd = max((r for r in resultados if r[3] >= 1280), default=None)
    melhor = max(resultados)
    escolha = melhor_hd if melhor_hd and melhor_hd[0] >= 0.9 * melhor[0] else melhor
    fps, backend, fourcc, larg, alt, _ = escolha
    sufixo = f" --camera {args.camera}" if args.camera else ""
    padrao = next((r for r in resultados if r[1:5] == COMBINACOES[0]), None)
    if padrao and padrao[0] >= 0.95 * fps:
        print(f"\nA configuracao padrao ja entrega {padrao[0]:.1f} fps. Basta:")
        print(f"  python espelho.py{sufixo}")
    else:
        print(f"\nMelhor configuracao: {fps:.1f} fps. Use:")
        print(f"  python espelho.py --backend {backend} --fourcc {fourcc} --largura {larg} --altura {alt}{sufixo}")

    brilho = sum(r[5] for r in resultados) / len(resultados)
    if melhor[0] < 20:
        print("\nNenhuma configuracao chegou a 20 fps.")
        if brilho < 90:
            print(f"A imagem esta escura (brilho medio {brilho:.0f} de 255): com pouca luz a camera aumenta o")
            print("tempo de exposicao e entrega menos quadros. Acenda mais luz de frente para voce e teste de novo.")
        else:
            print("A luz parece suficiente, entao o limite deve ser da propria camera ou da porta USB.")


if __name__ == "__main__":
    main()
