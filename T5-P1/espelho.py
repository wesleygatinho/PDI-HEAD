"""Simulador de Espelho Corporal - Tema 6 (Processamento de Imagens, UFMA).

WEBCAM -> OpenCV -> MediaPipe Pose -> 33 landmarks -> esqueleto em tempo real.

Uso basico:
    python espelho.py                    # webcam 0, modelo Full
    python espelho.py --modelo lite      # modelo mais leve
    python espelho.py --video arquivo.mp4 --sem-janela --registrar   # teste reprodutivel
    python espelho.py --roteiro          # grava os 3 modelos e gera a analise de desempenho
    python espelho.py --gravar-video     # grava um video com poses e mede a taxa de acerto
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.camera import BACKENDS, abrir_camera, descrever_camera
from src.desenho import (AMARELO, BRANCO, MODOS, VERDE, VERMELHO, Compositor,
                         desenhar_faixa, desenhar_painel, desenhar_rodape)
from src.gravador import GravadorVideo
from src.roteiro import Roteiro
from src.detector import MODELOS, N_LANDMARKS, DetectorPose
from src.metricas import LIMIAR_VISIBILIDADE, Metricas, RegistroCSV
from src.suavizacao import FiltroOneEuro

RAIZ = Path(__file__).resolve().parent
PASTA_RESULTADOS = RAIZ / "resultados"
JANELA = "Espelho Corporal"
RODAPE = "1-4 modo | L/F/H modelo | S suavizar | N numeros | E espelhar | M painel | R gravar CSV | P foto | Q sair"
TECLA_MODELO = {ord("l"): "lite", ord("f"): "full", ord("h"): "heavy"}
TECLAS_BLOQUEADAS_NO_ROTEIRO = {ord(c) for c in "lfhrse"}
# Tecla segurada gera repeticoes; so aceita de novo depois deste intervalo solta.
INTERVALO_TECLA_S = 0.7


def ler_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulador de Espelho Corporal (MediaPipe Pose + OpenCV)")
    p.add_argument("--camera", type=int, default=0, help="indice da webcam (padrao: 0)")
    p.add_argument("--video", type=Path, help="usa um arquivo de video no lugar da webcam")
    p.add_argument("--modelo", choices=MODELOS, default="full", help="tamanho do modelo (padrao: full)")
    p.add_argument("--largura", type=int, default=1280)
    p.add_argument("--altura", type=int, default=720)
    p.add_argument("--backend", choices=["auto", *BACKENDS], default="auto",
                   help="API de captura do Windows (padrao: auto = tenta dshow, depois msmf)")
    p.add_argument("--fourcc", default="MJPG", help="formato pedido a camera; 'nenhum' deixa o padrao dela (padrao: MJPG)")
    p.add_argument("--modo", type=int, choices=list(MODOS), default=1, help="1 sobreposto, 2 avatar, 3 lado a lado, 4 rastro")
    p.add_argument("--sem-suavizacao", action="store_true", help="desliga o filtro One Euro")
    p.add_argument("--sem-espelho", action="store_true", help="nao inverte a imagem horizontalmente")
    p.add_argument("--registrar", action="store_true", help="grava as metricas quadro a quadro em CSV desde o inicio")
    p.add_argument("--saida", type=Path, help="caminho do CSV (padrao: resultados/sessoes/sessao_<data>.csv)")
    p.add_argument("--duracao", type=float, help="encerra sozinho apos N segundos")
    p.add_argument("--sem-janela", action="store_true", help="nao abre janela (para medir desempenho em video)")
    p.add_argument("--roteiro", action="store_true",
                   help="gravacao guiada: percorre Lite, Full e Heavy sozinho e gera a analise no final")
    p.add_argument("--tempo-fase", type=float, default=10.0, help="segundos parado e em movimento por modelo no roteiro")
    p.add_argument("--tempo-preparo", type=float, default=5.0, help="segundos de preparo antes de cada modelo no roteiro")
    p.add_argument("--gravar-video", action="store_true",
                   help="grava o video bruto da camera seguindo um roteiro de poses e mede a taxa de acerto no final")
    p.add_argument("--tempo-pose", type=float, default=5.0, help="segundos de cada pose em --gravar-video")
    args = p.parse_args()
    if args.roteiro and args.gravar_video:
        p.error("use --roteiro ou --gravar-video, nao os dois juntos")
    return args


def abrir_fonte(args) -> tuple[cv2.VideoCapture, bool, float]:
    """Abre a webcam ou o arquivo. Devolve (captura, eh_arquivo, fps_da_fonte)."""
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise SystemExit(f"Nao foi possivel abrir o video: {args.video}")
        return cap, True, cap.get(cv2.CAP_PROP_FPS) or 30.0

    cap = abrir_camera(args.camera, args.backend, args.fourcc, args.largura, args.altura)
    if cap is None:
        raise SystemExit(f"Nao foi possivel abrir a webcam {args.camera}. Feche outros programas que estejam "
                         "usando a camera (inclusive outra janela do espelho) ou tente --camera 1.")
    print(f"[camera] {descrever_camera(cap)}")
    return cap, False, cap.get(cv2.CAP_PROP_FPS) or 30.0


def linhas_do_painel(m: Metricas, modelo, modo, suavizar, gravando: list[str], vis) -> list:
    if vis is None:
        pose = ("Pose: NAO DETECTADA", VERMELHO)
    else:
        n = int((vis >= LIMIAR_VISIBILIDADE).sum())
        pose = (f"Pose: detectada | {n}/{N_LANDMARKS} landmarks visiveis", VERDE)
    linhas = [
        (f"ESPELHO CORPORAL | modelo {modelo.upper()} | modo {modo}-{MODOS[modo]}", BRANCO),
        (f"FPS: {m.fps:5.1f} | espera da camera: {m.captura_ms:.0f} ms", AMARELO),
        (f"Latencia: {m.latencia_ms:5.1f} ms (inferencia {m.inferencia_ms:.1f} ms)", AMARELO),
        pose,
        (f"Deteccao: {m.taxa_deteccao:5.1f} % | Jitter: {m.jitter_px:.2f} px", BRANCO),
        (f"Suavizacao: {'LIGADA' if suavizar else 'desligada'}", BRANCO),
    ]
    if gravando:
        linhas.append(("GRAVANDO " + " + ".join(gravando), VERMELHO))
    return linhas


def salvar_foto(tela) -> None:
    pasta = PASTA_RESULTADOS / "capturas"
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / f"captura_{datetime.now():%Y%m%d_%H%M%S}.png"
    ok, dados = cv2.imencode(".png", tela)     # imencode + write_bytes aceita acentos no caminho
    if ok:
        destino.write_bytes(dados.tobytes())
        print(f"[foto] {destino}")


def main() -> None:
    args = ler_argumentos()
    cap, eh_arquivo, fps_fonte = abrir_fonte(args)

    roteiro = None
    if args.roteiro:
        roteiro = Roteiro.de_modelos(MODELOS, args.tempo_preparo, args.tempo_fase)
    elif args.gravar_video:
        roteiro = Roteiro.de_poses(args.modelo, args.tempo_pose, 2 * args.tempo_preparo)

    modelo = roteiro.etapa.modelo if roteiro else args.modelo
    modo = args.modo
    suavizar = True if roteiro else not args.sem_suavizacao
    espelhar = not args.sem_espelho
    numeros = False
    painel = True

    detector = DetectorPose(modelo)
    filtro = FiltroOneEuro()
    compositor = Compositor()
    metricas = Metricas()
    registro = None
    gravador = None
    if args.gravar_video:
        gravador = GravadorVideo(PASTA_RESULTADOS / "videos" / f"video_{datetime.now():%Y%m%d_%H%M%S}.mp4", fps_fonte)
        print(f"[video] Gravando em {gravador.caminho}")

    def novo_registro() -> RegistroCSV:
        prefixo = "roteiro" if args.roteiro else "sessao"
        caminho = args.saida or PASTA_RESULTADOS / "sessoes" / f"{prefixo}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        print(f"[csv] Gravando em {caminho}")
        return RegistroCSV(caminho)

    def trocar_modelo(novo: str) -> None:
        nonlocal detector, modelo, metricas
        metricas.imprimir_resumo(f"Resumo - modelo {modelo.upper()}")
        detector.fechar()
        modelo = novo
        detector = DetectorPose(modelo)
        metricas = Metricas()
        filtro.reiniciar()

    if args.registrar or args.roteiro:
        registro = novo_registro()
    if not args.sem_janela:
        cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
        print(RODAPE)
    if roteiro:
        print(f"[roteiro] Cerca de {roteiro.duracao_total:.0f} s. Fique de corpo inteiro no quadro e siga a "
              "faixa na tela. Q interrompe.")
        roteiro.iniciar_etapa(perf_counter())
    caminho_registro = registro.caminho if registro else None

    ultimo_toque: dict[int, float] = {}

    def tocou(tecla: int) -> bool:
        """True so no primeiro evento de um toque (ignora a repeticao de tecla segurada)."""
        agora = perf_counter()
        novo = agora - ultimo_toque.get(tecla, -1e9) > INTERVALO_TECLA_S
        ultimo_toque[tecla] = agora
        return novo

    t_inicio = perf_counter()
    n = 0
    try:
        while True:
            t_antes = perf_counter()
            ok, quadro = cap.read()             # bloqueia ate a camera entregar um quadro
            t_captura = perf_counter()          # o quadro acabou de chegar da camera
            if not ok:
                break
            t_s = n / fps_fonte if eh_arquivo else t_captura - t_inicio
            fase = roteiro.etapa.fase if roteiro else ""

            # 0) Video bruto para a taxa de acerto: quadro original, antes de espelhar.
            if gravador and fase != "preparo":
                gravador.escrever(quadro, fase, t_s)

            # 1) Deteccao no quadro ORIGINAL, para que "esquerda/direita" do
            #    MediaPipe continuem sendo os lados anatomicos da pessoa.
            pontos, inferencia_ms = detector.detectar(quadro, t_s * 1000.0)

            # 2) Efeito espelho: inverte a imagem e a coordenada x dos landmarks.
            altura, largura = quadro.shape[:2]
            if espelhar:
                quadro = cv2.flip(quadro, 1)
                if pontos is not None:
                    pontos[:, 0] = 1.0 - pontos[:, 0]

            # 3) Coordenadas normalizadas -> pixels, com suavizacao opcional.
            brutos_px = exibidos_px = vis = None
            if pontos is not None:
                vis = pontos[:, 3]
                tamanho = np.array([largura, altura])
                brutos_px = pontos[:, :2] * tamanho
                # O filtro trabalha nas coordenadas normalizadas: assim o ajuste
                # dele nao depende da resolucao da camera.
                exibidos_px = filtro(pontos[:, :2], t_s) * tamanho if suavizar else brutos_px
            if pontos is None or not suavizar:
                filtro.reiniciar()

            # 4) Desenho do esqueleto e do painel.
            tela = compositor.compor(modo, quadro, exibidos_px, vis, numeros)
            if painel:
                gravando = (["CSV"] if registro else []) + (["VIDEO"] if gravador and fase != "preparo" else [])
                desenhar_painel(tela, linhas_do_painel(metricas, modelo, modo, suavizar, gravando, vis))
                desenhar_rodape(tela, RODAPE)
            if roteiro:
                cor = {"preparo": BRANCO, "parado": AMARELO}.get(fase, VERDE)
                desenhar_faixa(tela, roteiro.texto(perf_counter()), cor)

            # 5) Apresentacao na tela.
            tecla = -1
            if not args.sem_janela:
                cv2.imshow(JANELA, tela)
                espera = 1
                if eh_arquivo:      # video gravado: respeita a velocidade original
                    espera = max(1, int(1000.0 / fps_fonte - (perf_counter() - t_captura) * 1000.0))
                tecla = cv2.waitKey(espera) & 0xFF
            t_fim = perf_counter()              # o quadro ja esta visivel

            linha = metricas.atualizar(t_captura, t_fim, inferencia_ms, brutos_px, exibidos_px, vis,
                                       modelo=modelo, suavizacao=suavizar,
                                       captura_ms=(t_captura - t_antes) * 1000.0, fase=fase)
            if registro and fase != "preparo":  # o preparo do roteiro nao entra nas medidas
                registro.escrever(linha)
            n += 1

            # 6) Teclado.
            if tecla in (ord("q"), ord("Q"), 27):
                break
            tecla_min = ord(chr(tecla).lower()) if 0 <= tecla < 128 else -1
            if roteiro and tecla_min in TECLAS_BLOQUEADAS_NO_ROTEIRO:
                tecla_min = -1                  # o roteiro controla modelo, gravacao e suavizacao
            if tecla_min in TECLA_MODELO and TECLA_MODELO[tecla_min] != modelo:
                trocar_modelo(TECLA_MODELO[tecla_min])
            elif ord("1") <= tecla <= ord("4"):
                modo = tecla - ord("0")
            elif tecla_min < 0 or not tocou(tecla_min):
                pass                            # nenhuma tecla, ou repeticao de tecla segurada
            elif tecla_min == ord("s"):
                suavizar = not suavizar
            elif tecla_min == ord("n"):
                numeros = not numeros
            elif tecla_min == ord("e"):
                espelhar = not espelhar
                filtro.reiniciar()
            elif tecla_min == ord("m"):
                painel = not painel
            elif tecla_min == ord("p"):
                salvar_foto(tela)
            elif tecla_min == ord("r"):
                if registro:
                    registro.fechar()
                    print(f"[csv] Gravacao encerrada: {registro.caminho}")
                    registro = None
                else:
                    registro = novo_registro()

            # 7) Roteiro guiado: avanca de etapa e troca o modelo quando for a hora.
            if roteiro:
                roteiro.atualizar(perf_counter())
                if roteiro.concluido:
                    break
                if roteiro.etapa.modelo != modelo:
                    trocar_modelo(roteiro.etapa.modelo)
                    roteiro.iniciar_etapa(perf_counter())   # carregar o modelo nao gasta o tempo da etapa

            if not args.sem_janela and cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break                           # janela fechada no X
            if args.duracao and t_fim - t_inicio >= args.duracao:
                break
    finally:
        cap.release()
        detector.fechar()
        if registro:
            registro.fechar()
        if gravador:
            gravador.fechar()
        cv2.destroyAllWindows()

    metricas.imprimir_resumo(f"Resumo - modelo {modelo.upper()}")

    if gravador:
        if gravador.quadros == 0:
            print("\n[video] Nenhum quadro gravado (interrompido no preparo).")
        elif roteiro.concluido:
            print(f"\n[video] Concluido: {gravador.quadros} quadros em {gravador.caminho}")
            print("[video] Medindo a taxa de acerto (processa o video com os 3 modelos, leva 1 a 2 minutos)...\n")
            import acuracia     # importado aqui para o matplotlib nao atrasar a abertura da camera
            acuracia.main(["--video", str(gravador.caminho)])
        else:
            print(f"\n[video] Interrompido: {gravador.quadros} quadros gravados em {gravador.caminho}.")
            print(f'        Para medir mesmo assim:  python acuracia.py --video "{gravador.caminho}"')
    elif roteiro and roteiro.concluido:
        print("\n[roteiro] Concluido. Gerando tabelas e graficos so desta gravacao...\n")
        import analise      # importado aqui para o matplotlib nao atrasar a abertura da camera
        analise.main(["--arquivos", str(caminho_registro)])
    elif roteiro:
        print(f"\n[roteiro] Interrompido antes do fim: analise nao gerada. Gravacao parcial em {caminho_registro}")


if __name__ == "__main__":
    main()
