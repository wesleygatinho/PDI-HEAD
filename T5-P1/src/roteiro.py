"""Roteiros guiados: uma sequencia de etapas cronometradas com a instrucao na tela.

Roteiro.de_modelos  (python espelho.py --roteiro)
    Para cada modelo (Lite, Full, Heavy):
        preparo   -> nao e gravado (tempo para a pessoa se posicionar e o modelo "aquecer")
        parado    -> gravado; mede o tremor da deteccao
        movimento -> gravado; mede o comportamento com a pessoa se mexendo

Roteiro.de_poses    (python espelho.py --gravar-video)
    Um preparo e depois uma sequencia de poses variadas, gravadas em video para
    medir a taxa de acerto (acuracia.py). As poses cobrem os casos dificeis:
    bracos acima da cabeca, oclusao (bracos cruzados), perfil, agachamento,
    equilibrio em uma perna e deslocamento lateral.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Sem acentos: o cv2.putText nao desenha caracteres acentuados.
INSTRUCAO = {
    "preparo": "PREPARE-SE",
    "parado": "FIQUE PARADO",
    "movimento": "MOVIMENTE-SE",
    "bracos_cima": "BRACOS PARA CIMA",
    "bracos_abertos": "BRACOS ABERTOS",
    "bracos_cruzados": "BRACOS CRUZADOS",
    "agachar": "AGACHE DEVAGAR",
    "perfil": "VIRE DE LADO",
    "uma_perna": "FIQUE EM UMA PERNA",
    "andar": "ANDE PARA OS LADOS",
    "livre": "MOVA-SE A VONTADE",
}

# Nome das fases de pose para tabelas e graficos (com acentos).
NOME_POSE = {
    "bracos_cima": "Braços para cima",
    "bracos_abertos": "Braços abertos",
    "bracos_cruzados": "Braços cruzados",
    "agachar": "Agachamento",
    "perfil": "De perfil",
    "uma_perna": "Em uma perna",
    "andar": "Andando de lado",
    "livre": "Movimento livre",
}
POSES = tuple(NOME_POSE)


@dataclass
class Etapa:
    modelo: str
    fase: str
    duracao: float
    titulo: str         # o que aparece antes da instrucao na faixa, ex.: "FULL (2/3)"


class Roteiro:
    def __init__(self, etapas: list[Etapa]) -> None:
        self.etapas = etapas
        self._i = 0
        self._inicio = 0.0
        self.concluido = False

    @classmethod
    def de_modelos(cls, modelos, t_preparo: float = 5.0, t_fase: float = 10.0) -> "Roteiro":
        modelos = list(modelos)
        etapas = [Etapa(m, fase, dur, f"{m.upper()} ({i + 1}/{len(modelos)})")
                  for i, m in enumerate(modelos)
                  for fase, dur in (("preparo", t_preparo), ("parado", t_fase), ("movimento", t_fase))]
        etapas[0].duracao *= 2          # tempo extra para entrar no quadro no comeco
        return cls(etapas)

    @classmethod
    def de_poses(cls, modelo: str, t_pose: float = 5.0, t_preparo: float = 10.0) -> "Roteiro":
        etapas = [Etapa(modelo, "preparo", t_preparo, "VIDEO")]
        etapas += [Etapa(modelo, pose, t_pose, f"POSE {i + 1}/{len(POSES)}") for i, pose in enumerate(POSES)]
        return cls(etapas)

    @property
    def etapa(self) -> Etapa:
        return self.etapas[self._i]

    @property
    def duracao_total(self) -> float:
        return sum(e.duracao for e in self.etapas)

    def iniciar_etapa(self, agora: float) -> None:
        """Zera o relogio da etapa atual (usado apos carregar um modelo, que demora)."""
        self._inicio = agora

    def atualizar(self, agora: float) -> None:
        """Passa para a proxima etapa quando o tempo da atual acabou."""
        if self.concluido or agora - self._inicio < self.etapa.duracao:
            return
        if self._i + 1 >= len(self.etapas):
            self.concluido = True
            return
        self._i += 1
        self._inicio = agora

    def texto(self, agora: float) -> str:
        e = self.etapa
        restante = max(0.0, e.duracao - (agora - self._inicio))
        return f"{e.titulo}  |  {INSTRUCAO[e.fase]}  |  {math.ceil(restante)} s"
