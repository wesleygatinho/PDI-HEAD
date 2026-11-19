"""Filtro One Euro para suavizar os landmarks ao longo do tempo.

Ideia (Casiez et al., 2012): um passa-baixa cuja frequencia de corte cresce
com a velocidade do ponto. Parado -> corte baixo -> remove o tremor (jitter).
Em movimento rapido -> corte alto -> quase nao atrasa o esqueleto.
"""
from __future__ import annotations

import numpy as np


def _alfa(corte_hz: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * np.pi * corte_hz)
    return 1.0 / (1.0 + tau / dt)


class FiltroOneEuro:
    # Padroes ajustados para coordenadas NORMALIZADAS (0 a 1), nao pixels.
    def __init__(self, corte_min: float = 1.0, beta: float = 30.0, corte_derivada: float = 1.0) -> None:
        self.corte_min = corte_min          # Hz: menor = mais suave quando parado
        self.beta = beta                    # maior = menos atraso em movimento
        self.corte_derivada = corte_derivada
        self.reiniciar()

    def reiniciar(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filtra `x` (qualquer formato) observado no instante `t`, em segundos."""
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or t <= self._t:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x.copy()

        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = _alfa(self.corte_derivada, dt)
        dx_suave = a_d * dx + (1.0 - a_d) * self._dx

        corte = self.corte_min + self.beta * np.abs(dx_suave)
        a = _alfa(corte, dt)
        x_suave = a * x + (1.0 - a) * self._x

        self._x, self._dx, self._t = x_suave, dx_suave, t
        return x_suave.copy()
