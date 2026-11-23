"""Metricas de avaliacao (secao "2. Metricas" do protocolo da turma).

Segmentacao (por quadro, media sobre quadros):  IoU, Dice, Precision, Recall
Temporal (por trajetoria, media sobre trajetorias):
  AVE  Area Variation Error   = media_t | da_pred(t) - da_gt(t) |, da(t) = a(t) - a(t-1),
       a(t) = fracao do dominio segmentada (em % do dominio)
  CDE  Centroid Displacement Error (px): em cada quadro, as estruturas (componentes conexos,
       unidos atraves das bordas periodicas) tem centroide por media circular; cada centroide do
       GT e pareado ao centroide previsto mais proximo (distancia periodica) e vice-versa; CDE e a
       media simetrica dessas distancias (0 se ambos vazios; penalidade maxima se so um for vazio)
  TC   Temporal Consistency = media_t IoU(M(t), M(t-1));  TCE = |TC_pred - TC_gt|
Computacionais: latencia (ms/quadro) e memoria maxima (MiB) - medidas no treino/inferencia.
Estatistica: IC 95% por bootstrap por trajetoria (reamostra trajetorias, nao quadros,
pois quadros da mesma trajetoria sao correlacionados).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

EPS = 1e-9


def confusion(pred: np.ndarray, gt: np.ndarray):
    """pred, gt: [..., H, W] binarios. Retorna TP, FP, FN por quadro."""
    p, g = pred.astype(bool), gt.astype(bool)
    ax = (-2, -1)
    tp = (p & g).sum(ax).astype(np.float64)
    fp = (p & ~g).sum(ax).astype(np.float64)
    fn = (~p & g).sum(ax).astype(np.float64)
    return tp, fp, fn


def frame_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Metricas por quadro. Classe positiva = estrutura; *_bg = classe fundo (protocolo: metricas por classe)."""
    tp, fp, fn = confusion(pred, gt)
    tn = pred.shape[-2] * pred.shape[-1] - tp - fp - fn
    empty = (tp + fp + fn) == 0  # GT e predicao vazios -> acerto perfeito
    iou = np.where(empty, 1.0, tp / (tp + fp + fn + EPS))
    dice = np.where(empty, 1.0, 2 * tp / (2 * tp + fp + fn + EPS))
    prec = np.where((tp + fp) == 0, np.where(fn == 0, 1.0, 0.0), tp / (tp + fp + EPS))
    rec = np.where((tp + fn) == 0, np.where(fp == 0, 1.0, 0.0), tp / (tp + fn + EPS))
    empty_bg = (tn + fp + fn) == 0
    iou_bg = np.where(empty_bg, 1.0, tn / (tn + fp + fn + EPS))
    dice_bg = np.where(empty_bg, 1.0, 2 * tn / (2 * tn + fp + fn + EPS))
    return {"IoU": iou, "Dice": dice, "Precision": prec, "Recall": rec,
            "IoU_bg": iou_bg, "Dice_bg": dice_bg, "mIoU": (iou + iou_bg) / 2}


def structure_centroids(m: np.ndarray) -> np.ndarray:
    """Centroides [n, 2] (y, x) das estruturas de uma mascara periodica 2D (vetorizado).

    Componentes que se tocam atraves de bordas opostas sao unidos (conectividade periodica)
    via componentes conexos do grafo de adjacencia entre rotulos; centroide = media circular.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros((0, 2))
    a = np.concatenate([lab[0, :], lab[:, 0]])
    b = np.concatenate([lab[-1, :], lab[:, -1]])
    keep = (a > 0) & (b > 0)
    g = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(n + 1, n + 1))
    _, comp = connected_components(g, directed=False)
    roots = np.unique(comp[1:], return_inverse=True)[1] + 1          # rotulos 1..k
    yy, xx = np.nonzero(lab)
    lbl = roots[lab[yy, xx] - 1]
    k = lbl.max()
    H, W = m.shape
    out = []
    for coord, size in ((yy, H), (xx, W)):
        ang = 2 * np.pi * coord / size
        c = np.bincount(lbl, np.cos(ang), minlength=k + 1)[1:]
        s_ = np.bincount(lbl, np.sin(ang), minlength=k + 1)[1:]
        out.append((np.arctan2(s_, c) % (2 * np.pi)) * size / (2 * np.pi))
    return np.stack(out, 1)


def _periodic_dist(a: np.ndarray, b: np.ndarray, H: int, W: int) -> np.ndarray:
    """Distancias periodicas entre todos os pares: a [n,2], b [m,2] -> [n,m]."""
    d = np.abs(a[:, None, :] - b[None, :, :])
    d[..., 0] = np.minimum(d[..., 0], H - d[..., 0])
    d[..., 1] = np.minimum(d[..., 1], W - d[..., 1])
    return np.sqrt((d ** 2).sum(-1))


def centroid_error(pred: np.ndarray, gt: np.ndarray) -> float:
    H, W = gt.shape
    cp, cg = structure_centroids(pred), structure_centroids(gt)
    if len(cp) == 0 and len(cg) == 0:
        return 0.0
    if len(cp) == 0 or len(cg) == 0:
        return float(np.hypot(H / 2, W / 2))
    D = _periodic_dist(cg, cp, H, W)
    return float(0.5 * (D.min(1).mean() + D.min(0).mean()))


def temporal_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred, gt: [N, T, H, W]. Retorna vetores por trajetoria [N]."""
    H, W = pred.shape[-2:]
    ap = pred.mean((-2, -1)) * 100
    ag = gt.mean((-2, -1)) * 100
    ave = np.abs(np.diff(ap, axis=1) - np.diff(ag, axis=1)).mean(1)
    cde = np.array([[centroid_error(pred[i, t], gt[i, t]) for t in range(pred.shape[1])]
                    for i in range(pred.shape[0])]).mean(1)

    def tc(m):
        s = frame_scores(m[:, 1:], m[:, :-1])["IoU"]
        return s.mean(1)

    tcp, tcg = tc(pred), tc(gt)
    return {"AVE": ave, "CDE": cde, "TC": tcp, "TCE": np.abs(tcp - tcg)}


def evaluate(pred: np.ndarray, gt: np.ndarray, temporal: bool = True) -> dict:
    """Retorna metricas por trajetoria (para bootstrap) e por (trajetoria, tempo)."""
    fs = frame_scores(pred, gt)                       # cada [N, T]
    per_traj = {k: v.mean(1) for k, v in fs.items()}  # [N]
    if temporal:
        per_traj.update(temporal_scores(pred, gt))
    return {"per_traj": per_traj, "per_frame": fs}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, groups: np.ndarray | None = None):
    """IC 95% da media por bootstrap; se groups (regime) for dado, reamostra estratificado."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    if groups is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        means = values[idx].mean(1)
    else:
        parts = [np.where(groups == g)[0] for g in np.unique(groups)]
        means = np.zeros(n_boot)
        for p in parts:
            idx = rng.integers(0, len(p), size=(n_boot, len(p)))
            means += values[p][idx].sum(1)
        means /= len(values)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)
