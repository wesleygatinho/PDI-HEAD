"""Baixa um SUBCONJUNTO do gray_scott_reaction_diffusion (The Well) sem baixar os 154 GB.

Os arquivos HDF5 do The Well para este dataset sao contiguos e sem compressao
(t0_fields/A e t0_fields/B com shape [n_traj, 1001, 128, 128] float32). Logo, cada
quadro 128x128 ocupa 64 KiB em um deslocamento conhecido do arquivo e pode ser
obtido com uma requisicao HTTP de faixa (Range). Assim baixamos apenas as
trajetorias sorteadas, com passo temporal configuravel.

Saida: data/raw/{split}_{regime}.npz com A, B [n, T, 128, 128] float32,
traj_idx, steps, F, k.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import h5py
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gsseg.utils import data_dir, load_config  # noqa: E402

BASE = "https://huggingface.co/datasets/{repo}/resolve/main/data/{split}/{fname}"
FRAME_BYTES = 128 * 128 * 4


def resolve(url: str) -> str:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.url


def fetch(session: requests.Session, url_holder: dict, offset: int, nbytes: int) -> bytes:
    for attempt in range(8):
        try:
            r = session.get(
                url_holder["url"],
                headers={"Range": f"bytes={offset}-{offset + nbytes - 1}"},
                timeout=120,
            )
            if r.status_code in (401, 403):  # URL assinada expirou
                url_holder["url"] = resolve(url_holder["orig"])
                continue
            r.raise_for_status()
            if len(r.content) != nbytes:
                raise IOError(f"esperado {nbytes} bytes, recebido {len(r.content)}")
            return r.content
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {e} -> aguardando {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("falha ao baixar faixa apos varias tentativas")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    raw = data_dir(cfg) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    steps = np.arange(0, dcfg["n_time_total"], dcfg["time_stride"])

    for split, n_traj in dcfg["trajectories_per_regime"].items():
        for regime, fname in dcfg["regimes"].items():
            out = raw / f"{split}_{regime}.npz"
            # o sorteio acontece sempre, para a sequencia aleatoria nao depender de arquivos ja existentes
            url = BASE.format(repo=dcfg["hf_repo"], split=split, fname=fname)
            with fsspec.open(url, "rb", block_size=2**16) as fobj, h5py.File(fobj, "r") as h:
                dsA, dsB = h["t0_fields/A"], h["t0_fields/B"]
                n_avail, n_t = dsA.shape[0], dsA.shape[1]
                assert dsA.chunks is None and dsA.compression is None, "dataset nao contiguo"
                offA, offB = dsA.id.get_offset(), dsB.id.get_offset()
                F, k = float(h.attrs["F"]), float(h.attrs["k"])
            traj = np.sort(rng.choice(n_avail, size=n_traj, replace=False))
            if out.exists():
                print(f"[ok] {out.name} ja existe", flush=True)
                continue

            holder = {"orig": url, "url": resolve(url)}
            jobs = []
            for i, tr in enumerate(traj):
                for j, st in enumerate(steps):
                    pos = (int(tr) * n_t + int(st)) * FRAME_BYTES
                    jobs.append((i, j, 0, offA + pos))
                    jobs.append((i, j, 1, offB + pos))
            arr = np.empty((2, n_traj, len(steps), 128, 128), dtype=np.float32)
            t0 = time.time()
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
            session.mount("https://", adapter)

            def work(job):
                i, j, c, off = job
                buf = fetch(session, holder, off, FRAME_BYTES)
                arr[c, i, j] = np.frombuffer(buf, dtype="<f4").reshape(128, 128)

            with ThreadPoolExecutor(args.workers) as ex:
                for n, _ in enumerate(ex.map(work, jobs), 1):
                    if n % 2000 == 0:
                        print(f"  {split}/{regime}: {n}/{len(jobs)} quadros", flush=True)
            np.savez(out, A=arr[0], B=arr[1], traj_idx=traj, steps=steps, F=F, k=k)
            mb = arr.nbytes / 2**20
            print(f"[baixado] {out.name}: {n_traj} traj x {len(steps)} passos "
                  f"({mb:.0f} MiB em {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
