"""Cria p2_gray_scott_code.zip com apenas o codigo (sem data/, outputs/, .venv/) para enviar ao RunPod."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "scripts", "configs", "runpod", "README.md", "requirements.txt", ".gitignore", "PROTOCOLO_CHECKLIST.md",
           "paper/main.tex", "paper/references.bib"]
out = ROOT / "p2_gray_scott_code.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for item in INCLUDE:
        p = ROOT / item
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
        for f in files:
            info = zipfile.ZipInfo.from_file(f, f"ATV_NOTA_02/{f.relative_to(ROOT).as_posix()}")
            if f.suffix == ".sh":
                info.external_attr = 0o755 << 16
            with open(f, "rb") as fh:
                z.writestr(info, fh.read(), zipfile.ZIP_DEFLATED)
print(out, f"{out.stat().st_size/1024:.0f} KiB")
