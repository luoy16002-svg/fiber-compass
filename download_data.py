"""Download the public CT/NML pairs; raw data stays outside the Git repository."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

BASE = "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset001_sk-fibers-20250124/"
CASES = [
    ("s1", "00497_01497_03997_256", "v00"),
    ("s1", "00497_02497_02997_256", "v00"),
    ("s1", "00997_02497_02997_256", "v00"),
    ("s1", "08997_02997_02497_256", "v00"),
    ("s1", "10997_02997_02997_256", "v00"),
    ("s5", "03997_01497_03997_256", "v00"),
    ("s5", "06494_01994_03994_512", "v03"),
    ("s5", "06994_00994_04994_512", "v01"),
    ("s5", "07994_01994_05494_512", "v01"),
    ("s5", "07997_02997_05497_256", "v01"),
    ("s5", "14997_01497_01497_256", "v01"),
]


def cases():
    for scroll, suffix, version in CASES:
        z, y, x, size = suffix.split("_")
        name = f"{scroll}_{suffix}"
        nml_scroll = "s1a" if scroll == "s1" else scroll
        yield {
            "case": name, "scroll": scroll, "size": int(size),
            "origin_xyz": [int(x), int(y), int(z)],
            "image": f"imagesTr/{name}_0000.tif",
            "nml": f"nml/fibers_{nml_scroll}_{z}z_{y}y_{x}x_{size}_{version}.nml",
        }


def download(relative):
    target = Path("data") / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        for attempt in range(4):
            try:
                with requests.get(BASE + relative, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    expected = int(response.headers.get("Content-Length", "0"))
                    total = 0
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                            total += len(chunk)
                    if expected and total != expected:
                        raise IOError(f"truncated download: {total} != {expected}")
                temporary.replace(target)
                break
            except (requests.RequestException, IOError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    with target.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    print(f"ready {target.name} {target.stat().st_size:,} bytes", flush=True)
    return {"path": relative, "url": BASE + relative, "bytes": target.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    rows = list(cases())
    files = [row[key] for row in rows for key in ("nml", "image")]
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(download, files))
    Path("results").mkdir(exist_ok=True)
    Path("results/input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
