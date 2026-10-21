"""Download the LUMS Energy Institute "Residential Energy and Weather Data - Pakistan" (REWD-P) dataset:
59 homes in 6 cities, 1-minute smart-meter data (Nov 2023 - Oct 2024) plus hourly city weather.
Source: https://lei.lums.edu.pk/datasets/residential-energy-and-weather-data-pakistan.html
The page states no license, so the raw files stay out of git (data/raw/lums is ignored); cite LEI when using them."""
import zipfile
from pathlib import Path

import requests

BASE = "https://web.lums.edu.pk/~eig/gallery/datasets/rewdp"
OUT = Path("data/raw/lums")
FILES = ["rewdp_dataset.zip", "weather_dataset.zip", "metadata_ultimate.xlsx"]


def download(name):
    path = OUT / name
    if path.exists():
        print(f"  {name} already downloaded")
        return path
    print(f"  downloading {name} ...")
    with requests.get(f"{BASE}/{name}", stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return path


OUT.mkdir(parents=True, exist_ok=True)
for name in FILES:
    path = download(name)
    if path.suffix == ".zip":
        target = OUT / path.stem
        if not target.exists():
            with zipfile.ZipFile(path) as z:
                for member in z.namelist():          # refuse paths that would escape the target folder
                    if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                        raise SystemExit(f"unsafe path in {name}: {member}")
                z.extractall(target)
        print(f"  {name}: {sum(1 for p in target.rglob('*.csv'))} CSV files in {target}")
