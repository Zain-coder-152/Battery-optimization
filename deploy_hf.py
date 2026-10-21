"""Deploy the app to a Hugging Face Space (Gradio SDK, which is free; Docker Spaces need a paid plan).

    python deploy_hf.py <username>/<space-name>

Builds the React dashboard, then copies only what the app needs (no research scripts, models or
experiment results) into build/hf-space, a clone of the Space's git repository, and pushes it.
The Space runs app.py, which starts the FastAPI server on port 7860; the app is served at
https://<username>-<space-name>.hf.space.
Git asks for your Hugging Face username and a Write access token the first time you push."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "build" / "hf-space"
FILES = ["api.py", "core.py", "agent.py", "data/locations.json",
         "data/processed/hourly_dataset.csv", *[p.relative_to(ROOT).as_posix() for p in (ROOT / "data/raw").glob("solar_*.csv")]]

SPACE_README = """---
title: Smart Battery Optimizer
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.12"
pinned: false
---

# Smart Battery Optimizer

Hour-by-hour solar battery scheduling for Pakistani homes: pick a city, system size, tariff and
load-shedding schedule, and see the recommended battery schedule and the weekly savings.
"""


# The Space starts this file. It serves our FastAPI app (API + built dashboard) on the Space's port.
APP_PY = """import uvicorn

from api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
"""

# Unpinned on purpose: the Space also installs Gradio, and pip picks versions that suit both.
REQUIREMENTS = """fastapi
uvicorn
pydantic>=2
numpy
pandas
scipy
"""


def git(*args, check=True):
    print("$ git", " ".join(args))
    return subprocess.run(["git", *args], cwd=OUT, check=check)


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python deploy_hf.py <username>/<space-name>")
    space = sys.argv[1].strip().removeprefix("https://huggingface.co/spaces/").strip("/")
    url = f"https://huggingface.co/spaces/{space}"

    print("Building the dashboard...")
    subprocess.run("npm run build", cwd=ROOT / "frontend", shell=True, check=True)

    if not (OUT / ".git").exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", url, str(OUT)], check=True)
    else:
        git("pull", "--ff-only")

    for item in OUT.iterdir():                     # start from a clean copy, keep only git metadata
        if item.name not in (".git", ".gitattributes"):
            shutil.rmtree(item) if item.is_dir() else item.unlink()

    for f in FILES:
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, OUT / f)
    shutil.copytree(ROOT / "frontend" / "dist", OUT / "frontend" / "dist")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    (OUT / "app.py").write_text(APP_PY, encoding="utf-8")
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    git("add", "-A")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=OUT).returncode == 0:
        print("Nothing changed since the last deploy.")
        return
    git("commit", "-m", "Deploy Smart Battery Optimizer")
    git("push")
    name = space.replace("/", "-").replace("_", "-").lower()
    print(f"\nPushed. Build log: {url}\nApp (once built): https://{name}.hf.space")


if __name__ == "__main__":
    main()
