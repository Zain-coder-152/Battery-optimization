"""Deploy the app to a Hugging Face Docker Space.

    python deploy_hf.py <username>/<space-name>

Copies only what the container needs (no research scripts, models or experiment results) into
build/hf-space, which is a clone of the Space's git repository, then commits and pushes it.
Hugging Face builds the Dockerfile and serves the app at https://<username>-<space-name>.hf.space.
Git asks for your Hugging Face username and a Write access token the first time you push."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "build" / "hf-space"
FILES = ["Dockerfile", "requirements.txt", "api.py", "core.py", "agent.py", "data/locations.json",
         "data/processed/hourly_dataset.csv", *[p.relative_to(ROOT).as_posix() for p in (ROOT / "data/raw").glob("solar_*.csv")]]
FRONTEND = ["index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json",
            "tsconfig.app.json", "tsconfig.node.json", "public", "src"]

SPACE_README = """---
title: Smart Battery Optimizer
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Smart Battery Optimizer

Hour-by-hour solar battery scheduling for Pakistani homes: pick a city, system size, tariff and
load-shedding schedule, and see the recommended battery schedule and the weekly savings.
"""


def git(*args, check=True):
    print("$ git", " ".join(args))
    return subprocess.run(["git", *args], cwd=OUT, check=check)


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python deploy_hf.py <username>/<space-name>")
    space = sys.argv[1].strip().removeprefix("https://huggingface.co/spaces/").strip("/")
    url = f"https://huggingface.co/spaces/{space}"

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
    for f in FRONTEND:
        src, dst = ROOT / "frontend" / f, OUT / "frontend" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")

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
