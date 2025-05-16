# download_ravdess_subset.py
import os, zipfile, tempfile, shutil, requests
from pathlib import Path
from itertools import product
from tqdm import tqdm

ZIP_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
AUDIO_DIR = Path(__file__).parent / "samples"
AUDIO_DIR.mkdir(exist_ok=True)

EMOTIONS   = ["01", "03", "04", "05"]   # neutral, happy, sad, angry
STATEMENTS = ["01", "02"]                # two sentences
INTENSITY  = ["01", "02"]                # normal / strong
REPETITION = ["01"]                      # first take is enough

def want(path_in_zip: str) -> bool:
    *_, fname = path_in_zip.split("/")      # Actor_xx/03-...
    p = fname.split("-")
    return (
        p[2] in EMOTIONS and
        p[4] in STATEMENTS and
        p[3] in INTENSITY  and
        p[5] in REPETITION
    )

def main():
    # 1. download zip once
    tmp = tempfile.mkdtemp()
    zip_path = Path(tmp) / "ravdess.zip"
    if not zip_path.exists():
        resp = requests.get(ZIP_URL, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f, tqdm(total=int(resp.headers["content-length"]), unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk); bar.update(len(chunk))

    # 2. extract subset
    with zipfile.ZipFile(zip_path) as z:
        wanted = [f for f in z.namelist() if f.endswith(".wav") and want(f)]
        print(f"Extracting {len(wanted)} clips → {AUDIO_DIR}")
        for f in wanted:
            target = AUDIO_DIR / Path(f).name
            if target.exists(): continue
            with z.open(f) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    shutil.rmtree(tmp)

if __name__ == "__main__":
    main()
