import pandas as pd
import requests
from pathlib import Path

CSV_PATH  = "/content/drive/MyDrive/gan-anime/phase4_text2image/all_data.csv"
CACHE_DIR = "/content/drive/MyDrive/gan-anime/phase4_text2image/cache"
MAX_SAMPLES = 10000

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH, usecols=["sample_url", "tags"])
df = df.dropna(subset=["sample_url", "tags"]).head(MAX_SAMPLES * 3)

count = 0
for _, row in df.iterrows():
    if count >= MAX_SAMPLES:
        break
    url        = "https:" + row["sample_url"]
    filename   = url.split("/")[-1]
    cache_path = Path(CACHE_DIR) / filename

    if cache_path.exists():
        count += 1
        continue

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            cache_path.write_bytes(r.content)
            count += 1
    except:
        continue

    if count % 500 == 0:
        print(f"{count}/{MAX_SAMPLES} images téléchargées")

print(f"Done : {count} images dans {CACHE_DIR}")