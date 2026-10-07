import io
import os
import zipfile
import requests


GEONAMES_DATASET = "cities1000"
GEONAMES_URL = f"https://download.geonames.org/export/dump/{GEONAMES_DATASET}.zip"
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{GEONAMES_DATASET}.txt")


def ensure_dataset(dest_path: str = DATASET_PATH, timeout: int = 60) -> str:
    """Download and cache the GeoNames cities dump if it isn't already local."""
    # 1. If dest_path already exists, just return it — nothing to do.
    if os.path.exists(dest_path):
        return dest_path 
    print(f"Downloading {GEONAMES_DATASET}")

    response = requests.get(GEONAMES_URL, timeout=timeout)
    response.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(response.content))

    with zf.open(f"{GEONAMES_DATASET}.txt") as src, open(dest_path, "wb") as dst:
        dst.write(src.read())

    return dest_path