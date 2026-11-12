"""Download the public data used in the article into realdata/data.

  - three held-out datasets of the collection autogluon/chronos_datasets on the Hugging Face Hub
    (ercot, monash_traffic, monash_australian_electricity; about 84 MB in total);
  - the monthly price-and-demand files of the Australian Energy Market Operator (AEMO) for the regions
    NSW1, QLD1, SA1, TAS1 and VIC1, December 2025 to September 2026 (50 files, about 20 MB).

The data are not redistributed with this repository and remain subject to the terms of their sources.
Files that already exist are skipped.

Usage:  python download_data.py
"""

import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
HF = "https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/{name}/train-00000-of-00001.parquet"
AEMO = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_{region}.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (research data download)"}
MONTHS = ["202512"] + [f"2026{m:02d}" for m in range(1, 10)]
REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "exists"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
                data = r.read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            return f"{len(data) / 1e6:.1f} MB"
        except Exception as e:                                   # retry on network errors
            if attempt == 2:
                raise RuntimeError(f"could not download {url}: {e}") from e
            time.sleep(5 * (attempt + 1))


def main():
    for name in ["ercot", "monash_traffic", "monash_australian_electricity"]:
        print(f"{name:32s}", fetch(HF.format(name=name), os.path.join(DATA, f"{name}.parquet")), flush=True)
    for ym in MONTHS:
        for region in REGIONS:
            fname = f"PRICE_AND_DEMAND_{ym}_{region}.csv"
            print(f"{fname:32s}", fetch(AEMO.format(ym=ym, region=region), os.path.join(DATA, "aemo_2026", fname)),
                  flush=True)
    print("done")


if __name__ == "__main__":
    main()
