"""Summarize AnimalSpeak (BioLingual training data) into small files for Kaggle.

Outputs:
  data/animalspeak_species.csv  per-species caption counts (pretraining frequency)
  data/animalspeak_ids.csv      Xeno-canto / iNaturalist recording IDs (leakage matching)
  data/animalspeak_meta.json    ID ranges and row counts

Usage: python src/exposure.py [path/to/AnimalSpeak_limited.csv]
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/external/animalspeak.csv"
OUT = ROOT / "data"

XC_ID = r"/XC(\d+)-"
INAT_ID = r"inaturalist\.org/sounds/(\d+)\."


def norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def main(src: Path = SRC) -> None:
    d = pd.read_csv(src, low_memory=False,
                    usecols=["url", "source", "species_common", "species_scientific"])
    d = d.dropna(subset=["species_scientific"])
    d["sci"] = norm_name(d["species_scientific"])
    d["xc_id"] = d["url"].str.extract(XC_ID)[0].astype("Int64")
    d["inat_id"] = d["url"].str.extract(INAT_ID)[0].astype("Int64")

    by_src = (d.assign(src=d["source"].map({"Xeno-canto": "xc", "iNaturalist": "inat"}).fillna("other"))
                .pivot_table(index="sci", columns="src", values="url", aggfunc="size", fill_value=0)
                .add_prefix("n_"))
    species = by_src.assign(n_total=by_src.sum(axis=1))
    species["common_name"] = d.groupby("sci")["species_common"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "")
    species.sort_values("n_total", ascending=False).to_csv(OUT / "animalspeak_species.csv")

    ids = pd.concat([
        pd.DataFrame({"source": "xc", "id": d["xc_id"].dropna()}),
        pd.DataFrame({"source": "inat", "id": d["inat_id"].dropna()}),
    ]).drop_duplicates()
    ids.to_csv(OUT / "animalspeak_ids.csv", index=False)

    meta = {
        "rows": int(len(d)),
        "species": int(species.shape[0]),
        "xc_rows": int((d["source"] == "Xeno-canto").sum()),
        "xc_rows_with_id": int(d["xc_id"].notna().sum()),
        "xc_id_max": int(d["xc_id"].max()),
        "inat_rows_with_id": int(d["inat_id"].notna().sum()),
        "inat_id_max": int(d["inat_id"].max()),
    }
    (OUT / "animalspeak_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SRC)
