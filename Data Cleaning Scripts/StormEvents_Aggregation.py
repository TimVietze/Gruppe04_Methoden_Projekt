"""
Combine yearly NOAA Storm Events files into one chronological CSV, applying the
column triage from the housing-price project.

What this drops vs the raw 51-column files:
    - the two narratives (~1 GB of free text, no ML signal without NLP)
    - redundant date breakdowns (kept BEGIN_/END_DATE_TIME instead)
    - sub-county lat/lon and free-text locations (county-level is the target grain)
    - admin/metadata columns (WFO, SOURCE, EPISODE_ID, DATA_SOURCE)
    - rarely-populated INJURIES_INDIRECT/DEATHS_INDIRECT and the four TOR_OTHER_*
    - rows with CZ_TYPE='M' (marine zones, no land housing impact)

DAMAGE_PROPERTY and DAMAGE_CROPS are parsed from "2K"/"1.50M"/"100B" strings
into plain floats. NaN is preserved for missing damage values (not coerced to 0)
so downstream code can distinguish "unreported" from "$0".
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "Methoden Data" / "X_Original Data" / "Weather Data Original" / "Storm Original"
OUT = PROJECT_ROOT / "Methoden Data" / "Weather Data" / "Storm" / "StormEvents_ALL.csv"

YEAR_PATTERN = re.compile(r"StormEvents_(\d{4})\.csv$")

KEEP_COLS = [
    "EVENT_ID",
    "STATE_FIPS",
    "CZ_TYPE",
    "CZ_FIPS",
    "BEGIN_DATE_TIME",
    "END_DATE_TIME",
    "EVENT_TYPE",
    "INJURIES_DIRECT",
    "DEATHS_DIRECT",
    "DAMAGE_PROPERTY",
    "DAMAGE_CROPS",
    "MAGNITUDE",
    "MAGNITUDE_TYPE",
    "FLOOD_CAUSE",
    "CATEGORY",
    "TOR_F_SCALE",
    "TOR_LENGTH",
    "TOR_WIDTH",
]

DROP_CZ_TYPES = {"M"}

DAMAGE_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_DAMAGE_RE = re.compile(r"^\s*([\d.]+)\s*([KMBTkmbt]?)\s*$")


def parse_damage(s: pd.Series) -> pd.Series:
    """Parse '2K' / '1.50M' / '100B' / '0' / '' into floats. Bad strings → NaN."""
    out = np.full(len(s), np.nan, dtype="float64")
    arr = s.astype("string").to_numpy()
    for i, raw in enumerate(arr):
        if raw is pd.NA or raw is None:
            continue
        m = _DAMAGE_RE.match(str(raw))
        if not m:
            continue
        num, suf = m.group(1), m.group(2).upper()
        try:
            out[i] = float(num) * DAMAGE_MULT.get(suf, 1.0)
        except ValueError:
            continue
    return pd.Series(out, index=s.index, dtype="float64")


def year_of(path: Path) -> int:
    m = YEAR_PATTERN.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse year from filename: {path.name}")
    return int(m.group(1))


def load_year_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype=str, usecols=lambda c: c in KEEP_COLS)
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing expected columns {missing}")
    df = df[KEEP_COLS]
    df = df[~df["CZ_TYPE"].isin(DROP_CZ_TYPES)].copy()
    df["DAMAGE_PROPERTY"] = parse_damage(df["DAMAGE_PROPERTY"])
    df["DAMAGE_CROPS"] = parse_damage(df["DAMAGE_CROPS"])
    return df


def main() -> None:
    files = sorted(
        (p for p in SRC_DIR.glob("StormEvents_*.csv") if YEAR_PATTERN.search(p.name)),
        key=year_of,
    )
    if not files:
        raise FileNotFoundError(f"No StormEvents_YYYY.csv files found in {SRC_DIR}")

    print(f"Found {len(files)} files: {year_of(files[0])}..{year_of(files[-1])}")
    print(f"Keeping {len(KEEP_COLS)} cols, dropping CZ_TYPE in {sorted(DROP_CZ_TYPES)}\n")

    frames: list[pd.DataFrame] = []
    raw_total = 0
    kept_total = 0
    for f in files:
        raw = pd.read_csv(f, low_memory=False, dtype=str, usecols=["CZ_TYPE"])
        raw_n = len(raw)
        df = load_year_file(f)
        frames.append(df)
        raw_total += raw_n
        kept_total += len(df)
        print(f"  {year_of(f)}: {raw_n:>7,} raw -> {len(df):>7,} kept")

    combined = pd.concat(frames, ignore_index=True)
    assert len(combined) == kept_total

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False)

    size_mb = OUT.stat().st_size / 1e6
    print(
        f"\nCombined: {len(combined):,} rows ({raw_total - kept_total:,} marine rows dropped),"
        f" {combined.shape[1]} cols"
    )
    print(f"Saved -> {OUT.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
