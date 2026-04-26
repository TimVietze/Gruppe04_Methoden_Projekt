"""
Combine yearly NOAA Storm Events files into one chronological CSV, applying
column triage and impact-based row filtering for the housing-price project.

Cuts vs the raw 51-column source files:
    - the two narratives (~1 GB of free text, no ML signal without NLP)
    - redundant date breakdowns (BEGIN_DATE / END_DATE built as YYYY-MM-DD;
      time-of-day removed because the model is monthly)
    - sub-county lat/lon and free-text locations (county-level is the grain)
    - admin/metadata (WFO, SOURCE, EPISODE_ID, EVENT_ID, DATA_SOURCE, STATE name)
    - rarely-populated INJURIES_INDIRECT/DEATHS_INDIRECT and the four TOR_OTHER_*
    - rows with CZ_TYPE='M' (marine zones, no land housing impact)
    - zero-impact rows for low-signal event types only (see LOW_SIGNAL_TYPES):
      damage=0/NaN AND injuries=0 AND deaths=0 AND event is one where the
      occurrence itself carries no signal. Tornado / Flood / Hurricane /
      Drought / Heat zero-impact rows are KEPT — for those categories the
      occurrence is informative even when no damage was reported.

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

READ_COLS = [
    "STATE_FIPS",
    "CZ_TYPE",
    "CZ_FIPS",
    "BEGIN_YEARMONTH",
    "BEGIN_DAY",
    "END_YEARMONTH",
    "END_DAY",
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

OUT_COLS = [
    "STATE_FIPS",
    "CZ_TYPE",
    "CZ_FIPS",
    "BEGIN_DATE",
    "END_DATE",
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

LOW_SIGNAL_TYPES = {
    "Hail",
    "Thunderstorm Wind",
    "Heavy Snow",
    "Heavy Rain",
    "Dense Fog",
    "Lightning",
    "Frost/Freeze",
    "Winter Weather",
}

DAMAGE_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_DAMAGE_RE = re.compile(r"^\s*([\d.]+)\s*([KMBTkmbt]?)\s*$")


def parse_damage(s: pd.Series) -> pd.Series:
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


def build_iso_date(yearmonth: pd.Series, day: pd.Series) -> pd.Series:
    ym = yearmonth.astype("string").str.zfill(6)
    dd = day.astype("string").str.zfill(2)
    return ym.str[:4] + "-" + ym.str[4:6] + "-" + dd


def year_of(path: Path) -> int:
    m = YEAR_PATTERN.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse year from filename: {path.name}")
    return int(m.group(1))


def load_year_file(path: Path) -> tuple[pd.DataFrame, int, int, int]:
    df = pd.read_csv(path, low_memory=False, dtype=str, usecols=lambda c: c in READ_COLS)
    raw_n = len(df)
    missing = [c for c in READ_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing expected columns {missing}")

    marine_mask = df["CZ_TYPE"].isin(DROP_CZ_TYPES)
    n_marine = int(marine_mask.sum())
    df = df[~marine_mask]

    df = df.assign(
        BEGIN_DATE=build_iso_date(df["BEGIN_YEARMONTH"], df["BEGIN_DAY"]),
        END_DATE=build_iso_date(df["END_YEARMONTH"], df["END_DAY"]),
        DAMAGE_PROPERTY=parse_damage(df["DAMAGE_PROPERTY"]),
        DAMAGE_CROPS=parse_damage(df["DAMAGE_CROPS"]),
        INJURIES_DIRECT=pd.to_numeric(df["INJURIES_DIRECT"], errors="coerce").fillna(0).astype("int32"),
        DEATHS_DIRECT=pd.to_numeric(df["DEATHS_DIRECT"], errors="coerce").fillna(0).astype("int32"),
    )

    zero_impact = (
        df["DAMAGE_PROPERTY"].fillna(0).eq(0)
        & df["DAMAGE_CROPS"].fillna(0).eq(0)
        & df["INJURIES_DIRECT"].eq(0)
        & df["DEATHS_DIRECT"].eq(0)
    )
    drop_mask = zero_impact & df["EVENT_TYPE"].isin(LOW_SIGNAL_TYPES)
    n_lowsig = int(drop_mask.sum())
    df = df[~drop_mask]

    return df[OUT_COLS].copy(), raw_n, n_marine, n_lowsig


def main() -> None:
    files = sorted(
        (p for p in SRC_DIR.glob("StormEvents_*.csv") if YEAR_PATTERN.search(p.name)),
        key=year_of,
    )
    if not files:
        raise FileNotFoundError(f"No StormEvents_YYYY.csv files found in {SRC_DIR}")

    print(f"Found {len(files)} files: {year_of(files[0])}..{year_of(files[-1])}")
    print(f"Keeping {len(OUT_COLS)} cols, dropping CZ_TYPE in {sorted(DROP_CZ_TYPES)}")
    print(f"Dropping zero-impact rows for: {sorted(LOW_SIGNAL_TYPES)}\n")

    frames: list[pd.DataFrame] = []
    raw_total = marine_total = lowsig_total = kept_total = 0
    for f in files:
        df, raw_n, n_marine, n_lowsig = load_year_file(f)
        frames.append(df)
        raw_total += raw_n
        marine_total += n_marine
        lowsig_total += n_lowsig
        kept_total += len(df)
        print(
            f"  {year_of(f)}: {raw_n:>7,} raw"
            f"  -{n_marine:>4,} marine"
            f"  -{n_lowsig:>6,} low-sig"
            f"  = {len(df):>7,} kept"
        )

    combined = pd.concat(frames, ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False)

    size_mb = OUT.stat().st_size / 1e6
    print(f"\nCombined: {len(combined):,} rows, {combined.shape[1]} cols")
    print(f"  marine rows dropped:   {marine_total:,}")
    print(f"  low-signal rows dropped: {lowsig_total:,}")
    print(f"Saved -> {OUT.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
