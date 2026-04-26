"""Combine yearly NOAA Storm Events Fatalities files into a single chronological CSV."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "Methoden Data" / "X_Original Data" / "Weather Data Original" / "Storm Fatalities Original"
OUT = PROJECT_ROOT / "Methoden Data" / "Weather Data" / "Storm Fatalities" / "StormEvents_Fatalities_ALL.csv"

YEAR_PATTERN = re.compile(r"StormEvents_Fatalities_(\d{4})\.csv$")


def year_of(path: Path) -> int:
    m = YEAR_PATTERN.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse year from filename: {path.name}")
    return int(m.group(1))


def main() -> None:
    files = sorted(
        (p for p in SRC_DIR.glob("StormEvents_Fatalities_*.csv") if YEAR_PATTERN.search(p.name)),
        key=year_of,
    )
    if not files:
        raise FileNotFoundError(f"No StormEvents_Fatalities_YYYY.csv files found in {SRC_DIR}")

    print(f"Found {len(files)} files: {year_of(files[0])}..{year_of(files[-1])}")

    base_header: list[str] | None = None
    frames: list[pd.DataFrame] = []
    total = 0
    for f in files:
        df = pd.read_csv(f, low_memory=False, dtype=str)
        if base_header is None:
            base_header = list(df.columns)
        elif list(df.columns) != base_header:
            raise ValueError(f"Header mismatch in {f.name}")
        frames.append(df)
        total += len(df)
        print(f"  {year_of(f)}: {len(df):>5,} rows")

    combined = pd.concat(frames, ignore_index=True)
    assert len(combined) == total, "row count mismatch after concat"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False)

    print(f"Combined: {len(combined):,} rows, {combined.shape[1]} cols")
    print(f"Saved -> {OUT.relative_to(PROJECT_ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
