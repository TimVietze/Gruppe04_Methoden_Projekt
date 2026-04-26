"""
Clean the FEMA Disaster Declarations Summary for the housing-price project.

Drops administrative / redundant fields and keeps only what feeds a county-month
feature pipeline. Field-by-field rationale lives in the project notes; the short
version: keep FIPS join keys, the three relevant date columns, the categorical
hazard fields, and the four assistance-program flags as severity proxies.

Statewide rows (fipsCountyCode == '000') are kept and tagged via `is_statewide`
so the downstream feature builder can decide whether to explode them across all
counties of that state or drop them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "Methoden Data" / "X_Original Data" / "Weather Data Original" / "Fema_DisasterDeclarationsSummaries.csv"
OUT = PROJECT_ROOT / "Methoden Data" / "Weather Data" / "Fema" / "Fema_DisasterDeclarations_Cleaned.csv"

# Zillow ZHVI series begin in 2000 — drop earlier FEMA rows by default.
# Set to None to keep the full 1953+ history.
MIN_INCIDENT_YEAR: int | None = 2000

KEEP_COLS = [
    "disasterNumber",
    "fipsStateCode",
    "fipsCountyCode",
    "incidentBeginDate",
    "incidentEndDate",
    "declarationDate",
    "incidentType",
    "designatedIncidentTypes",
    "declarationType",
    "iaProgramDeclared",
    "ihProgramDeclared",
    "paProgramDeclared",
    "hmProgramDeclared",
]

DATE_COLS = ["incidentBeginDate", "incidentEndDate", "declarationDate"]
BOOL_COLS = [
    "iaProgramDeclared",
    "ihProgramDeclared",
    "paProgramDeclared",
    "hmProgramDeclared",
]


def main() -> None:
    print(f"Loading {SRC.name}")
    df = pd.read_csv(SRC, low_memory=False)
    n_in, c_in = df.shape
    print(f"  rows in: {n_in:,}, cols in: {c_in}")

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Source is missing expected columns: {missing}")

    df = df[KEEP_COLS].copy()

    # Preserve leading zeros on FIPS codes — they are join keys for county data.
    df["fipsStateCode"] = df["fipsStateCode"].astype(str).str.zfill(2)
    df["fipsCountyCode"] = df["fipsCountyCode"].astype(str).str.zfill(3)

    df["is_statewide"] = df["fipsCountyCode"] == "000"

    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.date

    for c in BOOL_COLS:
        df[c] = df[c].astype("Int8")

    if MIN_INCIDENT_YEAR is not None:
        cutoff = pd.Timestamp(year=MIN_INCIDENT_YEAR, month=1, day=1).date()
        before = len(df)
        df = df[df["incidentBeginDate"].notna() & (df["incidentBeginDate"] >= cutoff)]
        print(f"  dropped {before - len(df):,} rows with incidentBeginDate before {MIN_INCIDENT_YEAR} (or missing)")

    df = df.sort_values(
        ["incidentBeginDate", "fipsStateCode", "fipsCountyCode", "disasterNumber"],
        kind="stable",
    ).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"  rows out: {len(df):,}, cols out: {df.shape[1]}")
    print(f"  statewide rows flagged: {int(df['is_statewide'].sum()):,}")
    print(f"  unique counties: {df.loc[~df['is_statewide'], ['fipsStateCode', 'fipsCountyCode']].drop_duplicates().shape[0]:,}")
    print(f"Saved -> {OUT.relative_to(PROJECT_ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
