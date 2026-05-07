from __future__ import annotations

import argparse
import warnings
from io import BytesIO
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
PARQUET_DIR = DATA_DIR / "parquet"

EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(PROJECT_DIR))
from config.constants import PAIRS

YEARS = list(range(2015, 2026))

BASE_URL = "https://www.histdata.com"
PAGE_URL_TEMPLATE = (
    BASE_URL
    + "/download-free-forex-historical-data/"
    + "?/ascii/1-minute-bar-quotes/{pair}/{year}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
}


def _session_label(hour_series: pd.Series) -> pd.Series:
    out = pd.Series("New_York", index=hour_series.index)
    out[(hour_series >= 0) & (hour_series < 7)] = "Asia"
    out[(hour_series >= 7) & (hour_series < 13)] = "London"
    out[(hour_series >= 13) & (hour_series < 17)] = "Overlap"
    return out


def get_download_form_values(
    session: requests.Session, pair: str, year: int
) -> dict:
    url = PAGE_URL_TEMPLATE.format(pair=pair.lower(), year=year)
    response = session.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    form = soup.find("form", {"id": "file_down"})
    if form is None:
        raise ValueError(f"Download form not found for {pair} {year}")

    payload = {}
    for tag in form.find_all("input"):
        name = tag.get("name")
        if name:
            payload[name] = tag.get("value", "")

    required = ["tk", "date", "datemonth", "platform", "timeframe", "fxpair"]
    missing = [k for k in required if not payload.get(k)]
    if missing:
        raise ValueError(f"Missing form fields for {pair} {year}: {missing}")

    return payload


def download_zip_bytes(
    session: requests.Session, pair: str, year: int, retries: int = 3
) -> bytes:
    page_url = PAGE_URL_TEMPLATE.format(pair=pair.lower(), year=year)
    payload = get_download_form_values(session, pair, year)

    for attempt in range(1, retries + 1):
        try:
            response = session.post(
                BASE_URL + "/get.php",
                data=payload,
                headers={**HEADERS, "Origin": BASE_URL, "Referer": page_url},
                timeout=120,
            )
            response.raise_for_status()
            data = response.content

            if len(data) < 100:
                raise ValueError(f"File too small for {pair} {year}")
            if data[:4] != b"PK\x03\x04":
                sample = data[:200].decode("utf-8", errors="ignore")
                raise ValueError(f"Not a ZIP for {pair} {year}: {sample!r}")

            return data

        except Exception as exc:
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"Retry {attempt}/{retries} for {pair} {year} in {wait}s: {exc}")
            time.sleep(wait)


def extract_year_file(zip_bytes: bytes, pair: str, year: int) -> Path:
    pair_dir = EXTRACTED_DIR / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    output_path = pair_dir / f"{pair}_{year}.csv"

    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        data_files = [
            n for n in zf.namelist()
            if n.lower().endswith(".csv") or n.lower().endswith(".txt")
        ]
        if not data_files:
            raise ValueError(f"No CSV/TXT inside ZIP for {pair} {year}")

        with zf.open(data_files[0]) as src, open(output_path, "wb") as dst:
            dst.write(src.read())

    return output_path


def download_and_extract_year(
    session: requests.Session, pair: str, year: int
) -> Path:
    output_path = EXTRACTED_DIR / pair / f"{pair}_{year}.csv"
    if output_path.exists():
        print(f"Skipping {pair} {year}, already downloaded")
        return output_path

    print(f"Downloading {pair} {year}")
    zip_bytes = download_zip_bytes(session, pair, year)
    extracted_path = extract_year_file(zip_bytes, pair, year)
    print(f"Saved {extracted_path}")
    time.sleep(1)
    return extracted_path


def load_histdata_file(file_path: Path, pair: str) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=";",
        header=None,
        names=["timestamp_est", "open", "high", "low", "close", "volume"],
        dtype={
            "timestamp_est": "string",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )

    df["timestamp_est"] = pd.to_datetime(
        df["timestamp_est"],
        format="%Y%m%d %H%M%S",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp_est"]).copy()

    df["timestamp_utc"] = (
        df["timestamp_est"]
        .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    df = df.dropna(subset=["timestamp_utc"]).copy()

    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["pair"] = pair
    df["year"] = df["timestamp_utc"].dt.year
    df["month"] = df["timestamp_utc"].dt.month
    df["session"] = _session_label(df["timestamp_utc"].dt.hour)

    df = df[[
        "timestamp_est",
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "pair",
        "year",
        "month",
        "session",
    ]].copy()

    df = (
        df.sort_values("timestamp_utc")
        .drop_duplicates(subset=["timestamp_utc"])
        .reset_index(drop=True)
    )
    return df


def process_pair_to_parquet(pair: str) -> None:
    pair_dir = EXTRACTED_DIR / pair
    if not pair_dir.exists():
        print(f"No extracted files for {pair}")
        return

    frames = []
    for year in YEARS:
        fp = pair_dir / f"{pair}_{year}.csv"
        if not fp.exists():
            print(f"Missing {pair} {year}")
            continue
        try:
            frames.append(load_histdata_file(fp, pair))
        except Exception as exc:
            print(f"Failed reading {fp}: {exc}")

    if not frames:
        print(f"No usable data for {pair}")
        return

    full_df = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp_utc")
        .drop_duplicates(subset=["timestamp_utc"])
        .reset_index(drop=True)
    )

    output_path = PARQUET_DIR / f"{pair}_2015_2025.parquet"
    full_df.to_parquet(output_path, index=False)

    print(f"Saved parquet for {pair}: {len(full_df):,} rows")
    print(f"Path: {output_path}")
    print(f"Next step: python scripts\\clean_fx_data.py --pair {pair}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download yearly FX bar archives from histdata.com and write them "
            "to data/extracted/ and data/parquet/."
        ),
    )
    parser.add_argument(
        "--insecure", action="store_true",
        help=(
            "Skip TLS certificate verification when talking to histdata.com. "
            "Use this only if histdata.com's certificate is expired or otherwise "
            "untrusted (a known intermittent issue with the upstream site). "
            "The downloaded archive contents are still validated by zipfile, "
            "but the transport is no longer authenticated."
        ),
    )
    args = parser.parse_args()

    http = requests.Session()
    if args.insecure:
        # Suppress the InsecureRequestWarning so logs stay readable; the print
        # below is the user-visible warning.
        from urllib3.exceptions import InsecureRequestWarning
        warnings.simplefilter("ignore", InsecureRequestWarning)
        http.verify = False
        print(
            "WARNING: --insecure is set. TLS verification is disabled for the "
            "duration of this run. Re-enable it as soon as histdata.com's "
            "certificate is restored."
        )

    for pair in PAIRS:
        print(f"\n{'=' * 50}")
        print(f"  {pair}")
        print(f"{'=' * 50}")
        for year in YEARS:
            try:
                download_and_extract_year(http, pair, year)
            except Exception as exc:
                print(f"FAILED {pair} {year}: {exc}")

    print("\nBuilding parquets...")
    for pair in PAIRS:
        try:
            process_pair_to_parquet(pair)
        except Exception as exc:
            print(f"Parquet failed for {pair}: {exc}")

    print("\nDone.")
    print(f"Extracted CSVs : {EXTRACTED_DIR}")
    print(f"Raw parquets   : {PARQUET_DIR}")
    print(f"Next step      : python scripts\\clean_fx_data.py")


if __name__ == "__main__":
    main()