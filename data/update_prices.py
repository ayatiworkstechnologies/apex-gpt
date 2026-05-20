"""
Weekly material price updater.

Fetches configured source URLs from data/city_rates.csv, extracts current
cement/sand/brick/aggregate/steel rates, and writes the refreshed CSV.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DATA_DIR, ".."))
CITY_RATES_PATH = os.path.join(DATA_DIR, "city_rates.csv")

MATERIAL_FIELDS = ["cement", "sand", "brick", "aggregate", "steel"]


def _fetch(url: str, timeout: int = 30) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 ApexSteelEstimator/1.0 (+price updater)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _plain_text(raw_html: str) -> str:
    text = re.sub(r"<(script|style).*?</\1>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text


def _numbers(text: str) -> list[int]:
    return [int(value.replace(",", "")) for value in re.findall(r"\d[\d,]*", text)]


def _table_average(text: str, label: str, unit: str, divisor: int = 1) -> int | None:
    pattern = rf"{label}\s+{unit}\s+(?:₹\s*)?([\d,]+)\s+(?:₹\s*)?([\d,]+)\s+(?:₹\s*)?([\d,]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return round(int(match.group(2).replace(",", "")) / divisor)


def _range_average(text: str, label: str, divisor: int = 1) -> int | None:
    match = re.search(
        rf"{label}.{{0,80}}?₹?\s*([\d,]+)\s*[–-]\s*₹?\s*([\d,]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    low = int(match.group(1).replace(",", ""))
    high = int(match.group(2).replace(",", ""))
    return round(((low + high) / 2) / divisor)


def _range_average_by_unit(text: str, label: str, unit_divisors: dict[str, int]) -> int | None:
    match = re.search(
        rf"{label}.{{0,80}}?₹?\s*([\d,]+)\s*[–-]\s*₹?\s*([\d,]+)\s*per\s+([a-z0-9 ()]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    low = int(match.group(1).replace(",", ""))
    high = int(match.group(2).replace(",", ""))
    unit_text = re.sub(r"\s+", " ", match.group(3).strip().lower())
    for unit_name, divisor in unit_divisors.items():
        if unit_name in unit_text:
            return round(((low + high) / 2) / divisor)
    return None


def _resolve_source_label(url: str, fallback: str = "") -> str:
    host = urlparse(url).netloc.lower()
    if "todaypricerates.com" in host:
        return "todaypricerates.com live fetch"
    if host.startswith("www."):
        host = host[4:]
    return fallback or host or "live source"


def parse_todaypricerates(html_text: str) -> dict:
    text = _plain_text(html_text)
    parsed = {
        "cement": (
            _table_average(text, "Cement", "Bag")
            or _range_average_by_unit(text, "OPC 53 Grade Cement", {"50 kg bag": 1, "bag": 1})
            or _range_average(text, "OPC 53 Grade Cement")
        ),
        "sand": (
            _table_average(text, "M Sand", r"Unit\s+\(100\s*CFT\)", divisor=100)
            or _table_average(text, "River Sand", r"Unit\s+\(100\s*CFT\)", divisor=100)
            or _range_average_by_unit(text, "M-Sand", {"unit": 100, "ton": 25, "metric ton": 25})
            or _range_average_by_unit(text, "River Sand", {"unit": 100, "ton": 25, "metric ton": 25})
            or _range_average(text, "M-Sand", divisor=25)
            or _range_average(text, "River Sand", divisor=25)
        ),
        "brick": (
            _table_average(text, "Clay Bricks", "Piece")
            or _range_average_by_unit(text, "Red Clay Bricks", {"piece": 1, "nos": 1})
            or _range_average(text, "Red Clay Bricks")
        ),
        "aggregate": (
            _table_average(text, "Gravel", "CFT")
            or _range_average_by_unit(text, "20mm Aggregate", {"ton": 33, "metric ton": 33, "cft": 1})
            or _range_average_by_unit(text, "Crushed Stone Jelly", {"ton": 33, "metric ton": 33, "cft": 1})
            or _range_average(text, "20mm Aggregate", divisor=33)
            or _range_average(text, "Crushed Stone Jelly", divisor=33)
        ),
        "steel": (
            _table_average(text, "TMT Steel", "Ton", divisor=1000)
            or _range_average_by_unit(text, "Fe 500", {"metric ton": 1000, "ton": 1000, "kg": 1})
            or _range_average(text, "Fe 500", divisor=1000)
        ),
    }
    return {key: value for key, value in parsed.items() if value}


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_rows(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_notes(row: dict, parsed: dict) -> str:
    old_notes = row.get("notes", "")
    changed = ", ".join(f"{key} Rs {parsed[key]}" for key in MATERIAL_FIELDS if key in parsed)
    stamp = date.today().isoformat()
    source = row.get("source_label") or "configured source"
    return f"Auto-updated {stamp} from {source}. {changed}. Previous notes: {old_notes}".strip()


def update_prices(path: str = CITY_RATES_PATH, dry_run: bool = False, only_verified: bool = False) -> dict:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    fieldnames = list(rows[0].keys())
    updated = []
    skipped = []
    failed = []

    for row in rows:
        city = row.get("city", "")
        url = (row.get("source_url") or "").strip()
        is_verified = str(row.get("verified", "")).lower() == "true"

        if not url or (only_verified and not is_verified):
            skipped.append(city)
            continue

        try:
            raw = _fetch(url)
            parsed = parse_todaypricerates(raw)
        except (URLError, TimeoutError, ValueError, OSError) as exc:
            failed.append({"city": city, "error": str(exc)})
            continue

        if not parsed:
            failed.append({"city": city, "error": "no material prices parsed"})
            continue

        before = {field: row.get(field) for field in MATERIAL_FIELDS}
        for field, value in parsed.items():
            row[field] = str(int(value))
        row["last_updated"] = date.today().isoformat()
        row["source_label"] = _resolve_source_label(url, row.get("source_label", ""))
        row["notes"] = _build_notes(row, parsed)
        updated.append({"city": city, "before": before, "after": {field: row.get(field) for field in MATERIAL_FIELDS}})

    if updated and not dry_run:
        backup = f"{path}.{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak"
        shutil.copy2(path, backup)
        _write_rows(path, rows, fieldnames)

    return {
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "dry_run": dry_run,
    }


def main():
    parser = argparse.ArgumentParser(description="Update weekly construction material rates")
    parser.add_argument("--dry-run", action="store_true", help="Parse sources without writing CSV")
    parser.add_argument("--only-verified", action="store_true", help="Only update verified source rows")
    parser.add_argument("--auto-tune", action="store_true", help="Run model auto-tuning after updating prices")
    args = parser.parse_args()

    summary = update_prices(dry_run=args.dry_run, only_verified=args.only_verified)
    print(json_summary(summary))

    if args.auto_tune and not args.dry_run:
        subprocess.check_call([sys.executable, "-m", "model.auto_tune"], cwd=PROJECT_ROOT)


def json_summary(summary: dict) -> str:
    import json

    compact = {
        "updated": len(summary["updated"]),
        "skipped": len(summary["skipped"]),
        "failed": summary["failed"],
        "dry_run": summary["dry_run"],
        "updated_cities": [item["city"] for item in summary["updated"]],
    }
    return json.dumps(compact, indent=2)


if __name__ == "__main__":
    main()
